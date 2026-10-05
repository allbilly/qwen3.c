// Persistent direct-register FP16 backend for the matched-checkpoint experiment.
// Matrix multiplies use one or three NPU cores; CPU norms and nonlinear operations reuse the checkout's logic.
#include "fp16_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include "vector_math.h"
#define printf(...) ((void)0)
#define fprintf(...) ((void)0)
#include "rknnops.h"
#undef printf
#undef fprintf

typedef struct Plan {
    struct MemHandles mem;
    int K, N, offset, max_rows, current_rows, bundle_count;
    __fp16 *packed_input;
    float *native_output;
    struct Plan *next;
} Plan;
static Plan **plans;
static int plan_count, device_fd = -1, skip_classifier;
static int core_count = 1;
static int fused;
static int prepared_device;
static int domain_id;
static int classifier_tile=4096,split_down,down_part_count;
static Plan **down_parts;
static float *down_scratch;
static int stream_ffn;
int npu_stream_ffn_enabled(void) { return stream_ffn; }
int npu_core_count(void) { return core_count; }
int npu_uses_fused(void) { return fused; }
typedef struct { unsigned long long calls; double pack_ms, sync_ms, submit_ms, unpack_ms; } Profile;
static Profile profile[10];
static int profiling;
static double profile_time(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static void sync_verified(int fd,uint64_t obj,uint64_t offset,uint64_t size,uint32_t flags) {
    if(!size)return;
    struct rknpu_mem_sync request={.flags=flags,.obj_addr=obj,.offset=offset,.size=size};
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_SYNC,&request)<0) {
        perror("FP16 DMA synchronization failed");exit(1);
    }
}
#define mem_sync sync_verified

static void *native_allocate(int fd,size_t size,uint64_t *dma,uint64_t *object,uint32_t flags,uint32_t *handle) {
    size_t bytes=page_align_size(size);
    struct rknpu_mem_create request={.flags=flags,.size=bytes,.iommu_domain_id=domain_id};
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_CREATE,&request)<0){perror("NPU memory allocation");return NULL;}
    struct rknpu_mem_map mapper={.handle=request.handle};
    if(ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,&mapper)<0){mem_destroy(fd,request.handle,request.obj_addr);return NULL;}
    void *pointer=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,mapper.offset);
    if(pointer==MAP_FAILED){mem_destroy(fd,request.handle,request.obj_addr);return NULL;}
    *dma=request.dma_addr;*object=request.obj_addr;*handle=request.handle;
    return pointer;
}

// Use page-backed cacheable maps for CPU packing/unpacking. The contiguous DMA
// path maps coherent memory and consumes the small CMA pool on this kernel.
// Generate the same one-task command stream, with one reset per model load.
static struct MemHandles create_persistent_regs(int fd,size_t input_size,size_t weight_size,size_t output_size) {
    struct MemHandles mem={0};
    uint32_t flags=RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU|RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;
    if(!prepared_device){npu_reset(fd);prepared_device=1;}
    set_alu_algorithm(11);reset_handle_dma_map();
    mem.tasks_size=1024;mem.input_size=input_size;mem.output_size=output_size;
    mem.weights_alloc_size=REGCMD_RESERVED+align_up_size(weight_size,64);
    uint64_t tasks_dma;
    mem.tasks=native_allocate(fd,mem.tasks_size,&tasks_dma,&mem.tasks_obj,flags|RKNPU_MEM_KERNEL_MAPPING,&mem.tasks_handle);
    mem.weights=native_allocate(fd,mem.weights_alloc_size,&mem.weights_dma,&mem.weights_obj,flags,&mem.weights_handle);
    mem.input=native_allocate(fd,input_size,&mem.input_dma,&mem.input_obj,flags,&mem.input_handle);
    mem.output=native_allocate(fd,output_size,&mem.output_dma,&mem.output_obj,flags,&mem.output_handle);
    if(!mem.tasks||!mem.weights||!mem.input||!mem.output){release_memhandles(fd,&mem);return (struct MemHandles){0};}
    if(!regs.data)initArray(&regs,256);
    regs.size=0;tracked_pc_register_amount_idx=(size_t)-1;reset_reg_tracking();
    regcmd_helper(mem.input_dma,mem.weights_dma,mem.output_dma,input_size,output_size);
    if(reg_task_count==0 && regs.size)finish_current_task();
    disable_reg_tracking();
    if(reg_task_count!=1 || regs.size*8>REGCMD_RESERVED) {
        fprintf(stderr,"Expected one bounded native projection task\n");exit(1);
    }
    // The one-task stream terminates without a next command address.
    if(reg_pc_base_indices[0]!=(size_t)-1)overwrite_reg_value(reg_pc_base_indices[0],0);
    if(reg_pc_amount_indices[0]!=(size_t)-1)overwrite_reg_value(reg_pc_amount_indices[0],PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(reg_task_lengths[0]));
    memset(mem.weights,0,mem.weights_alloc_size);memcpy(mem.weights,regs.data,regs.size*8);
    memset(mem.tasks,0,mem.tasks_size);
    memset(mem.output,0,output_size);
    mem_sync(fd,mem.output_obj,0,output_size,RKNPU_MEM_SYNC_TO_DEVICE);
    struct rknpu_task *task=mem.tasks;
    task[0]=(struct rknpu_task){.op_idx=1,.enable_mask=0xd,.int_mask=0x300,.int_clear=0x1ffff,
        .regcfg_amount=reg_task_lengths[0],.regcmd_addr=mem.weights_dma};
    mem.task_count=1;
    return mem;
}
#ifdef MATCHED_CPU_REFERENCE
static const __fp16 **reference_matrices;
#endif

void npu_skip_classifier(int skip) { skip_classifier = skip; }

// RKLLM captures use token count as width, native FP16 (K/8,M,8) input,
// and native FP32 (N/4,M,4) output. Only register fields change between batches.
static void configure_rows(Plan *p, int rows, int sync) {
    if (p->current_rows == rows) return;
    uint64_t *cmd = p->mem.weights;
    struct rknpu_task *task = p->mem.tasks;
    uint32_t banks = ((size_t)rows * p->K * 2 + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE;
    if (!banks || banks > 4) { fprintf(stderr,"Invalid batch CBUF budget\n"); exit(1); }
    for (uint32_t i = 0; i < task[0].regcfg_amount; i++) {
        uint32_t address = cmd[i] & 65535, value = (cmd[i] >> 16) & 0xffffffff;
        switch (address) {
        case REG_CNA_CONV_CON1: value = CNA_CONV_CON1_PROC_PRECISION(2) | CNA_CONV_CON1_IN_PRECISION(2); break;
        case REG_CNA_CONV_CON2: value = CNA_CONV_CON2_FEATURE_GRAINS(2); break;
        case REG_CNA_DATA_SIZE0: value = CNA_DATA_SIZE0_DATAIN_WIDTH(rows) | CNA_DATA_SIZE0_DATAIN_HEIGHT(1); break;
        case REG_CNA_DATA_SIZE2: value = CNA_DATA_SIZE2_DATAOUT_WIDTH(rows); break;
        case REG_CNA_DATA_SIZE3: value = CNA_DATA_SIZE3_DATAOUT_ATOMICS(rows); break;
        case REG_CNA_CBUF_CON0: value = CNA_CBUF_CON0_WEIGHT_BANK(NPU_CBUF_BANKS-banks) | CNA_CBUF_CON0_DATA_BANK(banks); break;
        case REG_CNA_CBUF_CON1: value = CNA_CBUF_CON1_DATA_ENTRIES(rows * p->K / 32); break;
        case REG_CNA_DMA_CON1: value = CNA_DMA_CON1_LINE_STRIDE(rows * 4); break;
        case REG_CNA_DMA_CON2: value = CNA_DMA_CON2_SURF_STRIDE((uint32_t)(-3 * rows)); break;
        case REG_CNA_FC_DATA_SIZE0: value = CNA_FC_DATA_SIZE0_DMA_WIDTH(rows) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(1); break;
        case REG_CORE_DATAOUT_SIZE_0: value = CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(rows-1) | CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(0); break;
        case REG_DPU_DST_SURF_STRIDE: value = DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(rows); break;
        case REG_DPU_DATA_CUBE_WIDTH: value = DPU_DATA_CUBE_WIDTH_WIDTH(rows-1); break;
        case REG_DPU_DATA_CUBE_HEIGHT: value = DPU_DATA_CUBE_HEIGHT_HEIGHT(0); break;
        case REG_DPU_DATA_CUBE_NOTCH_ADDR: value = 0; break;
        case REG_DPU_WDMA_SIZE_1: value = DPU_WDMA_SIZE_1_WIDTH_WDMA(rows-1) | DPU_WDMA_SIZE_1_HEIGHT_WDMA(0); break;
        case REG_DPU_SURFACE_ADD: value = DPU_SURFACE_ADD_SURF_ADD(rows * 4); break;
        }
        cmd[i] = (cmd[i] & 0xffff00000000ffffull) | ((uint64_t)value << 16);
    }
    if (sync) mem_sync(device_fd, p->mem.weights_obj, 0, task[0].regcfg_amount * 8, RKNPU_MEM_SYNC_TO_DEVICE);
    p->current_rows = rows;
}

static Plan *prepare(const __fp16 *source, int K, int N, int offset) {
    Plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->K = K; p->N = N; p->offset = offset; p->current_rows = -1;
    // Wide shapes require extra CBUF partitioning beyond four data banks.
    // Use the verified 64/32/16 row limits for K=1024/2048/3072.
    p->max_rows = K > 2048 ? 16 : (K > 1024 ? 32 : 64);
    p->packed_input=malloc((size_t)p->max_rows*K*2);
    p->native_output=malloc((size_t)p->max_rows*N*4);
    if(!p->packed_input || !p->native_output) { free(p->packed_input);free(p->native_output);free(p);return NULL; }
    matmul_params = make_matmul_params(1, N, K);
    p->mem = create_persistent_regs(device_fd, (size_t)p->max_rows * K * sizeof(__fp16),
                         (size_t)K * N * sizeof(__fp16), (size_t)p->max_rows * N * sizeof(float));
    if (!p->mem.input || !p->mem.weights || !p->mem.output || !p->mem.tasks) {
        release_memhandles(device_fd, &p->mem);
        free(p->packed_input);free(p->native_output);
        free(p);
        return NULL;
    }
    __fp16 *packed = (__fp16 *)((char *)p->mem.weights + REGCMD_RESERVED);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            packed[weight_fp16(K, n + 1, k + 1)] = source[(size_t)(offset + n) * K + k];
    configure_rows(p, 1, 0);
    mem_sync(device_fd, p->mem.weights_obj, 0, p->mem.weights_alloc_size, RKNPU_MEM_SYNC_TO_DEVICE);
    return p;
}

// Tasks keep absolute command addresses in their own persistent weight BOs.
// The downstream driver selects [2..4] for three cores and [0..1] for two.
static void prepare_bundles(Plan *first) {
    for (Plan *p=first;p;) {
        Plan *q=p;
        struct rknpu_task *tasks=p->mem.tasks;
        int count=0;
        for (;q && count<core_count;q=q->next,count++) tasks[count]=((struct rknpu_task*)q->mem.tasks)[0];
        p->bundle_count=count;
        mem_sync(device_fd,p->mem.tasks_obj,0,count*sizeof(*tasks),RKNPU_MEM_SYNC_TO_DEVICE);
        p=q;
    }
}

static int submit_bundle(Plan *p) {
    int count=p->bundle_count;
    struct rknpu_submit request={
        .flags=RKNPU_JOB_PC|RKNPU_JOB_BLOCK|RKNPU_JOB_PINGPONG,
        .timeout=10000,.task_number=count,.task_obj_addr=p->mem.tasks_obj,
        .core_mask=(1u<<count)-1,.fence_fd=-1,.iommu_domain_id=domain_id,
    };
    int base=count==3?2:0;
    for(int i=0;i<count;i++)request.subcore_task[base+i]=(struct rknpu_subcore_task){.task_start=i,.task_number=1};
    return ioctl(device_fd,DRM_IOCTL_RKNPU_SUBMIT,&request);
}

static int prepare_projection(Plan **head,const __fp16 *source,int K,int N,int classifier) {
    Plan **tail=head;
    int limit=classifier?classifier_tile:4096;
    if(classifier && core_count==3) {
        // Equal N tiles keep all three cores busy in the final vocabulary
        // bundle; 8192 is the register ceiling, not a required tile length.
        int tiles=(N+limit-1)/limit;
        tiles=((tiles+2)/3)*3;
        limit=((N+tiles*32-1)/(tiles*32))*32;
    }
    for(int offset=0;offset<N;) {
        int remaining=N-offset,tile=remaining<limit?remaining:limit;
        if(core_count==3 && !classifier) { tile=((N+95)/96)*32;if(tile>remaining)tile=remaining; }
        // Keep vocabulary tiles below register limits and aligned to 32 channels.
        if(classifier && core_count==1 && remaining>limit && remaining-limit<1024)tile-=1024;
        *tail=prepare(source,K,tile,offset);
        if(!*tail)return 0;
        tail=&(*tail)->next;offset+=tile;
    }
    prepare_bundles(*head);
    return 1;
}

#include "attention_backend.h"

int npu_init_fp16(NpuMatmulContext *ctx, const Config *c, const __fp16 **matrices) {
    memset(ctx, 0, sizeof(*ctx));
    profiling = getenv("NPU_PROFILE") != NULL;
    core_count = getenv("NPU_CORES") ? atoi(getenv("NPU_CORES")) : 1;
    if(core_count!=1 && core_count!=3) { fprintf(stderr,"NPU_CORES must be 1 or 3\n");return 0; }
    fused=getenv("NPU_FUSED")?atoi(getenv("NPU_FUSED")):0;
    domain_id=getenv("NPU_DOMAIN_ID")?atoi(getenv("NPU_DOMAIN_ID")):0;
    classifier_tile=getenv("NPU_CLS_TILE")?atoi(getenv("NPU_CLS_TILE")):4096;
    split_down=getenv("NPU_SPLIT_DOWN")?atoi(getenv("NPU_SPLIT_DOWN")):0;
    stream_ffn=getenv("NPU_STREAM_FFN")?atoi(getenv("NPU_STREAM_FFN")):0;
    if(stream_ffn && (!split_down||!fused||core_count!=3))return 0;
    if(classifier_tile!=4096&&classifier_tile!=8192)return 0;
    if(domain_id<0||domain_id>=16)return 0;
    if (plans || c->dim != 1024 || c->hidden_dim != 3072 ||
        c->n_heads * c->head_dim != 2048 || c->n_kv_heads * c->head_dim != 1024) return 0;
    ctx->dim = c->dim; ctx->hidden_dim = c->hidden_dim; ctx->n_layers = c->n_layers;
    ctx->all_heads_dim = c->n_heads * c->head_dim;
    ctx->kv_dim = c->n_kv_heads * c->head_dim; ctx->vocab_size = c->vocab_size;
    plan_count = 7 * c->n_layers + 1;
#ifdef MATCHED_CPU_REFERENCE
    reference_matrices = malloc(plan_count * sizeof(*reference_matrices));
    if (!reference_matrices) return 0;
    memcpy(reference_matrices, matrices, plan_count * sizeof(*reference_matrices));
    ctx->enabled = 1;
    return 1;
#endif
    if(fused)plan_count+=2*c->n_layers;
    device_fd = getDeviceFd();
    plans = calloc(plan_count, sizeof(*plans));
    if (!plans) { close(device_fd); device_fd = -1; return 0; }
    int ks[] = {c->dim, c->dim, c->dim, ctx->all_heads_dim, c->dim, c->hidden_dim, c->dim, c->dim};
    int ns[] = {ctx->all_heads_dim, ctx->kv_dim, ctx->kv_dim, c->dim, c->hidden_dim, c->dim, c->hidden_dim, c->vocab_size};
    for (int kind = 0; kind < 8; kind++) {
        if(fused && (kind==0 || kind==1 || kind==2 || kind==4 || kind==6))continue;
        int layers = kind == 7 ? 1 : c->n_layers;
        for (int layer = 0; layer < layers; layer++) {
            int index = kind == 7 ? 7 * c->n_layers : kind * c->n_layers + layer;
            if(!prepare_projection(&plans[index],matrices[index],ks[kind],ns[kind],kind==7)) { npu_matmul_shutdown(ctx);return 0; }
        }
    }
    if(fused)for(int combined=0;combined<2;combined++)for(int layer=0;layer<c->n_layers;layer++) {
        int first_kind=combined?4:0,parts=combined?2:3;
        int part_kinds[]={first_kind,combined?6:1,2};
        int N=combined?2*c->hidden_dim:ctx->all_heads_dim+2*ctx->kv_dim;
        __fp16 *joined=malloc((size_t)N*c->dim*2);
        if(!joined){npu_matmul_shutdown(ctx);return 0;}
        size_t offset=0;
        for(int part=0;part<parts;part++) {
            int kind=part_kinds[part];size_t bytes=(size_t)ns[kind]*c->dim*2;
            memcpy((char*)joined+offset,matrices[kind*c->n_layers+layer],bytes);offset+=bytes;
        }
        int index=(7+combined)*c->n_layers+1+layer;
        int ok=prepare_projection(&plans[index],joined,c->dim,N,0);
        free(joined);
        if(!ok){npu_matmul_shutdown(ctx);return 0;}
    }
    if(split_down) {
        // Streaming prefill gives each core a complete output surface for one
        // K partition. The three independent partial sums share one submit.
        down_part_count=stream_ffn?c->n_layers:3*c->n_layers;
        down_parts=calloc(down_part_count,sizeof(*down_parts));
        down_scratch=malloc((size_t)c->seq_len*c->dim*4);
        __fp16 *part=malloc((size_t)c->dim*c->dim*2);
        if(!down_parts||!down_scratch||!part){free(part);npu_matmul_shutdown(ctx);return 0;}
        for(int layer=0;layer<c->n_layers;layer++)for(int block=0;block<3;block++) {
            const __fp16 *source=matrices[5*c->n_layers+layer];
            for(int n=0;n<c->dim;n++)memcpy(part+(size_t)n*c->dim,source+(size_t)n*c->hidden_dim+block*c->dim,c->dim*2);
            if(stream_ffn) {
                Plan **tail=&down_parts[layer];
                while(*tail)tail=&(*tail)->next;
                *tail=prepare(part,c->dim,c->dim,0);
                if(!*tail){free(part);npu_matmul_shutdown(ctx);return 0;}
                if(block==2)prepare_bundles(down_parts[layer]);
            } else if(!prepare_projection(&down_parts[3*layer+block],part,c->dim,c->dim,0)) {
                free(part);npu_matmul_shutdown(ctx);return 0;
            }
        }
        free(part);
    }
    attention_enabled=getenv("NPU_ATTENTION")?atoi(getenv("NPU_ATTENTION")):0;
    if(attention_enabled && (core_count!=3||c->head_dim!=128||!attention_prepare())) { npu_matmul_shutdown(ctx);return 0; }
    ctx->enabled = 1;
    return 1;
}

int npu_matmul_init(NpuMatmulContext *ctx, const Config *c, const TransformerWeights *w) {
    (void)ctx; (void)c; (void)w;
    fprintf(stderr, "This experimental backend requires the verified FP16 tensor container.\n");
    return 0;
}

int npu_matmul_run(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                   const float *input, float *output) {
    return npu_matmul_batch(ctx, kind, layer, input, output, 1);
}

static int run_projection(NpuMatmulContext *ctx,int kind,Plan *first,
                          const float *input,float *output,int rows,int output_stride,
                          float **outputs,const int *dimensions,int output_count,int input_stride);
static int run_split_down(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows);
static void submit_down_parts(NpuMatmulContext *ctx,Plan **down,float *output,int start,int rows);

int npu_matmul_fused(NpuMatmulContext *ctx,int combination,int layer,
                     const float *input,float *output,int rows) {
    if(!ctx->enabled || !fused || combination<0 || combination>1 || layer<0 || layer>=ctx->n_layers)exit(1);
    int index=(7+combination)*ctx->n_layers+1+layer;
    int N=combination?2*ctx->hidden_dim:ctx->all_heads_dim+2*ctx->kv_dim;
    return run_projection(ctx,8+combination,plans[index],input,output,rows,N,NULL,NULL,0,0);
}

int npu_matmul_fused_to(NpuMatmulContext *ctx,int combination,int layer,const float *input,
                        float *a,float *b,float *c,int rows) {
    if(!ctx->enabled||!fused||combination<0||combination>1||layer<0||layer>=ctx->n_layers)exit(1);
    int index=(7+combination)*ctx->n_layers+1+layer;
    int N=combination?2*ctx->hidden_dim:ctx->all_heads_dim+2*ctx->kv_dim;
    float *outputs[]={a,b,c};
    int dimensions[]={combination?ctx->hidden_dim:ctx->all_heads_dim,combination?ctx->hidden_dim:ctx->kv_dim,ctx->kv_dim};
    return run_projection(ctx,8+combination,plans[index],input,NULL,rows,N,outputs,dimensions,combination?2:3,0);
}

int npu_matmul_batch(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                     const float *input, float *output, int rows) {
    if (!ctx->enabled || kind > NPU_MATMUL_WCLS) {
        fprintf(stderr, "FP16 backend unavailable; refusing quantized CPU fallback.\n");
        exit(1);
    }
    if (kind == NPU_MATMUL_WCLS && skip_classifier) return 1;
    int index = kind == NPU_MATMUL_WCLS ? 7 * ctx->n_layers : kind * ctx->n_layers + layer;
#ifdef MATCHED_CPU_REFERENCE
    int ks[] = {ctx->dim, ctx->dim, ctx->dim, ctx->all_heads_dim, ctx->dim, ctx->hidden_dim, ctx->dim, ctx->dim};
    int ns[] = {ctx->all_heads_dim, ctx->kv_dim, ctx->kv_dim, ctx->dim, ctx->hidden_dim, ctx->dim, ctx->hidden_dim, ctx->vocab_size};
    int K = ks[kind], N = ns[kind];
    if (index < 0 || index >= plan_count) exit(1);
    __fp16 half_input[(size_t)rows*K];
    for (int k = 0; k < rows*K; k++) half_input[k] = (__fp16)input[k];
    const __fp16 *weight = reference_matrices[index];
    // Ground truth uses the same FP16 operands and double accumulation, with no integer quantizer.
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < rows; row++) for (int n = 0; n < N; n++) {
        double sum = 0;
        for (int k = 0; k < K; k++) sum += (double)half_input[(size_t)row*K+k] * (double)weight[(size_t)n * K + k];
        output[(size_t)row*N+n] = (float)sum;
    }
    ctx->cpu_ops++;
    return 1;
#endif
    if(split_down && kind==NPU_MATMUL_W2 && rows>1)return run_split_down(ctx,layer,input,output,rows);
    if (index < 0 || index >= plan_count || !plans[index]) exit(1);
    int output_dimensions[] = {ctx->all_heads_dim,ctx->kv_dim,ctx->kv_dim,ctx->dim,ctx->hidden_dim,ctx->dim,ctx->hidden_dim,ctx->vocab_size};
    int output_stride = output_dimensions[kind];
    return run_projection(ctx,kind,plans[index],input,output,rows,output_stride,NULL,NULL,0,0);
}

static int run_projection(NpuMatmulContext *ctx,int kind,Plan *first,
                          const float *input,float *output,int rows,int output_stride,
                          float **outputs,const int *dimensions,int output_count,int input_stride) {
    if(!first || rows<1)exit(1);
    if(!input_stride)input_stride=first->K;
    for (Plan *p = first; p;) {
      Plan *group[3],*next=p;
      for(int i=0;i<p->bundle_count;i++){group[i]=next;next=next->next;}
      for (int start=0; start<rows; start+=p->max_rows) {
        int m = rows-start < p->max_rows ? rows-start : p->max_rows;
        for(int i=0;i<p->bundle_count;i++)configure_rows(group[i], m, 1);
        double a=profiling?profile_time():0;
        __fp16 *in = p->packed_input;
        #pragma omp parallel for if(m>=16)
        for (int k=0; k<p->K; k+=8) for(int row=0; row<m; row++)
        {
            const float *source=input+(size_t)(start+row)*input_stride+k;
            float16x8_t half=vcombine_f16(vcvt_f16_f32(vld1q_f32(source)),vcvt_f16_f32(vld1q_f32(source+4)));
            vst1q_f16(in+(size_t)(k/8)*m*8+row*8,half);
        }
        for(int i=0;i<p->bundle_count;i++)memcpy(group[i]->mem.input,in,(size_t)m*p->K*2);
        double b=profiling?profile_time():0;
        for(int i=0;i<p->bundle_count;i++)mem_sync(device_fd, group[i]->mem.input_obj, 0, (size_t)m*p->K*2, RKNPU_MEM_SYNC_TO_DEVICE);
        double c=profiling?profile_time():0;
        if (submit_bundle(p) < 0) {
            fprintf(stderr, "NPU submission failed; benchmark invalid.\n");
            exit(1);
        }
        double d=profiling?profile_time():0;
        for(int i=0;i<p->bundle_count;i++)mem_sync(device_fd, group[i]->mem.output_obj, 0, (size_t)m*group[i]->N*4, RKNPU_MEM_SYNC_FROM_DEVICE);
        double e=profiling?profile_time():0;
        #pragma omp parallel for if(m>=32 && output_stride>=4096)
        for(int i=0;i<p->bundle_count;i++) {
            Plan *q=group[i];
            const float *native=q->mem.output;
            if (m==1 && !output_count) memcpy(output+(size_t)start*output_stride+q->offset,native,(size_t)q->N*4);
            else if(m==1) {
                int range=0;
                for(int target=0;target<output_count;target++) {
                    int begin=q->offset>range?q->offset:range;
                    int end=q->offset+q->N<range+dimensions[target]?q->offset+q->N:range+dimensions[target];
                    if(end>begin)memcpy(outputs[target]+(size_t)start*dimensions[target]+begin-range,native+begin-q->offset,(end-begin)*4);
                    range+=dimensions[target];
                }
            }
            else {
                // Bulk read the DMA map, then transpose small row tiles in cache.
                // Page-backed maps are already cacheable; read them directly.
                for(int row_block=0;row_block<m;row_block+=4)for(int n=0;n<q->N;n+=16) {
                    int target=0,column=q->offset+n;
                    if(output_count)while(column>=dimensions[target])column-=dimensions[target++];
                    for(int row=row_block;row<m && row<row_block+4;row++) {
                        float *destination=output_count?outputs[target]+(size_t)(start+row)*dimensions[target]+column:
                            output+(size_t)(start+row)*output_stride+q->offset+n;
                        // Complete a 64-byte destination line before moving to another row.
                        for(int channel=0;channel<16;channel+=4)
                            vst1q_f32(destination+channel,vld1q_f32(native+(size_t)((n+channel)/4)*m*4+row*4));
                    }
                }
            }
        }
        if(profiling) {
            Profile *pstat=&profile[kind];pstat->calls++;
            pstat->pack_ms+=b-a;pstat->sync_ms+=(c-b)+(e-d);
            pstat->submit_ms+=d-c;pstat->unpack_ms+=profile_time()-e;
        }
      }
      p=next;
    }
    ctx->npu_ops++;
    return 1;
}

static int run_split_down(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows) {
    if(stream_ffn) {
        Plan *down[3],*next=down_parts[layer];
        for(int i=0;i<3;i++){down[i]=next;next=next->next;}
        for(int start=0;start<rows;start+=64) {
            int m=rows-start<64?rows-start:64;
            for(int i=0;i<3;i++)configure_rows(down[i],m,1);
            #pragma omp parallel for collapse(2)
            for(int block=0;block<3;block++)for(int k=0;k<ctx->dim;k+=8)for(int row=0;row<m;row++) {
                const float *s=input+(size_t)(start+row)*ctx->hidden_dim+block*ctx->dim+k;
                vst1q_f16(down[block]->packed_input+(size_t)(k/8)*m*8+row*8,
                          vcombine_f16(vcvt_f16_f32(vld1q_f32(s)),vcvt_f16_f32(vld1q_f32(s+4))));
            }
            submit_down_parts(ctx,down,output,start,m);
        }
        ctx->npu_ops++;
        return 1;
    }
    for(int block=0;block<3;block++) {
        float *destination=block?down_scratch:output;
        run_projection(ctx,NPU_MATMUL_W2,down_parts[3*layer+block],input+block*ctx->dim,
                       destination,rows,ctx->dim,NULL,NULL,0,ctx->hidden_dim);
        if(block) {
            #pragma omp parallel for
            for(int i=0;i<rows*ctx->dim;i++)output[i]+=down_scratch[i];
        }
    }
    return 1;
}

static void submit_down_parts(NpuMatmulContext *ctx,Plan **down,float *output,int start,int rows) {
    for(int i=0;i<3;i++) {
        memcpy(down[i]->mem.input,down[i]->packed_input,(size_t)rows*ctx->dim*2);
        mem_sync(device_fd,down[i]->mem.input_obj,0,(size_t)rows*ctx->dim*2,RKNPU_MEM_SYNC_TO_DEVICE);
    }
    if(submit_bundle(down[0])<0){perror("NPU streaming down");exit(1);}
    for(int i=0;i<3;i++)mem_sync(device_fd,down[i]->mem.output_obj,0,(size_t)rows*ctx->dim*4,RKNPU_MEM_SYNC_FROM_DEVICE);
    for(int n=0;n<ctx->dim;n+=4)for(int row=0;row<rows;row++) {
        size_t index=(size_t)(n/4)*rows*4+row*4;
        float32x4_t value=vld1q_f32((const float*)down[0]->mem.output+index);
        // Preserve the original FP32 partial-sum addition order.
        for(int block=1;block<3;block++)value=vaddq_f32(value,vld1q_f32((const float*)down[block]->mem.output+index));
        vst1q_f32(output+(size_t)(start+row)*ctx->dim+n,value);
    }
}

int npu_stream_ffn_prefill(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows) {
    if(!stream_ffn||rows<2||rows>512||layer<0||layer>=ctx->n_layers)exit(1);
    Plan *ff[3],*p=plans[8*ctx->n_layers+1+layer];
    for(int i=0;i<3;i++){ff[i]=p;p=p->next;}
    for(int start=0;start<rows;start+=64) {
        int m=rows-start<64?rows-start:64;
        for(int i=0;i<3;i++)configure_rows(ff[i],m,1);
        attention_pack(ff[0],input+(size_t)start*ctx->dim,ctx->dim,m);
        for(int i=1;i<3;i++) {
            memcpy(ff[i]->mem.input,ff[0]->packed_input,(size_t)m*ctx->dim*2);
            mem_sync(device_fd,ff[i]->mem.input_obj,0,(size_t)m*ctx->dim*2,RKNPU_MEM_SYNC_TO_DEVICE);
        }
        if(submit_bundle(ff[0])<0){perror("NPU streaming gate/up");exit(1);}
        for(int i=0;i<3;i++)mem_sync(device_fd,ff[i]->mem.output_obj,0,(size_t)m*ff[i]->N*4,RKNPU_MEM_SYNC_FROM_DEVICE);
        Plan *down[3],*next=down_parts[layer];
        for(int i=0;i<3;i++){down[i]=next;next=next->next;configure_rows(down[i],m,1);}
        #pragma omp parallel for collapse(2)
        for(int block=0;block<3;block++)
            for(int k=0;k<1024;k+=8)for(int row=0;row<m;row++) {
                int channel=block*1024+k;
                const float *gate=ff[channel/2048]->mem.output;
                const float *up=ff[(channel+3072)/2048]->mem.output;
                size_t g=(size_t)((channel%2048)/4)*m*4+row*4;
                size_t u=(size_t)(((channel+3072)%2048)/4)*m*4+row*4;
                float32x4_t low=qwen_swiglu4(vld1q_f32(gate+g),vld1q_f32(up+u));
                float32x4_t high=qwen_swiglu4(vld1q_f32(gate+g+m*4),vld1q_f32(up+u+m*4));
                vst1q_f16(down[block]->packed_input+(size_t)(k/8)*m*8+row*8,
                          vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high)));
            }
        submit_down_parts(ctx,down,output,start,m);
    }
    ctx->npu_ops+=2;
    return 1;
}

void npu_matmul_reset_stats(NpuMatmulContext *ctx) {
    ctx->npu_ops = ctx->cpu_ops = 0; memset(profile,0,sizeof(profile));
}

void npu_profile_dump(const char *label) {
    if(!profiling)return;
    for(int i=0;i<10;i++) {
        Profile *p=&profile[i];
        fprintf(stderr,"{\"profile\":\"%s\",\"kind\":%d,\"calls\":%llu,\"pack_ms\":%.3f,\"sync_ms\":%.3f,\"submit_ms\":%.3f,\"unpack_ms\":%.3f}\n",label,i,p->calls,p->pack_ms,p->sync_ms,p->submit_ms,p->unpack_ms);
    }
}
void npu_profile_clear(void) { memset(profile,0,sizeof(profile)); }

void npu_matmul_shutdown(NpuMatmulContext *ctx) {
    attention_release();
    if(down_parts)for(int i=0;i<down_part_count;i++) {
        Plan *p=down_parts[i];
        while(p){Plan *next=p->next;release_memhandles(device_fd,&p->mem);free(p->packed_input);free(p->native_output);free(p);p=next;}
    }
    free(down_parts);free(down_scratch);down_parts=NULL;down_scratch=NULL;down_part_count=0;
#ifdef MATCHED_CPU_REFERENCE
    free(reference_matrices); reference_matrices = NULL;
#endif
    if (plans) {
        for (int i = 0; i < plan_count; i++) {
            Plan *p = plans[i];
            while (p) {
                Plan *next = p->next;
                release_memhandles(device_fd, &p->mem);
                free(p->packed_input);free(p->native_output);
                free(p);
                p = next;
            }
        }
        free(plans); plans = NULL;
    }
    if (device_fd >= 0) close(device_fd);
    device_fd = -1;
    prepared_device=0;
    freeArray(&regs);
    memset(ctx, 0, sizeof(*ctx));
}
