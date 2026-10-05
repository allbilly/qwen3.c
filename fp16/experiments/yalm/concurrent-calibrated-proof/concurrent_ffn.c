// Isolated concurrency experiment. The native task generator is unchanged.
// Only this thread allocates, synchronizes, submits, or releases NPU objects.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <pthread.h>
#include <dlfcn.h>
#include "npu_backend.c"
#include "concurrent_ffn.h"
#include "linear_source.h"
static int ff_cpu,ff_gpu,ff_npu,ff_layers,ff_position,ff_mode,ff_enabled,ff_events;
static Plan **ff_plans;
static int ff_native_mode;
static double ff_activation_pack_ms,ff_down_wall_ms;
static float *ff_joined,*ff_cpu_result,*ff_gpu_result,*ff_rounded,*ff_gate,*ff_up;
static float **ff_cpu_weights;
static __fp16 *ff_half;
static void *ff_blas;
typedef void (*FfSgemm)(int,int,int,int64_t,int64_t,int64_t,float,const float *,int64_t,const float *,int64_t,float,float *,int64_t);
static FfSgemm ff_sgemm;
static pthread_t ff_thread;
static pthread_mutex_t ff_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ff_condition=PTHREAD_COND_INITIALIZER;
static int ff_pending,ff_stop,ff_job_rows,ff_job_layer;
static double ff_cpu_start,ff_cpu_end;
static cl_context ff_context;
static cl_command_queue ff_queue;
static cl_program ff_program;
static cl_kernel ff_gemv,ff_reduce,ff_gemm;
static cl_mem *ff_gpu_weights,ff_input,ff_output,ff_partials;
typedef struct {
    uint64_t calls,cpu_calls,gpu_calls,npu_calls,rows,gpu_completed_during_npu,logical_cpu_ops,gpu_running_across_submit,cpu_spans_submit,all_three_spans_submit,gpu_calibrated_contains_submit,all_three_calibrated_contains_submit,valid_clock_maps;
    double wall_ms,round_ms,dispatch_ms,npu_ms,cpu_ms,cpu_npu_overlap_ms,join_wait_ms,merge_ms;
    double gpu_upload_ms,gpu_kernel_ms,gpu_download_ms;
} FfProfile;
static FfProfile ff_profile;
static double ff_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void ff_check(cl_int rc,const char *name){if(rc!=CL_SUCCESS){fprintf(stderr,"Concurrent Mali %s failed: %d\n",name,rc);exit(1);}}
static cl_mem ff_alloc(size_t size){cl_int e;cl_mem b=clCreateBuffer(ff_context,CL_MEM_READ_WRITE,size,NULL,&e);ff_check(e,"allocation");return b;}
static void ff_memarg(cl_kernel k,int index,cl_mem mem){ff_check(clSetKernelArg(k,index,sizeof(mem),&mem),"buffer argument");}
static void ff_intarg(cl_kernel k,int index,int value){ff_check(clSetKernelArg(k,index,sizeof(value),&value),"integer argument");}
static double ff_event_ms(cl_event event){cl_ulong a,b;ff_check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(a),&a,NULL),"event start");ff_check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(b),&b,NULL),"event end");if(b<a)exit(1);return (b-a)/1e6;}
// Marker END lies between host timestamps captured before enqueue and after wait.
// Both marker bounds must admit the same clock offset; widen by 10us for rounding.
static void ff_clock_bounds(double *low,double *high){
    cl_event marker;double begin=ff_now();
    ff_check(clEnqueueMarkerWithWaitList(ff_queue,0,NULL,&marker),"clock marker");
    ff_check(clWaitForEvents(1,&marker),"clock marker wait");double end=ff_now();
    cl_ulong device_end;ff_check(clGetEventProfilingInfo(marker,CL_PROFILING_COMMAND_END,sizeof(device_end),&device_end,NULL),"clock marker end");clReleaseEvent(marker);
    *low=begin-device_end/1e6-.01;*high=end-device_end/1e6+.01;
}
static void *ff_cpu_worker(void *unused){
    (void)unused;
    pthread_mutex_lock(&ff_mutex);
    for(;;){
        while(!ff_pending&&!ff_stop)pthread_cond_wait(&ff_condition,&ff_mutex);
        if(ff_stop)break;
        int rows=ff_job_rows,layer=ff_job_layer,N=2*ff_cpu;
        pthread_mutex_unlock(&ff_mutex);
        ff_cpu_start=ff_now();
        const float *w=ff_cpu_weights[layer];
        if(rows==1){
            for(int n=0;n<N;n++){
                float32x4_t low=vdupq_n_f32(0),high=low;
                for(int k=0;k<1024;k+=8){low=vfmaq_f32(low,vld1q_f32(ff_rounded+k),vld1q_f32(w+(size_t)n*1024+k));high=vfmaq_f32(high,vld1q_f32(ff_rounded+k+4),vld1q_f32(w+(size_t)n*1024+k+4));}
                ff_cpu_result[n]=vaddvq_f32(vaddq_f32(low,high));
            }
        }else ff_sgemm(101,111,112,rows,N,1024,1,ff_rounded,1024,w,1024,0,ff_cpu_result,N);
        ff_cpu_end=ff_now();
        pthread_mutex_lock(&ff_mutex);ff_pending=0;pthread_cond_signal(&ff_condition);
    }
    pthread_mutex_unlock(&ff_mutex);return NULL;
}
void ffn_phase(int position){ff_position=position;}
int ffn_active(void){return ff_enabled&&((ff_position==0&&(ff_mode&1))||(ff_position>0&&(ff_mode&2)));}
int ffn_init(NpuMatmulContext *ctx,const Config *c,const __fp16 **matrices){
    ff_cpu=getenv("FFN_CPU_CHANNELS")?atoi(getenv("FFN_CPU_CHANNELS")):0;
    ff_gpu=getenv("FFN_GPU_CHANNELS")?atoi(getenv("FFN_GPU_CHANNELS")):0;
    ff_mode=getenv("FFN_MODE")?atoi(getenv("FFN_MODE")):3;
    ff_npu=3072-ff_cpu-ff_gpu;ff_layers=c->n_layers;ff_events=getenv("FFN_PROFILE")!=NULL;
    if(!ff_cpu&&!ff_gpu)return 1;
    // 96-channel slices give three equal native tasks, each aligned to 32.
    if(!ctx->enabled||c->dim!=1024||c->hidden_dim!=3072||core_count!=3||!fused||!stream_ffn||ff_cpu<0||ff_gpu<0||ff_npu<=0||ff_cpu%96||ff_gpu%96||ff_mode<1||ff_mode>3)return 0;
    ff_plans=calloc(ff_layers,sizeof(*ff_plans));
    ff_joined=malloc((size_t)512*2*ff_npu*4);ff_gate=malloc((size_t)512*3072*4);ff_up=malloc((size_t)512*3072*4);
    ff_rounded=malloc((size_t)512*1024*4);ff_half=malloc((size_t)512*1024*2);
    ff_cpu_result=malloc((size_t)512*2*(ff_cpu?ff_cpu:1)*4);ff_gpu_result=malloc((size_t)512*2*(ff_gpu?ff_gpu:1)*4);
    if(!ff_plans||!ff_joined||!ff_gate||!ff_up||!ff_rounded||!ff_half||!ff_cpu_result||!ff_gpu_result)return 0;
    __fp16 *joined=malloc((size_t)2*ff_npu*1024*2);if(!joined)return 0;
    for(int layer=0;layer<ff_layers;layer++){
        memcpy(joined,matrices[4*ff_layers+layer],(size_t)ff_npu*1024*2);
        memcpy(joined+(size_t)ff_npu*1024,matrices[6*ff_layers+layer],(size_t)ff_npu*1024*2);
        if(!prepare_projection(&ff_plans[layer],joined,1024,2*ff_npu,0)){free(joined);return 0;}
        Plan *p=ff_plans[layer];int tasks=0;
        for(;p;p=p->next){if(p->N!=2*ff_npu/3)return 0;tasks++;}if(tasks!=3||ff_plans[layer]->bundle_count!=3)return 0;
    }free(joined);
    if(ff_cpu){
        ff_blas=dlopen("/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs/libopenblas64_p-r0-17488984.3.23.dev.so",RTLD_NOW|RTLD_LOCAL);
        if(!ff_blas){fprintf(stderr,"%s\n",dlerror());return 0;}
        ff_sgemm=(FfSgemm)dlsym(ff_blas,"cblas_sgemm64_");void (*threads)(int)=dlsym(ff_blas,"openblas_set_num_threads64_");
        if(!ff_sgemm||!threads)return 0;threads(1);
        ff_cpu_weights=calloc(ff_layers,sizeof(*ff_cpu_weights));if(!ff_cpu_weights)return 0;
        for(int layer=0;layer<ff_layers;layer++){
            ff_cpu_weights[layer]=malloc((size_t)2*ff_cpu*1024*4);if(!ff_cpu_weights[layer])return 0;
            for(int part=0;part<2;part++){const __fp16 *w=matrices[(part?6:4)*ff_layers+layer]+(size_t)(ff_npu+ff_gpu)*1024;
                for(size_t i=0;i<(size_t)ff_cpu*1024;i++)ff_cpu_weights[layer][(size_t)part*ff_cpu*1024+i]=w[i];}
        }
        if(pthread_create(&ff_thread,NULL,ff_cpu_worker,NULL))return 0;
    }
    if(ff_gpu){
        cl_platform_id platforms[8];cl_uint np;cl_device_id device=NULL;ff_check(clGetPlatformIDs(8,platforms,&np),"platforms");
        for(cl_uint p=0;p<np&&!device;p++){cl_device_id devices[8];cl_uint nd=0;cl_int e=clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,8,devices,&nd);if(e==CL_DEVICE_NOT_FOUND)continue;ff_check(e,"devices");for(cl_uint d=0;d<nd;d++){char name[256];ff_check(clGetDeviceInfo(devices[d],CL_DEVICE_NAME,sizeof(name),name,NULL),"name");if(strstr(name,"Mali")){device=devices[d];break;}}}
        if(!device)return 0;
        cl_int e;ff_context=clCreateContext(NULL,1,&device,NULL,NULL,&e);ff_check(e,"context");ff_queue=clCreateCommandQueue(ff_context,device,ff_events?CL_QUEUE_PROFILING_ENABLE:0,&e);ff_check(e,"queue");
        const char *source=linear_source;size_t len=strlen(source);ff_program=clCreateProgramWithSource(ff_context,1,&source,&len,&e);ff_check(e,"program");
        e=clBuildProgram(ff_program,1,&device,"-cl-std=CL1.2",NULL,NULL);if(e){char log[16384];clGetProgramBuildInfo(ff_program,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL);fprintf(stderr,"%s\n",log);}ff_check(e,"build");
        ff_gemv=clCreateKernel(ff_program,"linear_gemv",&e);ff_check(e,"gemv");ff_reduce=clCreateKernel(ff_program,"linear_reduce",&e);ff_check(e,"reduce");ff_gemm=clCreateKernel(ff_program,"linear_gemm",&e);ff_check(e,"gemm");
        int N=2*ff_gpu;ff_gpu_weights=calloc(ff_layers,sizeof(*ff_gpu_weights));__fp16 *transposed=malloc((size_t)N*1024*2);if(!ff_gpu_weights||!transposed)return 0;
        for(int layer=0;layer<ff_layers;layer++){
            for(int k=0;k<1024;k++)for(int n=0;n<N;n++){int part=n/ff_gpu,channel=n%ff_gpu+ff_npu;transposed[(size_t)k*N+n]=matrices[(part?6:4)*ff_layers+layer][(size_t)channel*1024+k];}
            ff_gpu_weights[layer]=ff_alloc((size_t)N*1024*2);ff_check(clEnqueueWriteBuffer(ff_queue,ff_gpu_weights[layer],CL_TRUE,0,(size_t)N*1024*2,transposed,0,NULL,NULL),"weights upload");
        }free(transposed);ff_input=ff_alloc((size_t)512*1024*2);ff_output=ff_alloc((size_t)512*N*4);ff_partials=ff_alloc((size_t)8*N*4);
        char name[256];ff_check(clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,NULL),"device name");fprintf(stderr,"{\"event\":\"ffn_gpu_init\",\"device\":\"%s\",\"weight_bytes\":%zu,\"projection_count\":%d,\"event_profiling\":%s}\n",name,(size_t)ff_layers*N*1024*2,ff_layers,ff_events?"true":"false");
    }
    ff_enabled=1;
    fprintf(stderr,"{\"event\":\"ffn_split_init\",\"cpu_channels\":%d,\"gpu_channels\":%d,\"npu_channels\":%d,\"mode\":%d,\"cpu_threads\":1,\"npu_tasks\":3,\"prefill_block_rows\":64,\"fp16_operands\":true,\"cpu_resident_weight_bytes\":%zu,\"extra_npu_weight_bytes\":%zu}\n",ff_cpu,ff_gpu,ff_npu,ff_mode,(size_t)ff_layers*2*ff_cpu*1024*4,(size_t)ff_layers*2*ff_npu*1024*2);
    return 1;
}
int ffn_gate_up(NpuMatmulContext *ctx,int layer,const float *input,float *gate,float *up,int rows){
    if(!ff_enabled||layer<0||layer>=ff_layers||rows<1||rows>512)exit(1);
    double begin=ff_now();
    for(int i=0;i<rows*1024;i+=8){float16x8_t h=vcombine_f16(vcvt_f16_f32(vld1q_f32(input+i)),vcvt_f16_f32(vld1q_f32(input+i+4)));vst1q_f16(ff_half+i,h);vst1q_f32(ff_rounded+i,vcvt_f32_f16(vget_low_f16(h)));vst1q_f32(ff_rounded+i+4,vcvt_f32_f16(vget_high_f16(h)));}
    double rounded_end=ff_now();
    double clock_low=0,clock_high=0;int calibrated=ff_gpu&&ff_events&&ff_native_mode;
    if(calibrated)ff_clock_bounds(&clock_low,&clock_high);
    if(ff_cpu){pthread_mutex_lock(&ff_mutex);ff_job_layer=layer;ff_job_rows=rows;ff_pending=1;pthread_cond_signal(&ff_condition);pthread_mutex_unlock(&ff_mutex);}
    cl_event upload=NULL,kernels[2]={NULL,NULL},download=NULL;int kernel_count=rows==1?2:1;
    if(ff_gpu){
        int N=2*ff_gpu;
        ff_check(clEnqueueWriteBuffer(ff_queue,ff_input,CL_FALSE,0,(size_t)rows*1024*2,ff_half,0,NULL,ff_events?&upload:NULL),"input upload");
        if(rows==1){
            ff_memarg(ff_gemv,0,ff_input);ff_memarg(ff_gemv,1,ff_gpu_weights[layer]);ff_memarg(ff_gemv,2,ff_partials);ff_intarg(ff_gemv,3,1024);ff_intarg(ff_gemv,4,N);ff_intarg(ff_gemv,5,8);
            size_t global[]={((N/4+63)/64)*64,8},local[]={64,1};ff_check(clEnqueueNDRangeKernel(ff_queue,ff_gemv,2,NULL,global,local,0,NULL,ff_events?&kernels[0]:NULL),"gemv");
            ff_memarg(ff_reduce,0,ff_partials);ff_memarg(ff_reduce,1,ff_output);ff_intarg(ff_reduce,2,N);ff_intarg(ff_reduce,3,8);
            size_t g=((N/4+63)/64)*64,l=64;ff_check(clEnqueueNDRangeKernel(ff_queue,ff_reduce,1,NULL,&g,&l,0,NULL,ff_events?&kernels[1]:NULL),"reduce");
        }else{
            ff_memarg(ff_gemm,0,ff_input);ff_memarg(ff_gemm,1,ff_gpu_weights[layer]);ff_memarg(ff_gemm,2,ff_output);ff_intarg(ff_gemm,3,rows);ff_intarg(ff_gemm,4,1024);ff_intarg(ff_gemm,5,N);ff_intarg(ff_gemm,6,0);
            size_t global[]={((N/4+63)/64)*64,(rows+7)/8},local[]={64,1};ff_check(clEnqueueNDRangeKernel(ff_queue,ff_gemm,2,NULL,global,local,0,NULL,ff_events?&kernels[0]:NULL),"gemm");
        }
        // The host result remains live and untouched until its event completes.
        ff_check(clEnqueueReadBuffer(ff_queue,ff_output,CL_FALSE,0,(size_t)rows*N*4,ff_gpu_result,0,NULL,&download),"async readback");ff_check(clFlush(ff_queue),"flush");
    }
    double submit_start=0,submit_end=0;int gpu_across=0;
    double npu_start=ff_now();cl_int before=CL_COMPLETE,after=CL_COMPLETE;
    if(ff_gpu)ff_check(clGetEventInfo(download,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(before),&before,NULL),"pre-NPU status");
    if(ff_native_mode){
        Plan *p[3],*next=ff_plans[layer];
        for(int i=0;i<3;i++){p[i]=next;next=next->next;configure_rows(p[i],rows,1);}
        attention_pack(p[0],input,1024,rows);
        for(int i=1;i<3;i++){memcpy(p[i]->mem.input,p[0]->packed_input,(size_t)rows*1024*2);mem_sync(device_fd,p[i]->mem.input_obj,0,(size_t)rows*1024*2,RKNPU_MEM_SYNC_TO_DEVICE);}
        cl_int running_before=CL_COMPLETE,running_after=CL_COMPLETE;
        if(ff_gpu&&ff_events)ff_check(clGetEventInfo(kernels[0],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(running_before),&running_before,NULL),"kernel before submit");
        submit_start=ff_now();
        if(submit_bundle(p[0])<0){perror("Concurrent native gate/up");exit(1);}
        submit_end=ff_now();
        if(ff_gpu&&ff_events)ff_check(clGetEventInfo(kernels[0],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(running_after),&running_after,NULL),"kernel after submit");
        gpu_across=running_before==CL_RUNNING&&running_after==CL_RUNNING;

        for(int i=0;i<3;i++)mem_sync(device_fd,p[i]->mem.output_obj,0,(size_t)rows*p[i]->N*4,RKNPU_MEM_SYNC_FROM_DEVICE);
        ctx->npu_ops++;
    }else run_projection(ctx,9,ff_plans[layer],input,ff_joined,rows,2*ff_npu,NULL,NULL,0,0);
    double npu_end=ff_now();
    if(ff_gpu)ff_check(clGetEventInfo(download,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(after),&after,NULL),"post-NPU status");
    if(ff_cpu){pthread_mutex_lock(&ff_mutex);while(ff_pending)pthread_cond_wait(&ff_condition,&ff_mutex);pthread_mutex_unlock(&ff_mutex);ctx->cpu_ops++;}
    if(ff_gpu){ff_check(clWaitForEvents(1,&download),"readback wait");if(ff_events){
        if(calibrated){
            double low,high;ff_clock_bounds(&low,&high);if(low>clock_low)clock_low=low;if(high<clock_high)clock_high=high;
            if(clock_low<=clock_high){
                ff_profile.valid_clock_maps++;
                cl_ulong start,end;ff_check(clGetEventProfilingInfo(kernels[0],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL),"calibrated start");ff_check(clGetEventProfilingInfo(kernels[0],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL),"calibrated end");
                int contains=start/1e6+clock_high<=submit_start&&end/1e6+clock_low>=submit_end;
                ff_profile.gpu_calibrated_contains_submit+=contains;
                ff_profile.all_three_calibrated_contains_submit+=contains&&ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end;
            }
        }
        ff_profile.gpu_upload_ms+=ff_event_ms(upload);clReleaseEvent(upload);ff_profile.gpu_download_ms+=ff_event_ms(download);for(int i=0;i<kernel_count;i++){ff_profile.gpu_kernel_ms+=ff_event_ms(kernels[i]);clReleaseEvent(kernels[i]);}}clReleaseEvent(download);}
    if(submit_start){
        int cpu_across=ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end;
        ff_profile.gpu_running_across_submit+=gpu_across;
        ff_profile.cpu_spans_submit+=cpu_across;
        ff_profile.all_three_spans_submit+=gpu_across&&cpu_across;
    }
    double joined_end=ff_now();
    if(!ff_native_mode)for(int row=0;row<rows;row++)for(int part=0;part<2;part++){
        float *out=(part?up:gate)+(size_t)row*3072;
        memcpy(out,ff_joined+(size_t)row*2*ff_npu+part*ff_npu,(size_t)ff_npu*4);
        if(ff_gpu)memcpy(out+ff_npu,ff_gpu_result+(size_t)row*2*ff_gpu+part*ff_gpu,(size_t)ff_gpu*4);
        if(ff_cpu)memcpy(out+ff_npu+ff_gpu,ff_cpu_result+(size_t)row*2*ff_cpu+part*ff_cpu,(size_t)ff_cpu*4);
    }
    double end=ff_now();
    ff_profile.calls++;ff_profile.cpu_calls+=!!ff_cpu;ff_profile.logical_cpu_ops+=!!ff_cpu;ff_profile.gpu_calls+=!!ff_gpu;ff_profile.npu_calls++;ff_profile.rows+=rows;
    ff_profile.wall_ms+=end-begin;ff_profile.round_ms+=rounded_end-begin;ff_profile.dispatch_ms+=npu_start-rounded_end;ff_profile.npu_ms+=npu_end-npu_start;ff_profile.join_wait_ms+=joined_end-npu_end;ff_profile.merge_ms+=end-joined_end;
    if(ff_cpu){ff_profile.cpu_ms+=ff_cpu_end-ff_cpu_start;double start=ff_cpu_start>npu_start?ff_cpu_start:npu_start,stop=ff_cpu_end<npu_end?ff_cpu_end:npu_end;if(stop>start)ff_profile.cpu_npu_overlap_ms+=stop-start;}
    if(ff_gpu&&before!=CL_COMPLETE&&after==CL_COMPLETE)ff_profile.gpu_completed_during_npu++;
    return 1;
}
// Read one four-channel group from the native prefix or disjoint host slices.
static inline float32x4_t ff_native_value(Plan **planes,int part,int channel,int row,int rows){
    if(channel<ff_npu){
        int global=part*ff_npu+channel,width=2*ff_npu/3,index=global/width,column=global%width;
        return vld1q_f32((const float*)planes[index]->mem.output+(size_t)(column/4)*rows*4+row*4);
    }
    channel-=ff_npu;
    if(channel<ff_gpu)return vld1q_f32(ff_gpu_result+(size_t)row*2*ff_gpu+part*ff_gpu+channel);
    channel-=ff_gpu;
    return vld1q_f32(ff_cpu_result+(size_t)row*2*ff_cpu+part*ff_cpu+channel);
}
int ffn_prefill(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows){
    unsigned long long previous_npu=ctx->npu_ops,previous_cpu=ctx->cpu_ops;
    int blocks=(rows+63)/64;
    ff_native_mode=1;
    Plan *planes[3],*next=ff_plans[layer];for(int i=0;i<3;i++){planes[i]=next;next=next->next;}
    for(int start=0;start<rows;start+=64){
        int m=rows-start<64?rows-start:64;
        ffn_gate_up(ctx,layer,input+(size_t)start*1024,NULL,NULL,m);
        Plan *down[3];next=down_parts[layer];for(int i=0;i<3;i++){down[i]=next;next=next->next;configure_rows(down[i],m,1);}
        double pack_start=ff_now();
        // Preserve the selected stream's SwiGLU and FP16 down-input layout.
        #pragma omp parallel for collapse(2)
        for(int block=0;block<3;block++)for(int k=0;k<1024;k+=8)for(int row=0;row<m;row++){
            int channel=block*1024+k;
            float32x4_t low=qwen_swiglu4(ff_native_value(planes,0,channel,row,m),ff_native_value(planes,1,channel,row,m));
            float32x4_t high=qwen_swiglu4(ff_native_value(planes,0,channel+4,row,m),ff_native_value(planes,1,channel+4,row,m));
            vst1q_f16(down[block]->packed_input+(size_t)(k/8)*m*8+row*8,vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high)));
        }
        double packed=ff_now();ff_activation_pack_ms+=packed-pack_start;
        submit_down_parts(ctx,down,output,start,m);ff_down_wall_ms+=ff_now()-packed;
    }
    ff_native_mode=0;
    // Context counters count logical layer operations, as in the selected stream.
    // The profile retains every physical CPU/GPU/native gate job separately.
    ctx->npu_ops=previous_npu+2;ctx->cpu_ops=previous_cpu+!!ff_cpu;
    if(ff_cpu)ff_profile.logical_cpu_ops-=blocks-1;
    return 1;
}
void ffn_clear(void){memset(&ff_profile,0,sizeof(ff_profile));ff_activation_pack_ms=ff_down_wall_ms=0;}
void ffn_dump(const char *phase,int run){if(!ff_enabled)return;
    fprintf(stderr,"{\"event\":\"ffn_split_profile\",\"phase\":\"%s\",\"run\":%d,\"warmup\":%s,\"calls\":%llu,\"cpu_calls\":%llu,\"gpu_calls\":%llu,\"npu_calls\":%llu,\"rows\":%llu,\"gpu_completed_during_npu\":%llu,\"wall_ms\":%.9f,\"round_ms\":%.9f,\"dispatch_ms\":%.9f,\"npu_branch_ms\":%.9f,\"cpu_branch_ms\":%.9f,\"cpu_npu_overlap_ms\":%.9f,\"join_wait_ms\":%.9f,\"merge_ms\":%.9f,\"gpu_upload_ms\":%.9f,\"gpu_kernel_ms\":%.9f,\"gpu_download_ms\":%.9f,\"activation_pack_ms\":%.9f,\"down_wall_ms\":%.9f,\"logical_cpu_ops\":%llu,\"gpu_running_across_submit\":%llu,\"cpu_spans_submit\":%llu,\"all_three_spans_submit\":%llu,\"valid_clock_maps\":%llu,\"gpu_calibrated_contains_submit\":%llu,\"all_three_calibrated_contains_submit\":%llu,\"event_profiling\":%s}\n",phase,run,run<=0?"true":"false",(unsigned long long)ff_profile.calls,(unsigned long long)ff_profile.cpu_calls,(unsigned long long)ff_profile.gpu_calls,(unsigned long long)ff_profile.npu_calls,(unsigned long long)ff_profile.rows,(unsigned long long)ff_profile.gpu_completed_during_npu,ff_profile.wall_ms,ff_profile.round_ms,ff_profile.dispatch_ms,ff_profile.npu_ms,ff_profile.cpu_ms,ff_profile.cpu_npu_overlap_ms,ff_profile.join_wait_ms,ff_profile.merge_ms,ff_profile.gpu_upload_ms,ff_profile.gpu_kernel_ms,ff_profile.gpu_download_ms,ff_activation_pack_ms,ff_down_wall_ms,(unsigned long long)ff_profile.logical_cpu_ops,(unsigned long long)ff_profile.gpu_running_across_submit,(unsigned long long)ff_profile.cpu_spans_submit,(unsigned long long)ff_profile.all_three_spans_submit,(unsigned long long)ff_profile.valid_clock_maps,(unsigned long long)ff_profile.gpu_calibrated_contains_submit,(unsigned long long)ff_profile.all_three_calibrated_contains_submit,ff_events?"true":"false");
}
void ffn_close(void){if(!ff_enabled)return;
    if(ff_cpu){pthread_mutex_lock(&ff_mutex);ff_stop=1;pthread_cond_signal(&ff_condition);pthread_mutex_unlock(&ff_mutex);pthread_join(ff_thread,NULL);for(int i=0;i<ff_layers;i++)free(ff_cpu_weights[i]);free(ff_cpu_weights);dlclose(ff_blas);}
    if(ff_gpu){ff_check(clFinish(ff_queue),"close finish");for(int i=0;i<ff_layers;i++)clReleaseMemObject(ff_gpu_weights[i]);free(ff_gpu_weights);clReleaseMemObject(ff_input);clReleaseMemObject(ff_output);clReleaseMemObject(ff_partials);clReleaseKernel(ff_gemv);clReleaseKernel(ff_reduce);clReleaseKernel(ff_gemm);clReleaseProgram(ff_program);clReleaseCommandQueue(ff_queue);clReleaseContext(ff_context);}
    for(int i=0;i<ff_layers;i++){Plan *p=ff_plans[i];while(p){Plan *next=p->next;release_memhandles(device_fd,&p->mem);free(p->packed_input);free(p->native_output);free(p);p=next;}}
    free(ff_plans);free(ff_joined);free(ff_gate);free(ff_up);free(ff_half);free(ff_rounded);free(ff_cpu_result);free(ff_gpu_result);ff_enabled=0;
}
