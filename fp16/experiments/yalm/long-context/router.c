#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <arm_neon.h>
#include "source/fp16/fp16_backend.h"
#include "linear_gpu.h"
#include "router.h"
#include "concurrent_ffn.h"
enum {CPU,GPU,NPU};
static int pre_device,decode_device,current_device,has_npu,has_gpu,has_cpu,skip_head,cpu_head;
static int layers,capacity;static const __fp16 **weights;
static float *rounded,*weight_tile,*result_tile;static void *blas;
typedef void (*Sgemm)(int,int,int,int64_t,int64_t,int64_t,float,const float *,int64_t,const float *,int64_t,float,float *,int64_t);
static Sgemm sgemm;
typedef struct {uint64_t calls,bytes;double wall_ms,pack_ms,compute_ms;} CpuProfile;
static CpuProfile profile;
static double cpu_attention_ms;static uint64_t cpu_attention_calls;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static const char *names[]={"cpu","gpu","npu"};
int hw_init_fp16(NpuMatmulContext *,const Config *,const __fp16 **);
int hw_matmul_batch(NpuMatmulContext *,NpuMatmulKind,int,const float *,float *,int);
int hw_matmul_fused(NpuMatmulContext *,int,int,const float *,float *,int);
int hw_matmul_fused_to(NpuMatmulContext *,int,int,const float *,float *,float *,float *,int);
int hw_stream_ffn_prefill(NpuMatmulContext *,int,const float *,float *,int);
int hw_attention_prefill(NpuMatmulContext *,const float *,const float *,const float *,float *,int);
void hw_shutdown(NpuMatmulContext *);
void hw_skip_classifier(int);
int hw_core_count(void);int hw_uses_fused(void);int hw_stream_ffn_enabled(void);
static int parse(const char *name){const char *s=getenv(name);if(!s)s="npu";for(int i=0;i<3;i++)if(!strcmp(s,names[i]))return i;fprintf(stderr,"Invalid %s=%s\n",name,s);exit(2);}
void route_phase(int position){current_device=position?decode_device:pre_device;ffn_phase(position);}
int npu_core_count(void){return has_npu?hw_core_count():0;}
int npu_uses_fused(void){return current_device==NPU&&hw_uses_fused();}
int npu_stream_ffn_enabled(void){return current_device==NPU&&hw_stream_ffn_enabled();}
int npu_attention_enabled(void){return 1;}
void npu_skip_classifier(int skip){skip_head=skip;if(has_npu)hw_skip_classifier(skip);}
int npu_init_fp16(NpuMatmulContext *ctx,const Config *c,const __fp16 **matrices){
    pre_device=parse("DENSE_PREFILL");decode_device=parse("DENSE_DECODE");current_device=pre_device;
    cpu_head=getenv("CPU_CLASSIFIER")!=NULL;
    has_npu=pre_device==NPU||decode_device==NPU;has_gpu=pre_device==GPU||decode_device==GPU;has_cpu=pre_device==CPU||decode_device==CPU||cpu_head;
    capacity=c->seq_len;layers=c->n_layers;weights=malloc((7*layers+1)*sizeof(*weights));if(!weights)return 0;memcpy(weights,matrices,(7*layers+1)*sizeof(*weights));
    if(has_npu){if(!hw_init_fp16(ctx,c,matrices))return 0;}
    else{memset(ctx,0,sizeof(*ctx));ctx->dim=c->dim;ctx->hidden_dim=c->hidden_dim;ctx->n_layers=c->n_layers;ctx->all_heads_dim=c->n_heads*c->head_dim;ctx->kv_dim=c->n_kv_heads*c->head_dim;ctx->vocab_size=c->vocab_size;ctx->enabled=1;}
    if(!ffn_init(ctx,c,matrices))return 0;
    if(has_gpu&&!linear_gpu_init(c,matrices))return 0;
    if(has_cpu){
        const char *library=getenv("YALM_BLAS");if(!library)library="/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs/libopenblas64_p-r0-17488984.3.23.dev.so";
        blas=dlopen(library,RTLD_NOW|RTLD_LOCAL);if(!blas){fprintf(stderr,"%s\n",dlerror());return 0;}
        sgemm=(Sgemm)dlsym(blas,"cblas_sgemm64_");void (*threads)(int)=dlsym(blas,"openblas_set_num_threads64_");char *(*config)(void)=dlsym(blas,"openblas_get_config64_");
        if(!sgemm||!threads||!config)return 0;threads(4);
        rounded=malloc((size_t)capacity*3072*4);weight_tile=malloc((size_t)256*3072*4);result_tile=malloc((size_t)capacity*256*4);
        if(!rounded||!weight_tile||!result_tile)return 0;
        fprintf(stderr,"{\"event\":\"linear_cpu_init\",\"storage\":\"FP16\",\"accumulation\":\"FP32\",\"blas\":\"%s\"}\n",config());
    }
    fprintf(stderr,"{\"event\":\"routing\",\"dense_prefill\":\"%s\",\"dense_decode\":\"%s\",\"cpu_classifier\":%s,\"npu_initialized\":%s,\"gpu_projections_initialized\":%s,\"host_non_linear_ops\":\"CPU\"}\n",names[pre_device],names[decode_device],cpu_head?"true":"false",has_npu?"true":"false",has_gpu?"true":"false");
    return 1;
}
static void cpu_projection(const __fp16 *w,const float *input,float *output,int rows,int K,int N,int split){
    double begin=now();
    for(int i=0;i<rows*K;i+=8){float16x8_t h=vcombine_f16(vcvt_f16_f32(vld1q_f32(input+i)),vcvt_f16_f32(vld1q_f32(input+i+4)));vst1q_f32(rounded+i,vcvt_f32_f16(vget_low_f16(h)));vst1q_f32(rounded+i+4,vcvt_f32_f16(vget_high_f16(h)));}
    profile.pack_ms+=now()-begin;double compute=now();
    if(rows==1){
        #pragma omp parallel for
        for(int n=0;n<N;n++){
            float32x4_t lo=vdupq_n_f32(0),hi=lo;
            for(int k=0;k<K;k+=8){float16x8_t h=vld1q_f16(w+(size_t)n*K+k);lo=vfmaq_f32(lo,vcvt_f32_f16(vget_low_f16(h)),vld1q_f32(rounded+k));hi=vfmaq_f32(hi,vcvt_f32_f16(vget_high_f16(h)),vld1q_f32(rounded+k+4));}
            output[n]=vaddvq_f32(vaddq_f32(lo,hi));
        }
    }else for(int n0=0;n0<N;n0+=256){
        int width=N-n0<256?N-n0:256,chunks=split?3:1,block=K/chunks;
        for(int c=0;c<chunks;c++){
            #pragma omp parallel for
            for(int n=0;n<width;n++)for(int k=0;k<block;k+=8){float16x8_t h=vld1q_f16(w+(size_t)(n+n0)*K+c*block+k);vst1q_f32(weight_tile+(size_t)n*block+k,vcvt_f32_f16(vget_low_f16(h)));vst1q_f32(weight_tile+(size_t)n*block+k+4,vcvt_f32_f16(vget_high_f16(h)));}
            sgemm(101,111,112,rows,width,block,1,rounded+c*block,K,weight_tile,block,0,result_tile,width);
            for(int r=0;r<rows;r++)for(int n=0;n<width;n++){size_t index=(size_t)r*N+n0+n;if(c==0)output[index]=result_tile[(size_t)r*width+n];else output[index]+=result_tile[(size_t)r*width+n];}
        }
    }
    profile.compute_ms+=now()-compute;profile.wall_ms+=now()-begin;profile.calls++;profile.bytes+=(size_t)K*N*2;
}
int npu_matmul_batch(NpuMatmulContext *ctx,NpuMatmulKind kind,int layer,const float *input,float *output,int rows){
    if(kind==NPU_MATMUL_WCLS&&skip_head)return 1;
    int device=cpu_head&&kind==NPU_MATMUL_WCLS?CPU:current_device;
    if(device==NPU)return hw_matmul_batch(ctx,kind,layer,input,output,rows);
    int ks[]={1024,1024,1024,2048,1024,3072,1024,1024},ns[]={2048,1024,1024,1024,3072,1024,3072,ctx->vocab_size};
    int index=kind==7?7*layers:kind*layers+layer,K=ks[kind],N=ns[kind],split=kind==5&&rows>1;
    if(device==GPU)linear_gpu_run(index,input,output,rows,K,N,split);
    else{cpu_projection(weights[index],input,output,rows,K,N,split);ctx->cpu_ops++;}
    return 1;
}
int npu_matmul_run(NpuMatmulContext *ctx,NpuMatmulKind kind,int layer,const float *input,float *output){return npu_matmul_batch(ctx,kind,layer,input,output,1);}
int npu_matmul_fused(NpuMatmulContext *ctx,int combo,int layer,const float *input,float *output,int rows){return hw_matmul_fused(ctx,combo,layer,input,output,rows);}
int npu_matmul_fused_to(NpuMatmulContext *ctx,int combo,int layer,const float *input,float *a,float *b,float *c,int rows){if(combo==1&&ffn_active())return ffn_gate_up(ctx,layer,input,a,b,rows);return hw_matmul_fused_to(ctx,combo,layer,input,a,b,c,rows);}
int npu_stream_ffn_prefill(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows){if(ffn_active())return ffn_prefill(ctx,layer,input,output,rows);return hw_stream_ffn_prefill(ctx,layer,input,output,rows);}
int npu_attention_prefill(NpuMatmulContext *ctx,const float *q,const float *k,const float *v,float *out,int rows){
    if(current_device==NPU)return hw_attention_prefill(ctx,q,k,v,out,rows);
    double begin=now();
    // Match prompt operand precision; decode remains the original FP32 CPU path.
    #pragma omp parallel for collapse(2)
    for(int row=0;row<rows;row++)for(int head=0;head<16;head++){
        float prob[capacity];__fp16 hq[128];for(int f=0;f<128;f++)hq[f]=(__fp16)q[(size_t)row*2048+head*128+f];
        for(int t=0;t<=row;t++){
            float32x4_t sum=vdupq_n_f32(0);
            for(int f=0;f<128;f+=4){float32x4_t kv=vcvt_f32_f16(vcvt_f16_f32(vld1q_f32(k+(size_t)t*1024+(head/2)*128+f)));sum=vfmaq_f32(sum,vcvt_f32_f16(vld1_f16(hq+f)),kv);}
            prob[t]=vaddvq_f32(sum)/sqrtf(128);
        }
        extern void softmax(float *,int);softmax(prob,row+1);
        for(int f=0;f<128;f+=4){float32x4_t sum=vdupq_n_f32(0);for(int t=0;t<=row;t++)sum=vfmaq_f32(sum,vdupq_n_f32((float)(__fp16)prob[t]),vcvt_f32_f16(vcvt_f16_f32(vld1q_f32(v+(size_t)t*1024+(head/2)*128+f))));vst1q_f32(out+(size_t)row*2048+head*128+f,sum);}
    }
    cpu_attention_ms+=now()-begin;cpu_attention_calls++;return 1;
}
void route_clear(void){memset(&profile,0,sizeof(profile));cpu_attention_ms=0;cpu_attention_calls=0;linear_gpu_clear();ffn_clear();}
void route_dump(const char *phase,int run){
    fprintf(stderr,"{\"event\":\"linear_cpu_profile\",\"phase\":\"%s\",\"run\":%d,\"warmup\":%s,\"calls\":%llu,\"weight_bytes\":%llu,\"wall_ms\":%.9f,\"pack_ms\":%.9f,\"compute_ms\":%.9f,\"prefill_attention_calls\":%llu,\"prefill_attention_ms\":%.9f}\n",phase,run,run<=0?"true":"false",(unsigned long long)profile.calls,(unsigned long long)profile.bytes,profile.wall_ms,profile.pack_ms,profile.compute_ms,(unsigned long long)cpu_attention_calls,cpu_attention_ms);
    linear_gpu_dump(phase,run);ffn_dump(phase,run);
}
void npu_matmul_shutdown(NpuMatmulContext *ctx){ffn_close();if(has_npu)hw_shutdown(ctx);linear_gpu_close();free(weights);free(rounded);free(weight_tile);free(result_tile);if(blas)dlclose(blas);}
