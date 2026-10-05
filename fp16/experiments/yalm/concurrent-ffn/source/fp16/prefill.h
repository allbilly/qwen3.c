// Layer-by-layer prompt processing with persistent CPU scratch and NPU batches.
#include <arm_neon.h>
#include "vector_math.h"
#include "gpu_attention.h"
#include "../../router.h"
typedef struct {
    float *x, *xb, *q, *k, *v, *hb, *hb2, *cosine, *sine, *joined;
    float *key_heads, *value_heads;
    int heads_ready;
    int profiling;
    double attention_ms;
} Prefill;

static double prefill_clock(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}

static void prefill_profile_dump(Prefill *b,const char *phase) {
    if(b->profiling)fprintf(stderr,"{\"attention_phase\":\"%s\",\"attention_ms\":%.3f}\n",phase,b->attention_ms);
    b->attention_ms=0;
}

static void prepare_prefill(Prefill *b, const Config *p) {
    b->profiling=getenv("NPU_PROFILE")!=NULL;b->attention_ms=0;
    b->heads_ready=0;
    size_t rows=p->seq_len;
    b->x=malloc(rows*p->dim*4);
    b->xb=malloc(rows*p->n_heads*p->head_dim*4);
    b->q=malloc(rows*p->n_heads*p->head_dim*4);
    b->k=malloc(rows*p->n_kv_heads*p->head_dim*4);
    b->v=malloc(rows*p->n_kv_heads*p->head_dim*4);
    b->hb=malloc(rows*p->hidden_dim*4);
    b->hb2=malloc(rows*p->hidden_dim*4);
    b->cosine=malloc(rows*p->head_dim/2*4);
    b->sine=malloc(rows*p->head_dim/2*4);
    b->joined=malloc(rows*2*p->hidden_dim*4);
    b->key_heads=malloc((size_t)p->n_layers*p->n_kv_heads*rows*p->head_dim*4);
    b->value_heads=malloc((size_t)p->n_layers*p->n_kv_heads*rows*p->head_dim*4);
    if(!b->x||!b->xb||!b->q||!b->k||!b->v||!b->hb||!b->hb2||!b->cosine||!b->sine||!b->joined)exit(1);
    if(!b->key_heads||!b->value_heads)exit(1);
    for(int pos=0;pos<p->seq_len;pos++)for(int j=0;j<p->head_dim/2;j++) {
        float frequency=powf(1e6f,-(float)j/(p->head_dim/2));
        b->cosine[pos*(p->head_dim/2)+j]=cosf(pos*frequency);
        b->sine[pos*(p->head_dim/2)+j]=sinf(pos*frequency);
    }
}

static void free_prefill(Prefill *b) {
    free(b->x);free(b->xb);free(b->q);free(b->k);free(b->v);
    free(b->hb);free(b->hb2);free(b->cosine);free(b->sine);
    free(b->joined);
    free(b->key_heads);free(b->value_heads);
}

static void normalize_rotate(float *vector, float *weight, const Prefill *b,
                              int pos, int head_dim) {
    rmsnorm(vector,vector,weight,head_dim);
    for(int j=0;j<head_dim/2;j++) {
        float c=b->cosine[pos*(head_dim/2)+j],s=b->sine[pos*(head_dim/2)+j];
        float x=vector[j],y=vector[j+head_dim/2];
        vector[j]=x*c-y*s;vector[j+head_dim/2]=x*s+y*c;
    }
}

// Keep each output accumulator in a register through the token loop.
// Per-channel accumulation visits the same tokens in the same order.
static void weighted_values(float *output,const float *weights,const float *values,int length,int stride) {
    for(int feature=0;feature<128;feature+=32) {
        float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
        for(int time=0;time<length;time++) {
            const float *v=values+(size_t)time*stride+feature;
            float32x4_t weight=vdupq_n_f32(weights[time]);
            a0=vfmaq_f32(a0,vld1q_f32(v),weight);a1=vfmaq_f32(a1,vld1q_f32(v+4),weight);
            a2=vfmaq_f32(a2,vld1q_f32(v+8),weight);a3=vfmaq_f32(a3,vld1q_f32(v+12),weight);
            a4=vfmaq_f32(a4,vld1q_f32(v+16),weight);a5=vfmaq_f32(a5,vld1q_f32(v+20),weight);
            a6=vfmaq_f32(a6,vld1q_f32(v+24),weight);a7=vfmaq_f32(a7,vld1q_f32(v+28),weight);
        }
        vst1q_f32(output+feature,a0);vst1q_f32(output+feature+4,a1);
        vst1q_f32(output+feature+8,a2);vst1q_f32(output+feature+12,a3);
        vst1q_f32(output+feature+16,a4);vst1q_f32(output+feature+20,a5);
        vst1q_f32(output+feature+24,a6);vst1q_f32(output+feature+28,a7);
    }
}

static float *forward_batch_atpos(Transformer *t, Prefill *b, const int *ids, int count, int position) {
    route_phase(position);
    Config *p=&t->config; TransformerWeights *w=&t->weights; RunState *s=&t->state;
    int dim=p->dim,hidden=p->hidden_dim,hd=p->head_dim;
    int qdim=p->n_heads*hd,kdim=p->n_kv_heads*hd,share=p->n_heads/p->n_kv_heads;
    if(position==0)b->heads_ready=0;
    else if(!b->heads_ready && !gpu_decode_enabled()) {
        // Build the CPU decode layout on the first decode call. This work is
        // inside that call and its measured decode span, never an untimed gap.
        #pragma omp parallel for collapse(2)
        for(int layer=0;layer<p->n_layers;layer++)for(int head=0;head<p->n_kv_heads;head++) {
            size_t offset=(size_t)layer*p->seq_len*kdim;
            for(int row=0;row<position;row++) {
                size_t src=offset+(size_t)row*kdim+head*hd;
                size_t dst=offset+(size_t)head*p->seq_len*hd+(size_t)row*hd;
                memcpy(b->key_heads+dst,s->key_cache+src,hd*4);
                memcpy(b->value_heads+dst,s->value_cache+src,hd*4);
            }
        }
        b->heads_ready=1;
    }
    for(int row=0;row<count;row++)memcpy(b->x+(size_t)row*dim,w->token_embedding_table+(size_t)ids[row]*dim,dim*4);
    for(int layer=0;layer<p->n_layers;layer++) {
        size_t offset=(size_t)layer*p->seq_len*kdim;
        #pragma omp parallel for if(count>1)
        for(int row=0;row<count;row++)rmsnorm(b->xb+(size_t)row*dim,b->x+(size_t)row*dim,w->rms_att_weight+layer*dim,dim);
        if(npu_uses_fused()) {
            npu_matmul_fused_to(&g_npu,0,layer,b->xb,b->q,b->k,b->v,count);
        } else {
            npu_matmul_batch(&g_npu,NPU_MATMUL_WQ,layer,b->xb,b->q,count);
            npu_matmul_batch(&g_npu,NPU_MATMUL_WK,layer,b->xb,b->k,count);
            npu_matmul_batch(&g_npu,NPU_MATMUL_WV,layer,b->xb,b->v,count);
        }
        #pragma omp parallel for collapse(2) if(count>1)
        for(int row=0;row<count;row++)for(int head=0;head<p->n_heads;head++)
            normalize_rotate(b->q+(size_t)row*qdim+head*hd,w->q_norm_weights+layer*hd,b,position+row,hd);
        #pragma omp parallel for collapse(2) if(count>1)
        for(int row=0;row<count;row++)for(int head=0;head<p->n_kv_heads;head++)
            normalize_rotate(b->k+(size_t)row*kdim+head*hd,w->k_norm_weights+layer*hd,b,position+row,hd);
        // Keep each KV head contiguous through time for CPU decode attention.
        // Cache values remain FP32; only their address layout changes.
        if(!b->heads_ready) {
            memcpy(s->key_cache+offset+(size_t)position*kdim,b->k,(size_t)count*kdim*4);
            memcpy(s->value_cache+offset+(size_t)position*kdim,b->v,(size_t)count*kdim*4);
        } else {
        #pragma omp parallel for if(count>1)
        for(int head=0;head<p->n_kv_heads;head++)for(int row=0;row<count;row++) {
            size_t at=offset+(size_t)head*p->seq_len*hd+(size_t)(position+row)*hd;
            memcpy(b->key_heads+at,b->k+(size_t)row*kdim+head*hd,hd*4);
            memcpy(b->value_heads+at,b->v+(size_t)row*kdim+head*hd,hd*4);
        }
        }
        double attention_start=b->profiling?prefill_clock():0;
        if(position==0 && gpu_decode_enabled() && !(count>1 && gpu_prefill_enabled()))
            gpu_cache_upload(layer,b->k,b->v,count);
        if((position==0 && count>1 && gpu_prefill_enabled()) || (position>0 && gpu_decode_enabled())) {
            gpu_attention(layer,b->q,b->k,b->v,b->xb,count,position);
        } else if(npu_attention_enabled() && position==0 && count>1) {
            npu_attention_prefill(&g_npu,b->q,b->k,b->v,b->xb,count);
        } else {
        #pragma omp parallel for collapse(2)
        for(int row=0;row<count;row++)for(int head=0;head<p->n_heads;head++) {
            float scores[512];
            float *q=b->q+(size_t)row*qdim+head*hd;
            int cache_stride=b->heads_ready?hd:kdim;
            const float *key=b->heads_ready?b->key_heads+offset+(size_t)(head/share)*p->seq_len*hd:
                s->key_cache+offset+(head/share)*hd;
            const float *value=b->heads_ready?b->value_heads+offset+(size_t)(head/share)*p->seq_len*hd:
                s->value_cache+offset+(head/share)*hd;
            for(int time=0;time<=position+row;time++) {
                const float *k=key+(size_t)time*cache_stride;
                float score=0;for(int i=0;i<hd;i++)score+=q[i]*k[i];
                scores[time]=score/sqrtf(hd);
            }
            softmax(scores,position+row+1);
            float *out=b->xb+(size_t)row*qdim+head*hd;
            weighted_values(out,scores,value,position+row+1,cache_stride);
        }
        }
        if(b->profiling)b->attention_ms+=prefill_clock()-attention_start;
        npu_matmul_batch(&g_npu,NPU_MATMUL_WO,layer,b->xb,b->xb,count);
        #pragma omp parallel for if(count>1)
        for(int row=0;row<count;row++) {
            float *x=b->x+(size_t)row*dim,*xb=b->xb+(size_t)row*dim;
            for(int i=0;i<dim;i++)x[i]+=xb[i];
            rmsnorm(xb,x,w->rms_ffn_weight+layer*dim,dim);
        }
        if(npu_stream_ffn_enabled() && position==0 && count>1) {
            npu_stream_ffn_prefill(&g_npu,layer,b->xb,b->xb,count);
        } else {
        if(npu_uses_fused()) {
            npu_matmul_fused_to(&g_npu,1,layer,b->xb,b->hb,b->hb2,NULL,count);
        } else {
            npu_matmul_batch(&g_npu,NPU_MATMUL_W1,layer,b->xb,b->hb,count);
            npu_matmul_batch(&g_npu,NPU_MATMUL_W3,layer,b->xb,b->hb2,count);
        }
        #pragma omp parallel for if(count>1)
        for(int i=0;i<count*hidden;i+=4)
            vst1q_f32(b->hb+i,qwen_swiglu4(vld1q_f32(b->hb+i),vld1q_f32(b->hb2+i)));
        npu_matmul_batch(&g_npu,NPU_MATMUL_W2,layer,b->hb,b->xb,count);
        }
        #pragma omp parallel for if(count>1)
        for(int i=0;i<count*dim;i++)b->x[i]+=b->xb[i];
    }
    rmsnorm(s->x,b->x+(size_t)(count-1)*dim,w->rms_final_weight,dim);
    npu_skip_classifier(0);
    npu_matmul_run(&g_npu,NPU_MATMUL_WCLS,0,s->x,s->logits);
    return s->logits;
}

static float *forward_prefill(Transformer *t, Prefill *b, const int *ids, int count) {
    return forward_batch_atpos(t,b,ids,count,0);
}
