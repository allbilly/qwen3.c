// Check causal prefill and FP32 decode on deterministic data, including GQA.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "gpu_attention.h"
static void reference(const float *q,const float *k,const float *v,float *out,int rows,int position,int half_operands) {
    for(int row=0;row<rows;row++)for(int head=0;head<16;head++) {
        float scores[512],prob[512];float maximum=-INFINITY;
        int length=position+row+1;
        for(int t=0;t<length;t++) {
            double sum=0;
            for(int f=0;f<128;f++) {
                float a=q[row*2048+head*128+f],b=k[t*1024+(head/2)*128+f];
                if(half_operands){a=(__fp16)a;b=(__fp16)b;}
                sum+=(double)a*b;
            }
            scores[t]=(float)sum/sqrtf(128);if(scores[t]>maximum)maximum=scores[t];
        }
        float total=0;for(int t=0;t<length;t++){prob[t]=expf(scores[t]-maximum);total+=prob[t];}
        for(int t=0;t<length;t++){prob[t]/=total;if(half_operands)prob[t]=(__fp16)prob[t];}
        for(int f=0;f<128;f++) {
            double sum=0;
            for(int t=0;t<length;t++){float value=v[t*1024+(head/2)*128+f];if(half_operands)value=(__fp16)value;sum+=(double)prob[t]*value;}
            out[row*2048+head*128+f]=(float)sum;
        }
    }
}

#include "source/fp16/model_helpers.h"
#ifndef NPU_CHECK_ATTN_ROWS
#define NPU_CHECK_ATTN_ROWS 128
#endif
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    setenv("DENSE_PREFILL","npu",1);setenv("DENSE_DECODE","npu",1);
    unsetenv("GPU_ATTENTION");unsetenv("CPU_CLASSIFIER");
    setenv("NPU_CORES","3",1);setenv("NPU_DOMAIN_ID","1",1);
    setenv("NPU_FUSED","1",1);setenv("NPU_ATTENTION","1",1);
    setenv("NPU_SPLIT_DOWN","1",1);setenv("NPU_STREAM_FFN","1",1);
    setenv("NPU_CLS_TILE","8192",1);
    Transformer t;const __fp16 **matrices;
    build_fp16(&t,argv[1],&matrices);
    if(!npu_init_fp16(&g_npu,&t.config,matrices)){free(matrices);free_transformer(&t);return 1;}
    free(matrices);
    float *q=malloc(512*2048*4),*k=malloc(512*1024*4),*v=malloc(512*1024*4);
    float *out=malloc(512*2048*4),*ref=malloc(512*2048*4);int status=0;
    if(!q||!k||!v||!out||!ref){status=1;goto cleanup;}
    for(int i=0;i<512*2048;i++)q[i]=((i*13%127)-63)/97.0f;
    for(int i=0;i<512*1024;i++){k[i]=((i*17%131)-65)/101.0f;v[i]=((i*19%137)-68)/103.0f;}
    int lengths[]={1,24,73,128,256};
    for(int index=0;index<5;index++) {
        int rows=lengths[index];npu_matmul_reset_stats(&g_npu);route_phase(0);
        npu_attention_prefill(&g_npu,q,k,v,out,rows);
        reference(q,k,v,ref,rows,0,1);
        double error=0,energy=0,maximum=0;
        for(int i=0;i<rows*2048;i++) {
            uint32_t bits;memcpy(&bits,out+i,4);
            if((bits&0x7f800000u)==0x7f800000u){status=2;goto cleanup;}
            double delta=out[i]-ref[i];error+=delta*delta;energy+=(double)ref[i]*ref[i];
            if(fabs(delta)>maximum)maximum=fabs(delta);
        }
        double relative=sqrt(error/energy);
        int expected=12*((rows+NPU_CHECK_ATTN_ROWS-1)/NPU_CHECK_ATTN_ROWS);
        int passed=relative<1e-4&&g_npu.npu_ops==(unsigned long long)expected;
        printf("{\"event\":\"attention_check\",\"phase\":\"prefill\",\"context\":%d,\"checked\":%d,\"relative_rmse\":%.12g,\"max_absolute_error\":%.12g,\"limit\":0.0001,\"tile_rows\":%d,\"npu_ops\":%llu,\"expected_npu_ops\":%d,\"passed\":%s}\n",rows,rows*2048,relative,maximum,NPU_CHECK_ATTN_ROWS,g_npu.npu_ops,expected,passed?"true":"false");
        fflush(stdout);if(!passed){status=3;goto cleanup;}
    }
cleanup:
    npu_matmul_shutdown(&g_npu);free_transformer(&t);
    free(q);free(k);free(v);free(out);free(ref);return status;
}
