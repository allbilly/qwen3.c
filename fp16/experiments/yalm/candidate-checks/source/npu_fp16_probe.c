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
static unsigned half_bits(float value){__fp16 half=(__fp16)value;uint16_t bits;memcpy(&bits,&half,2);return bits;}
int main(int argc,char **argv){
 if(argc!=2)return 2;
 setenv("DENSE_PREFILL","npu",1);setenv("DENSE_DECODE","npu",1);unsetenv("GPU_ATTENTION");unsetenv("CPU_CLASSIFIER");
 setenv("NPU_CORES","3",1);setenv("NPU_DOMAIN_ID","1",1);setenv("NPU_FUSED","1",1);setenv("NPU_ATTENTION","1",1);
 setenv("NPU_SPLIT_DOWN","1",1);setenv("NPU_STREAM_FFN","1",1);setenv("NPU_CLS_TILE","8192",1);
 Transformer t;const __fp16 **matrices;build_fp16(&t,argv[1],&matrices);
 if(!npu_init_fp16(&g_npu,&t.config,matrices))return 1;free(matrices);
 float *q=calloc(2*2048,4),*k=calloc(2*1024,4),*v=calloc(2*1024,4),*out=calloc(2*2048,4),*ref=calloc(2*2048,4);
 if(!q||!k||!v||!out||!ref)return 1;
 unsigned long fpcr;asm volatile("mrs %0, fpcr":"=r"(fpcr));
 printf("{\"event\":\"fpcr\",\"value\":%lu}\n",fpcr);
 v[0]=ldexpf(1,-24);v[1]=ldexpf(1,-17);v[2]=ldexpf(1,-14)-ldexpf(1,-24);v[3]=ldexpf(1,-14);
 npu_attention_prefill(&g_npu,q,k,v,out,1);reference(q,k,v,ref,1,0,1);
 for(int i=0;i<4;i++)printf("{\"event\":\"fp16_probe\",\"scene\":\"value_weight\",\"feature\":%d,\"source_half_bits\":%u,\"ieee_reference\":%.12g,\"actual\":%.12g}\n",i,half_bits(v[i]),ref[i],out[i]);
 const char *names[]={"score_weight","score_input","probability_input"};
 for(int scene=0;scene<3;scene++){
  memset(q,0,2*2048*4);memset(k,0,2*1024*4);memset(v,0,2*1024*4);
  if(scene==0){q[2048]=65504;k[0]=ldexpf(1,-17);v[0]=1;}
  if(scene==1){q[2048]=ldexpf(1,-17);k[0]=65504;v[0]=1;}
  if(scene==2){q[2048]=65504;k[0]=ldexpf(1,-9);v[1024]=1;}
  npu_attention_prefill(&g_npu,q,k,v,out,2);reference(q,k,v,ref,2,0,1);
  float operands[]={scene==2?ref[2048]:q[2048],k[0],v[0],v[1024]};uint16_t vector_bits[4];
  vst1_u16(vector_bits,vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(operands))));
  printf("{\"event\":\"fp16_probe\",\"scene\":\"%s\",\"q_half_bits\":%u,\"k_half_bits\":%u,\"first_operand_vector_half_bits\":%u,\"ieee_reference\":%.12g,\"actual\":%.12g}\n",names[scene],half_bits(q[2048]),half_bits(k[0]),vector_bits[0],ref[2048],out[2048]);
 }
 npu_matmul_shutdown(&g_npu);free_transformer(&t);free(q);free(k);free(v);free(out);free(ref);return 0;
}
