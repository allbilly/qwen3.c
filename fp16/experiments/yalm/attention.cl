#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Shared FP16 prefill operands, FP32 arithmetic, then FP16 probabilities as
// in the original NPU path. Decode preserves the original FP32 Q/K/V cache.
__kernel void pack_kv(__global const float *k, __global const float *v,
                      __global half *kt, __global half *vh, int rows,
                      __global const float *q,__global half *qh) {
    int i=get_global_id(0);if(i>=rows*2048)return;
    qh[i]=convert_half_rte(q[i]);
    if(i<rows*1024) {
        int time=i/1024, head=(i%1024)/128, feature=i%128;
        kt[(head*128+feature)*rows+time]=convert_half_rte(k[i]);
        vh[i]=convert_half_rte(v[i]);
    }
}
__kernel void prefill_scores(__global const half *q,__global const half *kt,
                             __global float *scores,int rows) {
    int i=get_global_id(0);if(i>=16*rows*rows)return;
    int time=i%rows,row=(i/rows)%rows,head=i/(rows*rows);
    if(time>row){scores[i]=0;return;}
    float4 acc=(float4)(0);
    for(int f=0;f<128;f+=4) {
        float4 query=vload_half4(0,q+row*2048+head*128+f);
        int base=((head/2)*128+f)*rows+time;
        float4 key=(float4)((float)kt[base],(float)kt[base+rows],(float)kt[base+2*rows],(float)kt[base+3*rows]);
        acc=fma(query,key,acc);
    }
    scores[i]=(acc.s0+acc.s1+acc.s2+acc.s3)*0.08838834764831845f;
}
__kernel void prefill_softmax(__global float *scores,__global half *prob,int rows) {
    int i=get_global_id(0);if(i>=16*rows)return;
    int length=i%rows+1,base=i*rows;
    // Match qwen3.c's zero initial maximum and materialized FP32 exponentials.
    float maximum=0;
    for(int t=0;t<length;t++)maximum=fmax(maximum,scores[base+t]);
    float total=0;for(int t=0;t<length;t++){float value=exp(scores[base+t]-maximum);scores[base+t]=value;total+=value;}
    for(int t=0;t<length;t++)prob[base+t]=convert_half_rte(scores[base+t]/total);
}
__kernel void prefill_values(__global const half *prob,__global const half *v,
                             __global float *out,int rows) {
    int i=get_global_id(0);if(i>=rows*512)return;
    int row=i/512,head=(i%512)/32,feature=(i%32)*4;
    float4 acc=(float4)(0);
    for(int t=0;t<=row;t++)acc=fma((float4)((float)prob[(head*rows+row)*rows+t]),vload_half4(0,v+t*1024+(head/2)*128+feature),acc);
    vstore4(acc,0,out+row*2048+head*128+feature);
}
__kernel void decode_scores(__global const float *q,__global const float *k,
                            __global float *scores,int length) {
    int i=get_global_id(0);if(i>=16*length)return;
    int time=i%length,head=i/length;
    float4 acc=(float4)(0);
    for(int f=0;f<128;f+=4)acc=fma(vload4(0,q+head*128+f),vload4(0,k+time*1024+(head/2)*128+f),acc);
    scores[i]=(acc.s0+acc.s1+acc.s2+acc.s3)*0.08838834764831845f;
}
__kernel void decode_softmax(__global float *scores,__global float *prob,int length) {
    int head=get_global_id(0);if(head>=16)return;
    int base=head*length;float maximum=0;
    for(int t=0;t<length;t++)maximum=fmax(maximum,scores[base+t]);
    float total=0;for(int t=0;t<length;t++){float value=exp(scores[base+t]-maximum);scores[base+t]=value;total+=value;}
    for(int t=0;t<length;t++)prob[base+t]=scores[base+t]/total;
}
__kernel void decode_values(__global const float *prob,__global const float *v,
                            __global float *out,int length) {
    int i=get_global_id(0);if(i>=2048)return;
    int head=i/128,feature=i%128;float acc=0;
    for(int t=0;t<length;t++)acc=fma(prob[head*length+t],v[t*1024+(head/2)*128+feature],acc);
    out[i]=acc;
}
