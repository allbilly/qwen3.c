#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// K-major persistent FP16 weights let adjacent lanes read adjacent channels.
// FP32 inputs are rounded once on the host to match NPU projection operands.
__kernel void linear_gemv(__global const half *x,__global const half *w,
                          __global float *partial,int K,int N,int parts) {
    int n=get_global_id(0)*4,part=get_global_id(1);
    if(n>=N)return;
    float4 acc=(float4)(0);
    int first=part*(K/parts),last=(part+1)*(K/parts);
    for(int k=first;k<last;k++)acc=fma((float4)((float)x[k]),vload_half4(0,w+(size_t)k*N+n),acc);
    vstore4(acc,0,partial+(size_t)part*N+n);
}
__kernel void linear_reduce(__global const float *partial,__global float *out,int N,int parts) {
    int n=get_global_id(0)*4;if(n>=N)return;
    float4 acc=vload4(0,partial+n);
    for(int p=1;p<parts;p++)acc+=vload4(0,partial+(size_t)p*N+n);
    vstore4(acc,0,out+n);
}
// Eight prompt rows reuse each weight. Every lane writes contiguous channels.
// Down prefill preserves three K=1024 partial sums and their addition order.
__kernel void linear_gemm(__global const half *x,__global const half *w,
                          __global float *out,int M,int K,int N,int split_down) {
    int n=get_global_id(0)*4,m=get_global_id(1)*8;if(n>=N)return;
    float4 total[8];for(int r=0;r<8;r++)total[r]=(float4)(0);
    int chunks=split_down?3:1,block=K/chunks;
    for(int c=0;c<chunks;c++) {
        float4 acc[8];for(int r=0;r<8;r++)acc[r]=(float4)(0);
        for(int k=c*block;k<(c+1)*block;k++) {
            float4 weight=vload_half4(0,w+(size_t)k*N+n);
            for(int r=0;r<8;r++)if(m+r<M)acc[r]=fma((float4)((float)x[(size_t)(m+r)*K+k]),weight,acc[r]);
        }
        for(int r=0;r<8;r++)total[r]+=acc[r];
    }
    for(int r=0;r<8;r++)if(m+r<M)vstore4(total[r],0,out+(size_t)(m+r)*N+n);
}
