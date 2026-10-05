// Check wide native attention matrices without changing submit metadata.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "concurrent_ffn.c"
void softmax(float *p,int n){float maximum=p[0],total=0;for(int i=1;i<n;i++)if(p[i]>maximum)maximum=p[i];for(int i=0;i<n;i++){p[i]=expf(p[i]-maximum);total+=p[i];}for(int i=0;i<n;i++)p[i]/=total;}
static float pattern(int i,int seed){return ((i*seed%127)-63)/97.0f;}
int main(int argc,char **argv){
    if(argc!=2)return 2;
    setenv("BENCH_CONTEXT","4128",1);setenv("DENSE_PREFILL","npu",1);setenv("DENSE_DECODE","npu",1);
    unsetenv("FFN_CPU_CHANNELS");unsetenv("FFN_GPU_CHANNELS");unsetenv("FFN_MODE");unsetenv("GPU_ATTENTION");
    int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))return 2;
    char *data=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)return 2;close(fd);
    Config c;memcpy(&c,data,sizeof(c));if(c.magic_number!=0x616a6331||c.version!=2||c.dim!=1024||c.n_layers!=28)return 2;c.seq_len=4128;
    size_t norms=2*c.n_layers*c.dim+c.dim+2*c.n_layers*c.head_dim;
    const __fp16 *embedding=(const __fp16 *)(data+256+norms*4),*p=embedding+(size_t)c.vocab_size*c.dim,*matrices[197];
    int ks[]={1024,1024,1024,2048,1024,3072,1024},ns[]={2048,1024,1024,1024,3072,1024,3072};
    for(int kind=0;kind<7;kind++)for(int layer=0;layer<28;layer++){matrices[kind*28+layer]=p;p+=(size_t)ks[kind]*ns[kind];}matrices[196]=embedding;if((const char*)p!=data+st.st_size)return 2;
    NpuMatmulContext ctx;if(!npu_init_fp16(&ctx,&c,matrices))return 1;
    int lengths[]={512,1024,2048,4096};int status=0;
    for(int ix=0;ix<4&&!status;ix++)for(int pv=0;pv<2&&!status;pv++){
        int T=lengths[ix],M=pv?4*NPU_CBUF_BANK_SIZE/(T*2):64;
        if(M>64)M=64;
        int K=pv?T:128,N=pv?128:T;Plan *p=pv?attention_value[0]:attention_score[0];
        attention_shape(p,K,N);configure_rows(p,M,1);p->bundle_count=1;
        __fp16 *weights=(__fp16*)((char*)p->mem.weights+REGCMD_RESERVED);
        float *input=malloc((size_t)M*K*4),*out=malloc((size_t)M*N*4);
        if(!input||!out){status=2;break;}
        for(int i=0;i<M*K;i++)input[i]=pattern(i,17);
        for(int n=0;n<N;n++)for(int k=0;k<K;k++)weights[weight_fp16(K,n+1,k+1)]=(__fp16)pattern(n*K+k,13);
        mem_sync(device_fd,p->mem.weights_obj,REGCMD_RESERVED,(size_t)K*N*2,RKNPU_MEM_SYNC_TO_DEVICE);
        attention_pack(p,input,K,M);
        if(submit_bundle(p)<0){status=3;break;}
        attention_read(p,out,N,M);
        double error=0,energy=0,maximum=0;
        for(int m=0;m<M;m++)for(int n=0;n<N;n++){
            double expected=0;
            for(int k=0;k<K;k++)expected+=(double)(__fp16)input[m*K+k]*(double)(__fp16)pattern(n*K+k,13);
            double delta=out[m*N+n]-expected;
            uint32_t bits;memcpy(&bits,out+m*N+n,4);if((bits&0x7f800000u)==0x7f800000u){status=4;break;}
            error+=delta*delta;energy+=expected*expected;if(fabs(delta)>maximum)maximum=fabs(delta);
        }
        double relative=sqrt(error/energy);int passed=!status&&relative<1e-5;
        printf("{\"event\":\"wide_attention_check\",\"kind\":\"%s\",\"context\":%d,\"rows\":%d,\"K\":%d,\"N\":%d,\"checked\":%d,\"relative_rmse\":%.12g,\"max_absolute_error\":%.12g,\"limit\":0.00001,\"passed\":%s}\n",pv?"PV":"QK",T,M,K,N,M*N,relative,maximum,passed?"true":"false");fflush(stdout);
        free(input);free(out);if(!passed){status=5;break;}
    }
    npu_matmul_shutdown(&ctx);munmap(data,st.st_size);return status;
}
