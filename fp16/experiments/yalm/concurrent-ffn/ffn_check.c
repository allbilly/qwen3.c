#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "concurrent_ffn.h"
void softmax(float *p,int n){float max=p[0],total=0;for(int i=1;i<n;i++)if(p[i]>max)max=p[i];for(int i=0;i<n;i++){p[i]=expf(p[i]-max);total+=p[i];}for(int i=0;i<n;i++)p[i]/=total;}
int main(int argc,char **argv){
    if(argc!=2)return 2;
    int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))return 2;
    char *data=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)return 2;close(fd);
    Config c;memcpy(&c,data,sizeof(c));if(c.magic_number!=0x616a6331||c.version!=2||c.dim!=1024||c.n_layers!=28)return 2;
    size_t norms=2*c.n_layers*c.dim+c.dim+2*c.n_layers*c.head_dim;
    const __fp16 *embedding=(const __fp16 *)(data+256+norms*4),*p=embedding+(size_t)c.vocab_size*c.dim,*matrices[197];
    int ks[]={1024,1024,1024,2048,1024,3072,1024},ns[]={2048,1024,1024,1024,3072,1024,3072};
    for(int kind=0;kind<7;kind++)for(int layer=0;layer<28;layer++){matrices[kind*28+layer]=p;p+=(size_t)ks[kind]*ns[kind];}matrices[196]=embedding;if((const char*)p!=data+st.st_size)return 2;
    NpuMatmulContext ctx;if(!npu_init_fp16(&ctx,&c,matrices))return 1;
    float *input=malloc(64*1024*4),*gate=malloc(64*3072*4),*up=malloc(64*3072*4);if(!input||!gate||!up)return 1;
    int cases[]={1,24,64};
    for(int test=0;test<3;test++){
        int rows=cases[test];for(int i=0;i<rows*1024;i++)input[i]=(float)((i*29)%257-128)/83;
        ffn_gate_up(&ctx,0,input,gate,up,rows);
        double error=0,energy=0,max_error=0;
        for(int part=0;part<2;part++)for(int row=0;row<rows;row++)for(int n=0;n<3072;n++){
            const __fp16 *w=matrices[(part?6:4)*28]+(size_t)n*1024;double sum=0;
            for(int k=0;k<1024;k++)sum+=(double)(__fp16)input[row*1024+k]*(double)w[k];
            float value=(part?up:gate)[row*3072+n];uint32_t bits;memcpy(&bits,&value,4);if((bits&0x7f800000u)==0x7f800000u)return 1;
            double diff=value-sum;error+=diff*diff;energy+=sum*sum;if(fabs(diff)>max_error)max_error=fabs(diff);
        }
        double relative=sqrt(error/energy);int pass=relative<1e-5;
        printf("{\"event\":\"ffn_primitive_check\",\"rows\":%d,\"outputs_checked\":%d,\"relative_rmse\":%.12g,\"max_absolute_error\":%.12g,\"limit\":0.00001,\"passed\":%s}\n",rows,rows*6144,relative,max_error,pass?"true":"false");fflush(stdout);if(!pass)return 1;
    }
    npu_matmul_shutdown(&ctx);free(input);free(gate);free(up);munmap(data,st.st_size);return 0;
}
