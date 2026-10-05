#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "source/fp16/fp16_backend.h"
#include "router.h"
void softmax(float *v,int length){float maximum=0,total=0;for(int i=0;i<length;i++)if(v[i]>maximum)maximum=v[i];for(int i=0;i<length;i++){v[i]=expf(v[i]-maximum);total+=v[i];}for(int i=0;i<length;i++)v[i]/=total;}
int main(int argc,char **argv){
    if(argc!=2)return 2;
    int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))return 2;
    char *data=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)return 2;close(fd);
    Config c;memcpy(&c,data,sizeof(c));if(c.magic_number!=0x616a6331||c.version!=2||c.dim!=1024||c.n_layers!=28)return 2;
    size_t norm_count=2*c.n_layers*c.dim+c.dim+2*c.n_layers*c.head_dim;
    const __fp16 *embedding=(const __fp16 *)(data+256+norm_count*4),*p=embedding+(size_t)c.vocab_size*c.dim;
    const __fp16 *matrices[197];int ks[]={1024,1024,1024,2048,1024,3072,1024,1024},ns[]={2048,1024,1024,1024,3072,1024,3072,c.vocab_size};
    for(int kind=0;kind<7;kind++)for(int layer=0;layer<28;layer++){matrices[kind*28+layer]=p;p+=(size_t)ks[kind]*ns[kind];}matrices[196]=embedding;
    if((const char *)p!=data+st.st_size)return 2;
    NpuMatmulContext ctx;if(!npu_init_fp16(&ctx,&c,matrices))return 1;
    float *input=malloc(64*3072*4),*output=malloc((size_t)64*c.vocab_size*4);if(!input||!output)return 1;
    int cases[][2]={{0,1},{0,24},{0,64},{3,24},{5,24},{7,1}};
    for(int test=0;test<6;test++){
        int kind=cases[test][0],rows=cases[test][1],K=ks[kind],N=ns[kind];
        for(int i=0;i<rows*K;i++)input[i]=(float)((i*29)%257-128)/83;
        route_phase(rows==1?1:0);npu_matmul_batch(&ctx,kind,0,input,output,rows);
        double error=0,energy=0,max_error=0;const __fp16 *w=matrices[kind==7?196:kind*28];
        for(int row=0;row<rows;row++)for(int n=0;n<N;n++){
            double sum=0;int chunks=kind==5&&rows>1?3:1;
            if(chunks==1)for(int k=0;k<K;k++)sum+=(double)(__fp16)input[row*K+k]*(double)w[(size_t)n*K+k];
            else{float total=0;for(int b=0;b<3;b++){double part=0;for(int k=b*1024;k<(b+1)*1024;k++)part+=(double)(__fp16)input[row*K+k]*(double)w[(size_t)n*K+k];total+=(float)part;}sum=total;}
            uint32_t bits;memcpy(&bits,output+(size_t)row*N+n,4);if((bits&0x7f800000u)==0x7f800000u)return 1;
            double diff=output[(size_t)row*N+n]-sum;error+=diff*diff;energy+=sum*sum;if(fabs(diff)>max_error)max_error=fabs(diff);
        }
        double relative=sqrt(error/energy);int pass=relative<1e-5;
        printf("{\"event\":\"linear_check\",\"kind\":%d,\"rows\":%d,\"K\":%d,\"N\":%d,\"outputs_checked\":%d,\"relative_rmse\":%.12g,\"max_absolute_error\":%.12g,\"limit\":0.00001,\"passed\":%s}\n",kind,rows,K,N,rows*N,relative,max_error,pass?"true":"false");fflush(stdout);if(!pass)return 1;
    }
    npu_matmul_shutdown(&ctx);free(input);free(output);munmap(data,st.st_size);return 0;
}
