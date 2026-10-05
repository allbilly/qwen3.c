#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>
#include "linear_gpu.h"
#include "linear_source.h"
static cl_context context;static cl_command_queue queue;static cl_program program;
static cl_kernel gemv,reduce,gemm;static cl_mem *weights,input_buffer,output_buffer,partials;
static __fp16 *half_input;static int count,event_profile,capacity;
typedef struct {uint64_t calls,rows,weights_bytes,upload_bytes,download_bytes,kernels;double pack_ms,wall_ms,upload_ms,kernel_ms,download_ms;} Profile;
static Profile profile;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void check(cl_int rc,const char *what){if(rc!=CL_SUCCESS){fprintf(stderr,"GPU linear %s failed %d; no fallback\n",what,rc);exit(1);}}
static cl_mem alloc(size_t bytes){cl_int e;cl_mem b=clCreateBuffer(context,CL_MEM_READ_WRITE,bytes,NULL,&e);check(e,"allocate");return b;}
static double elapsed(cl_event e){cl_ulong a,b;check(clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_START,sizeof(a),&a,NULL),"event start");check(clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_END,sizeof(b),&b,NULL),"event end");if(b<a)exit(1);clReleaseEvent(e);return (b-a)/1e6;}
int linear_gpu_init(const Config *c,const __fp16 **matrices){
    capacity=c->seq_len;
    cl_platform_id platforms[8];cl_uint np;cl_device_id device=NULL;
    check(clGetPlatformIDs(8,platforms,&np),"platforms");
    for(cl_uint p=0;p<np&&!device;p++){
        cl_device_id devices[8];cl_uint nd=0;cl_int e=clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,8,devices,&nd);
        if(e==CL_DEVICE_NOT_FOUND)continue;check(e,"devices");
        for(cl_uint d=0;d<nd;d++){char name[256];check(clGetDeviceInfo(devices[d],CL_DEVICE_NAME,sizeof(name),name,NULL),"device name");if(strstr(name,"Mali")){device=devices[d];break;}}
    }
    if(!device)return 0;
    cl_int e;context=clCreateContext(NULL,1,&device,NULL,NULL,&e);check(e,"context");
    event_profile=getenv("LINEAR_PROFILE")!=NULL;
    queue=clCreateCommandQueue(context,device,event_profile?CL_QUEUE_PROFILING_ENABLE:0,&e);check(e,"queue");
    const char *source=linear_source;size_t len=strlen(source);
    program=clCreateProgramWithSource(context,1,&source,&len,&e);check(e,"program");
    e=clBuildProgram(program,1,&device,"-cl-std=CL1.2",NULL,NULL);
    if(e){char log[16384];clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL);fprintf(stderr,"%s\n",log);check(e,"build");}
    gemv=clCreateKernel(program,"linear_gemv",&e);check(e,"gemv");
    reduce=clCreateKernel(program,"linear_reduce",&e);check(e,"reduce");
    gemm=clCreateKernel(program,"linear_gemm",&e);check(e,"gemm");
    count=7*c->n_layers+1;weights=calloc(count,sizeof(*weights));if(!weights)return 0;
    int ks[]={1024,1024,1024,2048,1024,3072,1024,1024};
    int ns[]={2048,1024,1024,1024,3072,1024,3072,c->vocab_size};
    size_t total=0;
    for(int kind=0;kind<8;kind++)for(int layer=0;layer<(kind==7?1:c->n_layers);layer++){
        int i=kind==7?7*c->n_layers:kind*c->n_layers+layer,K=ks[kind],N=ns[kind];
        size_t bytes=(size_t)K*N*2;__fp16 *transposed=malloc(bytes);if(!transposed)return 0;
        #pragma omp parallel for
        for(int k0=0;k0<K;k0+=32)for(int n0=0;n0<N;n0+=32)
            for(int k=k0;k<k0+32&&k<K;k++)for(int n=n0;n<n0+32&&n<N;n++)transposed[(size_t)k*N+n]=matrices[i][(size_t)n*K+k];
        weights[i]=alloc(bytes);check(clEnqueueWriteBuffer(queue,weights[i],CL_TRUE,0,bytes,transposed,0,NULL,NULL),"weights upload");free(transposed);total+=bytes;
    }
    half_input=malloc((size_t)capacity*3072*2);if(!half_input)return 0;
    input_buffer=alloc((size_t)capacity*3072*2);output_buffer=alloc((size_t)capacity*4096*4);partials=alloc((size_t)c->vocab_size*8*4);
    char name[256];clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,NULL);
    fprintf(stderr,"{\"event\":\"linear_gpu_init\",\"device\":\"%s\",\"weight_bytes\":%zu,\"projection_count\":%d,\"fp16_weights\":true,\"event_profiling\":%s}\n",name,total,count,event_profile?"true":"false");
    return 1;
}
static void memarg(cl_kernel k,int i,cl_mem b){check(clSetKernelArg(k,i,sizeof(b),&b),"buffer argument");}
static void intarg(cl_kernel k,int i,int v){check(clSetKernelArg(k,i,sizeof(v),&v),"integer argument");}
void linear_gpu_run(int index,const float *input,float *output,int rows,int K,int N,int split_down){
    if(index<0||index>=count||rows<1||rows>capacity||N%4||K%8)exit(1);
    double begin=now();
    for(int i=0;i<rows*K;i+=8)vst1q_f16(half_input+i,vcombine_f16(vcvt_f16_f32(vld1q_f32(input+i)),vcvt_f16_f32(vld1q_f32(input+i+4))));
    profile.pack_ms+=now()-begin;
    cl_event upload,compute[2],download;
    check(clEnqueueWriteBuffer(queue,input_buffer,CL_FALSE,0,(size_t)rows*K*2,half_input,0,NULL,event_profile?&upload:NULL),"input write");
    if(rows==1){
        int parts=8;memarg(gemv,0,input_buffer);memarg(gemv,1,weights[index]);memarg(gemv,2,partials);intarg(gemv,3,K);intarg(gemv,4,N);intarg(gemv,5,parts);
        size_t global[]={((N/4+63)/64)*64,parts},local[]={64,1};
        check(clEnqueueNDRangeKernel(queue,gemv,2,NULL,global,local,0,NULL,event_profile?&compute[0]:NULL),"gemv launch");
        memarg(reduce,0,partials);memarg(reduce,1,output_buffer);intarg(reduce,2,N);intarg(reduce,3,parts);
        size_t g=((N/4+63)/64)*64,l=64;
        check(clEnqueueNDRangeKernel(queue,reduce,1,NULL,&g,&l,0,NULL,event_profile?&compute[1]:NULL),"reduce launch");profile.kernels+=2;
    }else{
        memarg(gemm,0,input_buffer);memarg(gemm,1,weights[index]);memarg(gemm,2,output_buffer);
        intarg(gemm,3,rows);intarg(gemm,4,K);intarg(gemm,5,N);intarg(gemm,6,split_down);
        size_t global[]={((N/4+63)/64)*64,(rows+7)/8},local[]={64,1};
        check(clEnqueueNDRangeKernel(queue,gemm,2,NULL,global,local,0,NULL,event_profile?&compute[0]:NULL),"gemm launch");profile.kernels++;
    }
    check(clEnqueueReadBuffer(queue,output_buffer,CL_TRUE,0,(size_t)rows*N*4,output,0,NULL,event_profile?&download:NULL),"output read");
    if(event_profile){profile.upload_ms+=elapsed(upload);profile.kernel_ms+=elapsed(compute[0]);if(rows==1)profile.kernel_ms+=elapsed(compute[1]);profile.download_ms+=elapsed(download);}
    profile.calls++;profile.rows+=rows;profile.weights_bytes+=(size_t)K*N*2;
    profile.upload_bytes+=(size_t)rows*K*2;profile.download_bytes+=(size_t)rows*N*4;profile.wall_ms+=now()-begin;
}
void linear_gpu_clear(void){memset(&profile,0,sizeof(profile));}
void linear_gpu_dump(const char *phase,int run){if(!context)return;
    fprintf(stderr,"{\"event\":\"linear_gpu_profile\",\"phase\":\"%s\",\"run\":%d,\"warmup\":%s,\"calls\":%llu,\"rows\":%llu,\"weight_bytes\":%llu,\"upload_bytes\":%llu,\"download_bytes\":%llu,\"kernels\":%llu,\"pack_ms\":%.9f,\"wall_ms\":%.9f,\"upload_ms\":%.9f,\"kernel_ms\":%.9f,\"download_ms\":%.9f}\n",phase,run,run<=0?"true":"false",(unsigned long long)profile.calls,(unsigned long long)profile.rows,(unsigned long long)profile.weights_bytes,(unsigned long long)profile.upload_bytes,(unsigned long long)profile.download_bytes,(unsigned long long)profile.kernels,profile.pack_ms,profile.wall_ms,profile.upload_ms,profile.kernel_ms,profile.download_ms);
}
void linear_gpu_close(void){if(!context)return;check(clFinish(queue),"finish");for(int i=0;i<count;i++)clReleaseMemObject(weights[i]);free(weights);free(half_input);clReleaseMemObject(input_buffer);clReleaseMemObject(output_buffer);clReleaseMemObject(partials);clReleaseKernel(gemv);clReleaseKernel(reduce);clReleaseKernel(gemm);clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);context=NULL;}
