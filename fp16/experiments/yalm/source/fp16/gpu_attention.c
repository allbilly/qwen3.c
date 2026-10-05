#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_attention.h"
#include "gpu_source.h"
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernels[7];
static cl_mem queries,half_queries,output,scores,half_prob,float_prob,half_keys,half_values;
static cl_mem *key_cache,*value_cache;
static int layers,capacity,enabled_prefill,enabled_decode,event_profile;
static const char *kernel_names[]={"pack_kv","prefill_scores","prefill_softmax","prefill_values","decode_scores","decode_softmax","decode_values"};
typedef struct { uint64_t calls,kernel_calls,cache_upload_calls,upload_bytes,download_bytes; double wall_ms,enqueue_ms,blocking_read_ms,upload_device_ms,download_device_ms,kernel_device_ms[7],event_collection_ms; } GpuProfile;
static GpuProfile stats;
typedef struct { cl_event event; int kind; } Event;
static Event events[16];static int event_count;
static double clock_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void check(cl_int result,const char *operation){if(result!=CL_SUCCESS){fprintf(stderr,"OpenCL %s failed: %d; no fallback\n",operation,result);exit(1);}}
static cl_mem allocate(size_t size){cl_int error;cl_mem b=clCreateBuffer(context,CL_MEM_READ_WRITE,size,NULL,&error);check(error,"allocate");return b;}
int gpu_prefill_enabled(void){return enabled_prefill;}
int gpu_decode_enabled(void){return enabled_decode;}
static void record(cl_event event,int kind){if(!event_profile)return;if(event_count>=16)exit(1);events[event_count++]=(Event){event,kind};}
static void collect(void){
    if(!event_profile)return;
    double start=clock_ms();
    for(int i=0;i<event_count;i++) {
        cl_ulong begin,end;
        check(clGetEventProfilingInfo(events[i].event,CL_PROFILING_COMMAND_START,sizeof(begin),&begin,NULL),"event start");
        check(clGetEventProfilingInfo(events[i].event,CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL),"event end");
        if(end<begin)exit(1);double ms=(end-begin)/1e6;
        if(events[i].kind==-1)stats.upload_device_ms+=ms;
        else if(events[i].kind==-2)stats.download_device_ms+=ms;
        else stats.kernel_device_ms[events[i].kind]+=ms;
        clReleaseEvent(events[i].event);
    }
    event_count=0;stats.event_collection_ms+=clock_ms()-start;
}
static void write_buffer(cl_mem buffer,size_t offset,size_t bytes,const void *source){
    cl_event event;double start=clock_ms();
    check(clEnqueueWriteBuffer(queue,buffer,CL_FALSE,offset,bytes,source,0,NULL,event_profile?&event:NULL),"upload");
    stats.enqueue_ms+=clock_ms()-start;stats.upload_bytes+=bytes;
    if(event_profile)record(event,-1);
}
static void arg_mem(cl_kernel kernel,int index,cl_mem buffer){check(clSetKernelArg(kernel,index,sizeof(buffer),&buffer),"argument buffer");}
static void arg_int(cl_kernel kernel,int index,int value){check(clSetKernelArg(kernel,index,sizeof(value),&value),"argument integer");}
static void launch(int kind,size_t count){
    size_t global=(count+63)&~(size_t)63,local=64;
    cl_event event;double start=clock_ms();
    check(clEnqueueNDRangeKernel(queue,kernels[kind],1,NULL,&global,&local,0,NULL,event_profile?&event:NULL),"kernel launch");
    stats.enqueue_ms+=clock_ms()-start;stats.kernel_calls++;
    if(event_profile)record(event,kind);
}
int gpu_attention_init(int count,int max_context){
    const char *mode=getenv("GPU_ATTENTION");
    if(!mode||strcmp(mode,"none")==0)return 1;
    enabled_prefill=strcmp(mode,"prefill")==0||strcmp(mode,"both")==0;
    enabled_decode=strcmp(mode,"decode")==0||strcmp(mode,"both")==0;
    if(!enabled_prefill&&!enabled_decode)return 0;
    layers=count;capacity=max_context;
    if(layers!=28||capacity!=512)return 0;
    event_profile=getenv("GPU_PROFILE")!=NULL;
    cl_platform_id platforms[8];cl_uint platform_count;
    check(clGetPlatformIDs(8,platforms,&platform_count),"platforms");
    for(cl_uint p=0;p<platform_count&&!device;p++) {
        cl_device_id devices[8];cl_uint device_count=0;
        cl_int result=clGetDeviceIDs(platforms[p],CL_DEVICE_TYPE_GPU,8,devices,&device_count);
        if(result==CL_DEVICE_NOT_FOUND)continue;check(result,"GPU devices");
        for(cl_uint d=0;d<device_count;d++){
            char name[256];check(clGetDeviceInfo(devices[d],CL_DEVICE_NAME,sizeof(name),name,NULL),"device name");
            if(strstr(name,"Mali")){device=devices[d];break;}
        }
    }
    if(!device)return 0;
    cl_int error;context=clCreateContext(NULL,1,&device,NULL,NULL,&error);check(error,"context");
    queue=clCreateCommandQueue(context,device,event_profile?CL_QUEUE_PROFILING_ENABLE:0,&error);check(error,"queue");
    const char *text=gpu_source;size_t size=strlen(text);
    program=clCreateProgramWithSource(context,1,&text,&size,&error);check(error,"source");
    error=clBuildProgram(program,1,&device,"-cl-std=CL1.2",NULL,NULL);
    if(error){size_t log_size;clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&log_size);char *log=malloc(log_size+1);clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,log_size,log,NULL);log[log_size]=0;fprintf(stderr,"%s\n",log);free(log);check(error,"compile");}
    for(int i=0;i<7;i++){kernels[i]=clCreateKernel(program,kernel_names[i],&error);check(error,"kernel");}
    queries=allocate((size_t)capacity*2048*4);output=allocate((size_t)capacity*2048*4);
    half_queries=allocate((size_t)capacity*2048*2);
    scores=allocate((size_t)16*capacity*capacity*4);
    half_prob=allocate((size_t)16*capacity*capacity*2);float_prob=allocate(16*capacity*4);
    half_keys=allocate((size_t)capacity*1024*2);half_values=allocate((size_t)capacity*1024*2);
    key_cache=calloc(layers,sizeof(cl_mem));value_cache=calloc(layers,sizeof(cl_mem));
    if(!key_cache||!value_cache)return 0;
    for(int i=0;i<layers;i++){key_cache[i]=allocate((size_t)capacity*1024*4);value_cache[i]=allocate((size_t)capacity*1024*4);}
    char name[256],driver[256];clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,NULL);clGetDeviceInfo(device,CL_DRIVER_VERSION,sizeof(driver),driver,NULL);
    fprintf(stderr,"{\"event\":\"gpu_initialization\",\"device\":\"%s\",\"driver\":\"%s\",\"prefill\":%s,\"decode\":%s,\"event_profiling\":%s}\n",name,driver,enabled_prefill?"true":"false",enabled_decode?"true":"false",event_profile?"true":"false");
    return 1;
}
void gpu_cache_upload(int layer,const float *keys,const float *values,int rows){
    if(!enabled_decode||layer<0||layer>=layers||rows<1||rows>capacity)exit(1);
    double begin=clock_ms();
    write_buffer(key_cache[layer],0,(size_t)rows*1024*4,keys);
    write_buffer(value_cache[layer],0,(size_t)rows*1024*4,values);
    check(clFinish(queue),"cache upload finish");collect();
    stats.cache_upload_calls++;stats.wall_ms+=clock_ms()-begin;
}
void gpu_attention(int layer,const float *q,const float *k,const float *v,float *out,int rows,int position){
    if(layer<0||layer>=layers||rows<1||position<0||position+rows>capacity)exit(1);
    int prefill=position==0&&rows>1,length=position+rows;
    if((prefill&&!enabled_prefill)||(!prefill&&!enabled_decode))exit(1);
    double begin=clock_ms();
    write_buffer(queries,0,(size_t)rows*2048*4,q);
    write_buffer(key_cache[layer],(size_t)position*1024*4,(size_t)rows*1024*4,k);
    write_buffer(value_cache[layer],(size_t)position*1024*4,(size_t)rows*1024*4,v);
    if(prefill) {
        arg_mem(kernels[0],0,key_cache[layer]);arg_mem(kernels[0],1,value_cache[layer]);arg_mem(kernels[0],2,half_keys);arg_mem(kernels[0],3,half_values);arg_int(kernels[0],4,rows);arg_mem(kernels[0],5,queries);arg_mem(kernels[0],6,half_queries);launch(0,(size_t)rows*2048);
        arg_mem(kernels[1],0,half_queries);arg_mem(kernels[1],1,half_keys);arg_mem(kernels[1],2,scores);arg_int(kernels[1],3,rows);launch(1,(size_t)16*rows*rows);
        arg_mem(kernels[2],0,scores);arg_mem(kernels[2],1,half_prob);arg_int(kernels[2],2,rows);launch(2,16*rows);
        arg_mem(kernels[3],0,half_prob);arg_mem(kernels[3],1,half_values);arg_mem(kernels[3],2,output);arg_int(kernels[3],3,rows);launch(3,(size_t)rows*512);
    } else {
        arg_mem(kernels[4],0,queries);arg_mem(kernels[4],1,key_cache[layer]);arg_mem(kernels[4],2,scores);arg_int(kernels[4],3,length);launch(4,16*length);
        arg_mem(kernels[5],0,scores);arg_mem(kernels[5],1,float_prob);arg_int(kernels[5],2,length);launch(5,16);
        arg_mem(kernels[6],0,float_prob);arg_mem(kernels[6],1,value_cache[layer]);arg_mem(kernels[6],2,output);arg_int(kernels[6],3,length);launch(6,2048);
    }
    cl_event event;double read_begin=clock_ms();size_t bytes=(size_t)rows*2048*4;
    check(clEnqueueReadBuffer(queue,output,CL_TRUE,0,bytes,out,0,NULL,event_profile?&event:NULL),"result read");
    stats.blocking_read_ms+=clock_ms()-read_begin;stats.download_bytes+=bytes;
    if(event_profile)record(event,-2);
    collect();stats.calls++;stats.wall_ms+=clock_ms()-begin;
}
void gpu_profile_clear(void){memset(&stats,0,sizeof(stats));}
void gpu_profile_dump(const char *phase,int run){
    if(!enabled_prefill&&!enabled_decode)return;
    fprintf(stderr,"{\"event\":\"gpu_profile\",\"phase\":\"%s\",\"run\":%d,\"warmup\":%s,\"calls\":%llu,\"cache_upload_calls\":%llu,\"kernel_calls\":%llu,\"upload_bytes\":%llu,\"download_bytes\":%llu,\"wall_ms\":%.9f,\"enqueue_ms\":%.9f,\"blocking_read_ms\":%.9f,\"upload_device_ms\":%.9f,\"download_device_ms\":%.9f,\"event_collection_ms\":%.9f,\"kernels\":{",phase,run,run<=0?"true":"false",(unsigned long long)stats.calls,(unsigned long long)stats.cache_upload_calls,(unsigned long long)stats.kernel_calls,(unsigned long long)stats.upload_bytes,(unsigned long long)stats.download_bytes,stats.wall_ms,stats.enqueue_ms,stats.blocking_read_ms,stats.upload_device_ms,stats.download_device_ms,stats.event_collection_ms);
    for(int i=0;i<7;i++)fprintf(stderr,"%s\"%s\":%.9f",i?",":"",kernel_names[i],stats.kernel_device_ms[i]);
    fprintf(stderr,"}}\n");
}
void gpu_attention_shutdown(void){
    if(!queue)return;check(clFinish(queue),"shutdown finish");
    for(int i=0;i<layers;i++){clReleaseMemObject(key_cache[i]);clReleaseMemObject(value_cache[i]);}
    free(key_cache);free(value_cache);
    cl_mem buffers[]={queries,half_queries,output,scores,half_prob,float_prob,half_keys,half_values};
    for(int i=0;i<8;i++)clReleaseMemObject(buffers[i]);
    for(int i=0;i<7;i++)clReleaseKernel(kernels[i]);
    clReleaseProgram(program);clReleaseCommandQueue(queue);clReleaseContext(context);
}
