#ifndef YALM_LINEAR_GPU_H
#define YALM_LINEAR_GPU_H
#include "source/fp16/fp16_backend.h"
int linear_gpu_init(const Config *,const __fp16 **);
void linear_gpu_run(int index,const float *input,float *output,int rows,int K,int N,int split_down);
void linear_gpu_clear(void);
void linear_gpu_dump(const char *phase,int run);
void linear_gpu_close(void);
#endif
