#ifndef CONCURRENT_FFN_H
#define CONCURRENT_FFN_H
#include "source/fp16/fp16_backend.h"
int ffn_init(NpuMatmulContext *,const Config *,const __fp16 **);
void ffn_phase(int);
int ffn_active(void);
int ffn_gate_up(NpuMatmulContext *,int,const float *,float *,float *,int);
int ffn_prefill(NpuMatmulContext *,int,const float *,float *,int);
void ffn_clear(void);
void ffn_dump(const char *,int);
void ffn_close(void);
#endif
