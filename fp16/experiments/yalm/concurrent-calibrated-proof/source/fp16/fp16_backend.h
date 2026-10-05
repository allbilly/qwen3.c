#ifndef MATCHED_FP16_BACKEND_H
#define MATCHED_FP16_BACKEND_H
#include "../npu_matmul.h"
int npu_init_fp16(NpuMatmulContext *ctx, const Config *config, const __fp16 **matrices);
void npu_skip_classifier(int skip);
int npu_matmul_batch(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                     const float *input, float *output, int rows);
void npu_profile_dump(const char *label);
void npu_profile_clear(void);
int npu_core_count(void);
int npu_uses_fused(void);
int npu_matmul_fused(NpuMatmulContext *ctx,int combination,int layer,
                     const float *input,float *output,int rows);
int npu_matmul_fused_to(NpuMatmulContext *ctx,int combination,int layer,const float *input,
                        float *a,float *b,float *c,int rows);
int npu_attention_enabled(void);
int npu_attention_prefill(NpuMatmulContext *ctx,const float *queries,const float *keys,
                          const float *values,float *output,int rows);
int npu_stream_ffn_enabled(void);
int npu_stream_ffn_prefill(NpuMatmulContext *ctx,int layer,const float *input,float *output,int rows);
#endif
