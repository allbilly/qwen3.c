#ifndef QWEN3_ANE_MATMUL_H
#define QWEN3_ANE_MATMUL_H
#include "npu_matmul.h"

typedef struct AneDevice AneDevice;
typedef struct AnePlan AnePlan;

// Plans own packed, resident Q8 -> FP16 weights; execution allocates nothing.
AneDevice *ane_device_open(void);
void ane_device_close(AneDevice *device);
AnePlan *ane_plan_create(AneDevice *device, const QuantizedTensor *weights,
                         int inputs, int outputs, int group_size);
int ane_plan_run(AnePlan *plan, const float *input, float *output);
int ane_plan_run_batch(AnePlan *plan, const float *input, float *output, int rows);
int npu_matmul_run_batch(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                         const float *input, float *output, int rows);
void ane_plan_free(AnePlan *plan);
unsigned long long ane_device_submissions(const AneDevice *device);
int ane_decode_enabled(const NpuMatmulContext *ctx);
#endif
