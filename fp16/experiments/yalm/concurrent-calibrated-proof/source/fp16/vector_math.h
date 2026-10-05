// FP32 NEON exp with range reduction and a degree-seven Taylor polynomial.
// The residual lies in [-ln(2)/2, ln(2)/2]. Unusual inputs use scalar libm.
#ifndef QWEN3_VECTOR_MATH_H
#define QWEN3_VECTOR_MATH_H
#include <arm_neon.h>
#include <math.h>

static inline float32x4_t qwen_exp4(float32x4_t x) {
    uint32x4_t magnitude=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x7fffffffu));
    if(vmaxvq_u32(vcgeq_u32(magnitude,vdupq_n_u32(0x42a00000u)))) {
        float lane[4];vst1q_f32(lane,x);
        for(int i=0;i<4;i++)lane[i]=expf(lane[i]);
        return vld1q_f32(lane);
    }
    float32x4_t n=vrndaq_f32(vmulq_n_f32(x,0x1.715476p+0f));
    float32x4_t r=vfmsq_n_f32(x,n,0x1.62e4p-1f);
    r=vfmsq_n_f32(r,n,0x1.7f7d1cp-20f);
    float32x4_t p=vdupq_n_f32(1.0f/5040.0f);
    p=vfmaq_f32(vdupq_n_f32(1.0f/720.0f),r,p);
    p=vfmaq_f32(vdupq_n_f32(1.0f/120.0f),r,p);
    p=vfmaq_f32(vdupq_n_f32(1.0f/24.0f),r,p);
    p=vfmaq_f32(vdupq_n_f32(1.0f/6.0f),r,p);
    p=vfmaq_f32(vdupq_n_f32(0.5f),r,p);
    p=vfmaq_f32(vdupq_n_f32(1.0f),r,p);
    p=vfmaq_f32(vdupq_n_f32(1.0f),r,p);
    uint32x4_t exponent=vaddq_u32(vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)),23),vdupq_n_u32(0x3f800000u));
    return vmulq_f32(p,vreinterpretq_f32_u32(exponent));
}

static inline float32x4_t qwen_swiglu4(float32x4_t gate,float32x4_t up) {
    float32x4_t one=vdupq_n_f32(1.0f);
    float32x4_t sigmoid=vdivq_f32(one,vaddq_f32(one,qwen_exp4(vnegq_f32(gate))));
    return vmulq_f32(gate,vmulq_f32(up,sigmoid));
}
#endif
