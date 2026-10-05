#ifndef GPU_ATTENTION_H
#define GPU_ATTENTION_H
int gpu_attention_init(int layers,int context);
int gpu_prefill_enabled(void);
int gpu_decode_enabled(void);
void gpu_cache_upload(int layer,const float *keys,const float *values,int rows);
void gpu_attention(int layer,const float *queries,const float *keys,const float *values,
                   float *output,int rows,int position);
void gpu_profile_clear(void);
void gpu_profile_dump(const char *phase,int run);
void gpu_attention_shutdown(void);
#endif
