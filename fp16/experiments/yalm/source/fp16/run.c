// Full-model experiment using the shared, verified FP16 checkpoint.
#define main qwen3_cli_main
#include "../runq.c"
#undef main
#include "fp16_backend.h"
#include <sys/stat.h>
#include "prefill.h"

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int argmax(const float *p, int count) {
    int best = 0;
    for (int i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, p + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "Non-finite logits; invalid inference result.\n");
            exit(1);
        }
        // Match RKLLM ignore_eos_token: suppress the model EOS token when sampling.
        if (i == 151645) continue;
        if (p[i] > p[best]) best = i;
    }
    return best;
}

static void build_fp16(Transformer *t, const char *path, const __fp16 ***matrices) {
    memset(t, 0, sizeof(*t));
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size < 256) exit(1);
    t->file_size = st.st_size;
    t->data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (t->data == MAP_FAILED) exit(1);
    memcpy(&t->config, t->data, sizeof(Config));
    Config *c = &t->config;
    if (c->magic_number != 0x616a6331 || c->version != 2 || c->seq_len != 512) exit(1);
    GS = c->group_size; // Unused integer matmul scratch fields in the original RunState.
    float *p = (float *)((char *)t->data + 256);
    TransformerWeights *w = &t->weights;
    w->rms_att_weight = p; p += c->n_layers * c->dim;
    w->rms_ffn_weight = p; p += c->n_layers * c->dim;
    w->rms_final_weight = p; p += c->dim;
    w->q_norm_weights = p; p += c->n_layers * c->head_dim;
    w->k_norm_weights = p; p += c->n_layers * c->head_dim;
    const __fp16 *half = (const __fp16 *)p, *embedding = half;
    size_t embedding_count = (size_t)c->vocab_size * c->dim;
    w->token_embedding_table = malloc(embedding_count * sizeof(float));
    if (!w->token_embedding_table) exit(1);
    for (size_t i = 0; i < embedding_count; i++) w->token_embedding_table[i] = half[i];
    half += embedding_count;
    int sizes[] = {c->dim * c->n_heads * c->head_dim,
                   c->dim * c->n_kv_heads * c->head_dim,
                   c->dim * c->n_kv_heads * c->head_dim,
                   c->dim * c->n_heads * c->head_dim,
                   c->dim * c->hidden_dim, c->dim * c->hidden_dim, c->dim * c->hidden_dim};
    *matrices = calloc(7 * c->n_layers + 1, sizeof(**matrices));
    if (!*matrices) exit(1);
    for (int kind = 0; kind < 7; kind++) {
        for (int layer = 0; layer < c->n_layers; layer++) {
            (*matrices)[kind * c->n_layers + layer] = half;
            half += sizes[kind];
        }
    }
    (*matrices)[7 * c->n_layers] = c->shared_classifier ? embedding : half;
    if (!c->shared_classifier) half += embedding_count;
    if ((const char *)half != (const char *)t->data + st.st_size) {
        fprintf(stderr, "FP16 checkpoint size mismatch.\n"); exit(1);
    }
    malloc_run_state(&t->state, c);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s fp16_checkpoint input_ids new_tokens measured_runs [logits_dump]\n", argv[0]);
        return 2;
    }
    int ids[512], count = 0, generated[512], new_tokens = atoi(argv[3]), runs = atoi(argv[4]);
    int warmups=getenv("WARMUP_RUNS")?atoi(getenv("WARMUP_RUNS")):1;
    if(warmups<1)return 2;
    FILE *input = fopen(argv[2], "r");
    if (!input) return 2;
    while (count < 512 && fscanf(input, "%d", &ids[count]) == 1) count++;
    fclose(input);
    if (count < 1 || new_tokens < 2 || count + new_tokens > 512 || runs < 1) return 2;
    int teacher[512],teacher_count=0;
    const char *teacher_path=getenv("TEACHER_IDS");
    if(teacher_path){FILE *f=fopen(teacher_path,"r");if(!f)return 2;
        while(teacher_count<512&&fscanf(f,"%d",&teacher[teacher_count])==1)teacher_count++;
        fclose(f);if(teacher_count!=new_tokens)return 2;}
    double begin = now_ms();
    Transformer t;
    const __fp16 **matrices;
    build_fp16(&t, argv[1], &matrices);
    for (int i = 0; i < count; i++) if (ids[i] < 0 || ids[i] >= t.config.vocab_size) return 2;
    if (!npu_init_fp16(&g_npu, &t.config, matrices)) {
        fprintf(stderr, "Failed to prepare FP16 NPU plans.\n"); return 1;
    }
    free(matrices);
    Prefill prefill;
    prepare_prefill(&prefill,&t.config);
    if(!gpu_attention_init(t.config.n_layers,t.config.seq_len)) {
        fprintf(stderr,"Requested Mali attention backend unavailable; no fallback.\n");return 1;
    }
#ifdef MATCHED_CPU_REFERENCE
    const char *mode = "cpu_reference";
#else
    const char *mode = "npu_fp16";
#endif
    printf("{\"event\":\"initialization\",\"mode\":\"%s\",\"init_ms\":%.3f,\"weights\":\"shared_FP16\",\"npu_cores\":%d}\n", mode, now_ms() - begin,npu_core_count());
    fflush(stdout);
    for (int run = 1-warmups; run <= runs; run++) {
        // Run 0 is a full warmup. Each measured run starts from an empty logical KV cache.
        const char *cool=getenv("COOL_REQUEST_C");
        if(cool){int target=atoi(cool)*1000,temp=0,initial=0;double cool_start=now_ms();
            do{FILE *f=fopen("/sys/class/thermal/thermal_zone0/temp","r");
                if(!f||fscanf(f,"%d",&temp)!=1)exit(1);fclose(f);
                if(!initial)initial=temp;if(temp>target)usleep(500000);
            }while(temp>target);
            fprintf(stderr,"{\"event\":\"cooldown\",\"run\":%d,\"before_millidegrees\":%d,\"after_millidegrees\":%d,\"wait_ms\":%.3f}\n",run,initial,temp,now_ms()-cool_start);}
        npu_matmul_reset_stats(&g_npu);
        gpu_profile_clear();route_clear();
        begin = now_ms();
        float *logits = NULL;
        int serial=getenv("SERIAL_PREFILL")!=NULL;
#ifdef MATCHED_CPU_REFERENCE
        serial=1;
#endif
        if(!serial) logits=forward_prefill(&t,&prefill,ids,count);
        else for (int pos = 0; pos < count; pos++) {
            // Only the last prompt token needs a vocabulary projection in prefill.
            npu_skip_classifier(pos != count - 1);
            logits = forward(&t, ids[pos], pos);
        }
        npu_skip_classifier(0);
        generated[0] = argmax(logits, t.config.vocab_size);
        double first_ms = now_ms() - begin;
        gpu_profile_dump("prefill",run);route_dump("prefill",run);
        if (run == 1-warmups && argc > 5) {
            FILE *f = fopen(argv[5], "wb");
            if (!f || fwrite(logits, sizeof(float), t.config.vocab_size, f) != (size_t)t.config.vocab_size) exit(1);
            fclose(f);
        }
        npu_profile_dump(run<=0?"warmup-prefill":"measured-prefill");
        prefill_profile_dump(&prefill,run<=0?"warmup-prefill":"measured-prefill");
        npu_profile_clear();
        gpu_profile_clear();route_clear();
        unsigned long long pre_npu_ops=g_npu.npu_ops,pre_cpu_ops=g_npu.cpu_ops;
        fprintf(stderr,"{\"event\":\"device_phase\",\"phase\":\"prefill\",\"run\":%d,\"npu_ops\":%llu,\"cpu_ops\":%llu}\n",run,pre_npu_ops,pre_cpu_ops);
        double decode_start = now_ms();
        for (int step = 1; step < new_tokens; step++) {
            if(serial)logits = forward(&t, generated[step - 1], count + step - 1);
            else logits=forward_batch_atpos(&t,&prefill,teacher_path?teacher+step-1:generated+step-1,1,count+step-1);
            generated[step] = argmax(logits, t.config.vocab_size);
        }
        double decode_ms = now_ms() - decode_start;
        gpu_profile_dump("decode",run);route_dump("decode",run);
        fprintf(stderr,"{\"event\":\"device_phase\",\"phase\":\"decode\",\"run\":%d,\"npu_ops\":%llu,\"cpu_ops\":%llu}\n",run,g_npu.npu_ops-pre_npu_ops,g_npu.cpu_ops-pre_cpu_ops);
        npu_profile_dump(run<=0?"warmup-decode":"measured-decode");
        prefill_profile_dump(&prefill,run<=0?"warmup-decode":"measured-decode");
        printf("{\"event\":\"run\",\"mode\":\"%s\",\"warmup\":%s,\"run\":%d,\"prefill_tokens\":%d,\"new_tokens\":%d,"
               "\"first_token_ms\":%.3f,\"decode_steps\":%d,\"decode_ms\":%.3f,\"decode_tps\":%.3f,"
               "\"npu_ops\":%llu,\"cpu_matmul_ops\":%llu,\"generated_ids\":[",
               mode, run <= 0 ? "true" : "false", run, count, new_tokens, first_ms, new_tokens - 1,
               decode_ms, (new_tokens - 1) * 1000.0 / decode_ms, g_npu.npu_ops, g_npu.cpu_ops);
        for (int i = 0; i < new_tokens; i++) printf("%s%d", i ? "," : "", generated[i]);
        printf("]}\n");
        fflush(stdout);
#ifndef MATCHED_CPU_REFERENCE
        // Explicit CPU projection routes are authorized by this experiment.
#endif
    }
    gpu_attention_shutdown();
    npu_matmul_shutdown(&g_npu);
    free_prefill(&prefill);
    free_transformer(&t);
    return 0;
}
