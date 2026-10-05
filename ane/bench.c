// Full-model benchmark, also usable for numerical comparison on fixed tokens.
#define main qwen3_cli_main
#include "../runq.c"
#undef main
#include "ane_matmul.h"
#include <sys/stat.h>

static double milliseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int top1(const float *logits, int n) {
    int best = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(logits[i])) { fprintf(stderr, "Nonfinite logit %d\n", i); exit(1); }
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr,"Usage: %s checkpoint input_ids new_tokens runs [logits_dump [teacher_tokens]]\n",argv[0]);
        return 2;
    }
    int count = 0, ids[512], tokens[512], teachers[512], teacher_count = 0;
    int steps = atoi(argv[3]), runs = atoi(argv[4]);
    FILE *f = fopen(argv[2], "r");
    if (!f) return 2;
    while (count < 512 && fscanf(f, "%d", ids + count) == 1) count++;
    fclose(f);
    if (count < 1 || steps < 2 || count + steps > 512 || runs < 1) return 2;
    if (argc == 7) {
        f = fopen(argv[6], "r");
        if (!f) return 2;
        while (teacher_count < 512 && fscanf(f,"%d",teachers+teacher_count)==1) teacher_count++;
        fclose(f);
        if (teacher_count < steps) return 2;
    }
    Transformer t;
    double begin = milliseconds();
    build_transformer(&t, argv[1], 512);
    if (count + steps > t.config.seq_len) return 2;
    for (int i=0;i<count;i++) if (ids[i]<0 || ids[i]>=t.config.vocab_size) return 2;
    for (int i=0;i<teacher_count;i++) if (teachers[i]<0 || teachers[i]>=t.config.vocab_size) return 2;
    const char *env = getenv("ANE") ? getenv("ANE") : getenv("NPU");
    int requested = env && strcmp(env,"0");
    if (!npu_matmul_init(&g_npu,&t.config,&t.weights) && requested) return 1;
    const char *mode=g_npu.enabled?(ane_decode_enabled(&g_npu)?"ane":"ane_prefill_cpu_decode"):"cpu";
    printf("{\"event\":\"init\",\"backend\":\"%s\",\"init_ms\":%.3f}\n",mode,milliseconds()-begin);
    fflush(stdout);
    FILE *dump = argc >= 6 ? fopen(argv[5],"wb") : NULL;
    if (argc>=6 && !dump) return 2;
    // One full warmup, followed by runs from an empty logical KV cache.
    for (int run=0;run<=runs;run++) {
        npu_matmul_reset_stats(&g_npu);
        begin=milliseconds();
        float *logits=NULL;
        logits=forward_prompt(&t,ids,count,0);
        double prefill=milliseconds()-begin;
        tokens[0]=top1(logits,t.config.vocab_size);
        if (run==0 && dump && fwrite(logits,sizeof(float),t.config.vocab_size,dump)!=(size_t)t.config.vocab_size) return 1;
        begin=milliseconds();
        for (int step=1;step<steps;step++) {
            logits=forward(&t,teacher_count?teachers[step-1]:tokens[step-1],count+step-1);
            tokens[step]=top1(logits,t.config.vocab_size);
            if (run==0 && dump && fwrite(logits,sizeof(float),t.config.vocab_size,dump)!=(size_t)t.config.vocab_size) return 1;
        }
        double decode=milliseconds()-begin;
        printf("{\"event\":\"run\",\"backend\":\"%s\",\"warmup\":%s,\"run\":%d,\"prompt_tokens\":%d,\"new_tokens\":%d,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"decode_tps\":%.3f,\"ane_ops\":%llu,\"cpu_ops\":%llu,\"ids\":[",
               mode,run==0?"true":"false",run,count,steps,prefill,decode,(steps-1)*1000/decode,g_npu.npu_ops,g_npu.cpu_ops);
        for (int i=0;i<steps;i++) printf("%s%d",i?",":"",tokens[i]);
        printf("]}\n"); fflush(stdout);
    }
    if (dump && fclose(dump)) return 1;
    npu_matmul_shutdown(&g_npu);
    free_transformer(&t);
    return 0;
}
