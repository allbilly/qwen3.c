#define main qwen3_cli_main
#include "../runq.c"
#undef main
#include <sys/resource.h>

static double benchmark_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec * 1000.0 + value.tv_nsec / 1e6;
}

static int benchmark_top1(const float *values, int count) {
    int best = 0;
    for (int i = 0; i < count; ++i) {
        if (!isfinite(values[i])) { fprintf(stderr, "Nonfinite logits\n"); exit(1); }
        if (values[i] > values[best]) best = i;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s checkpoint input-ids output-tokens trials logits-output\n", argv[0]);
        return 2;
    }
    int ids[512], count = 0, tokens[512], steps = atoi(argv[3]), trials = atoi(argv[4]);
    FILE *input = fopen(argv[2], "r");
    if (!input) return 2;
    while (count < 512 && fscanf(input, "%d", ids + count) == 1) ++count;
    fclose(input);
    if (count < 1 || steps < 2 || trials < 1 || count + steps > 512) return 2;
    Transformer model;
    build_transformer(&model, argv[1], 512);
    if (count + steps > model.config.seq_len) return 2;
    for (int i = 0; i < count; ++i) if (ids[i] < 0 || ids[i] >= model.config.vocab_size) return 2;
    int ready = npu_matmul_init(&g_npu, &model.config, &model.weights);
    const char *request = getenv("ANE") ? getenv("ANE") : getenv("NPU");
    if (request && strcmp(request, "0") && !ready) return 1;
    FILE *dump = fopen(argv[5], "wb");
    if (!dump) return 1;
    for (int trial = 0; trial <= trials; ++trial) {
        npu_matmul_reset_stats(&g_npu);
        double start = benchmark_ms();
        float *logits = forward_prompt(&model, ids, count, 0);
        double prefill = benchmark_ms() - start;
        tokens[0] = benchmark_top1(logits, model.config.vocab_size);
        if (trial == 0 && fwrite(logits, sizeof(float), model.config.vocab_size, dump) != (size_t)model.config.vocab_size) return 1;
        start = benchmark_ms();
        for (int step = 1; step < steps; ++step) {
            logits = forward(&model, tokens[step-1], count+step-1);
            tokens[step] = benchmark_top1(logits, model.config.vocab_size);
            if (trial == 0 && fwrite(logits, sizeof(float), model.config.vocab_size, dump) != (size_t)model.config.vocab_size) return 1;
        }
        double decode = benchmark_ms() - start;
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage)) return 1;
        uint64_t rss = (uint64_t)usage.ru_maxrss;
#ifndef __APPLE__
        rss *= 1024;
#endif
        printf("{\"event\":\"run\",\"warmup\":%s,\"run\":%d,\"prompt_tokens\":%d,\"new_tokens\":%d,"
               "\"prefill_ms\":%.6f,\"decode_ms\":%.6f,\"decode_tps\":%.6f,"
               "\"npu_ops\":%llu,\"cpu_ops\":%llu,\"peak_rss_bytes\":%llu,\"ids\":[",
               trial == 0 ? "true" : "false", trial, count, steps, prefill, decode,
               (steps-1)*1000.0/decode, g_npu.npu_ops, g_npu.cpu_ops, (unsigned long long)rss);
        for (int i = 0; i < steps; ++i) printf("%s%d", i ? "," : "", tokens[i]);
        puts("]}");
        fflush(stdout);
    }
    if (fclose(dump)) return 1;
    npu_matmul_shutdown(&g_npu);
    free_transformer(&model);
    return 0;
}
