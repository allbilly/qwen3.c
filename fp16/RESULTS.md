# Previous matched baseline

Current roofline measurements and register optimizations are in
[ROOFLINE.md](ROOFLINE.md). The measurements below are preserved as the
baseline before those two task-layout changes.

The custom engine is faster in both warm full-model phases on the two matched
FP16 workloads, with modest gains. All 311 effective tensors match; integer
quantization is disabled in both engines. The original unmatched Q8_0 versus
W8A8 numbers are withdrawn.

| Input tokens | Custom TTFT ms | Stock TTFT ms | Custom tokens/s | Stock tokens/s | Measured runs per engine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 258.119 | 273.401 | 20.4995 | 19.2985 | 10 |
| 256 | 548.218 | 566.253 | 19.6495 | 18.1580 | 8 |

Both engines use three NPU cores, four big CPU cores, identical fixed clocks,
32 greedy generated tokens, empty KV history, and two complete warmups per
fresh process. Initialization is excluded. The first decode call builds the
head-contiguous FP32 KV layout and its cost is included in decode timing.
128-token blocks alternate custom/stock/stock/custom; 256-token blocks use the
reverse order. All warm and measured output IDs match.

The final binary also matched all 16 generated IDs on five independent checks
with 1, 24, 73, 128, and 256 input tokens. The current strict Hello reference
diagnostic passes its unchanged 1e-4 limit with relative RMSE
4.68639732e-05. Stock logit errors on the five checks range
from 0.00571%
to 0.15295%.
Floating point results are not bit-identical. The previous failed diagnostic
and all unsuccessful performance experiments are retained.

Implemented: persistent direct-register DMA tasks, batched and fused three-core
projections, NPU prompt attention, streamed FFN packing, FP32 NEON SwiGLU, and
contiguous FP32 KV reads. The executable is built by `make fp16`; usage is in
[README.md](README.md). The tested checkpoint is
`/home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16`.

Evidence: [benchmark-summary.json](benchmark-summary.json), and the complete
workspace report at `/home/orangepi/qwen3-bench/matched/README.md`. The independent
audit recomputed timings from raw logs, checked artifact/source/binary hashes,
tensor identity, clock samples, warmups, outputs, resources, and actual restored
sysfs settings. The prompt helper was built and run successfully. No NPU test
process remains running and board clock settings are restored.

INTENT: code used strided KV reads; the task expects faster warm inference; the README requires matched FP16 weights and output checks.

TWINS: searched RKLLM_INFER_GET_LOGITS in both active benchmark runners - found 0 other sites with the one-token GET_LOGITS-to-GENERATE transition.
