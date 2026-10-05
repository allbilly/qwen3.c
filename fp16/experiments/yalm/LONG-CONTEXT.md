# 1K / 2K / 4K prompt benchmark — WIP

The isolated context extension builds as `f8b4f9eb...`. The shared pinned
Qwen3-0.6B FP16 container is unchanged (`c9280793...`), including its 512-token
header. `BENCH_CONTEXT=4128` expands runtime KV/scratch capacity only. The
pinned model configuration supports 40960 positions with the existing 1e6
RoPE base; this runner measures at most 4096 input tokens plus 32 outputs.

Host, CPU projection, Mali OpenCL and concurrent FFN scratch buffers use the
runtime capacity. Native attention reserves 64 query rows in its BOs and caps
actual submissions using the existing four-data-bank budget: 64/32/16 rows
for 1024/2048/4096-key PV matrices. Task masks, flags, timeout, domain and the
register emitter are preserved; all NPU initialization/sync/submits stay
serial on the main thread.

Eight [wide QK/PV primitive cases](long-context/wide-attention-check.jsonl)
check all 514,048 outputs against double accumulation of identical FP16
operands. Largest relative RMSE is `2.02905781962e-7`, below the unchanged
`1e-5` limit. Numerical timings are not performance claims.

All six 128-token short regressions pass the unchanged full-model quality,
prediction, device-count and fixed/restored-clock audits. Full-vocabulary
first-token logits are [bit-identical to the previous corresponding
routes](long-context/long-context-short-parity.json). The [independent short
audit](long-context/long-context-short-audit.json) retains all phase/request
measurements and checks actual execution; all four requests per route pass.

Long measurements are pending at this milestone. The first 51°C attempt
completed 1K/2K requests but aborted normally before the second 4K warmup
when bounded cooling failed. Its raw evidence is retained under
`evidence/long-context-rejected51`; the GPU also briefly throttled. The second one-time fan-setting attempt was also rejected: the kernel
notifier changed PWM during the run. The final fresh sweep uses a common
55°C start limit and a PWM255 feedback setpoint on the little cores for all
placements. The kernel notifier and thermal protection stay active; the
original PWM setting is restored afterward. The sweep will measure
CPU-only, Mali OpenCL, NPU and CPU+NPU/GPU+NPU/CPU+GPU+NPU FFN decode splits
at 1024/2048/4096 input tokens. Mixed rows retain NPU prefill. All use the same
weights, runtime context, clocks, four host threads, 32 outputs, two complete
warmups and two measurements. A checked native NPU request supplies each
long prompt's logits/teacher IDs; this is not an independent HF full-model
oracle. Every actual prediction is still checked. Quality and clock failures
will remain diagnostic rows at the existing gates.

The [immutable prompts](long-context/prompts/provenance.json) repeat the
existing exact 128-token benchmark prompt and take nested prefixes. They
measure synthetic throughput, not natural long-document task quality.

The exported `build.py` rebuilds the isolated runner directly from this
repository source copy. Harness scripts are snapshots of the comparison
workspace `/home/orangepi/qwen3-bench/matched/roofline/yalm`; run hardware jobs
there, serially. Model loading, cooling and tokenization remain outside
request timing; movement, synchronization and waits stay inside it.

INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.

AUTH: user said "wip commit per milestone".
