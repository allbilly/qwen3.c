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

The three fresh NPU reference rows have completed with all sampled clocks
held and restored. Other placements and the independent full sweep audit
are pending. Preliminary NPU medians:

| Input tokens | Prefill ms | Prefill tokens/s | Decode tokens/s | Request ms |
|---|---:|---:|---:|---:|
| 1024 | 3780.94 | 270.83 | 14.478 | 5922.19 |
| 2048 | 10867.50 | 188.45 | 9.868 | 14009.13 |
| 4096 | 37473.03 | 109.31 | 5.490 | 43120.18 |

 The first 51°C attempt
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

The subsequent CPU1K row at2.256GHz thermally dropped to2.208GHz (NPU800MHz and Mali300MHz also observed), so it is retained as a failed-clock diagnostic. The GPU1K row passed clocks but stays in that separate attempt. All final18 rows are being rerun at common CPU1.800GHz and NPU/GPU/DDR1.000/1.000/2.112GHz, with all other conditions and gates retained. Original clock/fan settings were restored after each rejected attempt.

## Current bare-board1.8GHz milestone

All six1K placements and NPU2K/4K have completed. Their [harness summary](long-context/long-context-cpu1800-partial-summary.json) records prefillms/tokens/s, decode tokens/s and independent request spans; the independent full audit is pending until hardware stops. CPU1K fails the strict all-clock check because idle accelerator clocks drop near86C, while CPU clocks remain1.8GHz. Other1K rows pass the harness gates. The three mixed FFN decode rows have overlapping ranges with NPU-only; no reliable speedup is established. The user reports no heatsink; actual device-tree model is OrangePi5/RK3588S. Recommended a fitted heatsink+5Vfan, with [52PiEP-0167](https://wiki.52pi.com/index.php?title=EP-0167) as a documented compatible example. Physical fan presence/cooling effectiveness is not inferred from software PWM commands. The2K/4K placements are still running, and a preference question about continuing versus installing cooling is pending.
