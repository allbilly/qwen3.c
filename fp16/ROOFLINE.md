# RK3588 roofline and register optimization

The optimized FP16 engine decodes at **20.85 tokens/s**, equivalent to
**24.85 GB/s of useful projection weights**, after the 128-token prompt.
That is **80.3%** of the measured three-core NPU stream rate, so whole-model
decode has not reached 90%. Isolating NPU submission intervals gives an
estimated **93.0%**; CPU operations and synchronization also consume time.
These percentages describe useful weight throughput. Actual DRAM byte counters
are unavailable on this vendor kernel.

Two retained task-layout changes make the streamed FFN down projection about
**1.90 times faster** and the output classifier **4.6% faster** in checked
primitive tests. Paired full-model runs improve prefill speed by **6.3–7.9%**
over the preserved custom baseline; the decode gain is **0.6–1.3%** and run
ranges overlap. A fresh comparison with stock RKLLM has faster custom medians
in both phases on both tested prompts.

Date: 2026-10-05. Board: local RK3588, 16 GB RAM, vendor kernel 6.1.99,
RKNPU driver 0.9.8. Raw evidence is in
`/home/orangepi/qwen3-bench/matched/roofline`. Calculations and hashes are in
[roofline-summary.json](roofline-summary.json).

Rendered figures: [roofline PNG](roofline.png), [JPG](roofline.jpg),
[SVG](roofline.svg), and [complete phase profile](PROFILE.md).

![Empirical useful-weight roofline](roofline.png)

## Same model and precision

Both engines use `Qwen/Qwen3-0.6B`, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`: 28 layers, dimension 1024,
FFN dimension 3072, 16 query heads, 8 KV heads, head dimension 128,
vocabulary 151936, tied embeddings, context 512.

The original BF16 checkpoint was rounded once to a shared FP16 checkpoint.
All **311 effective tensors** are audited. Projection and embedding tensors
match exactly; CPU norm values are exact FP32 expansions of the shared FP16
values. There is no integer quantization. Arithmetic and reduction order
between engines can still differ.

Stock uses toolkit 1.2.3 with `do_quantization=False`, default optimization
level 1, and three cores; the unmodified official runtime is 1.3.1. Runtime
logs confirm `model_dtype: FP16` and three NPU cores. The converter's `w8a8`
argument is unused with quantization disabled.

| Artifact | SHA256 |
| --- | --- |
| Shared `fp16-hf/model.safetensors` | `31127445785ad9f9db25f34f6b7e3a662b5a5034d1ace033d93e57a24c58a895` |
| Custom `Qwen3-0.6B.fp16` | `c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879` |
| Stock three-core FP16 RKLLM | `a777ccda03c982653c754405394d6daed354d773404e07f4c607a1b1e3e80e0a` |
| Selected `runq-fp16` | `100716de3aca7c4efc7a5b0a5a87178c41adc3bc4049a5f1c009a83fe051ff58` |

## Measured bandwidth and roofline

CPU probes stream three 256 MiB arrays, much larger than cache. NPU probes
cycle independent persistent weights: 128 MiB for one core and 192 MiB for
three cores. Both use two warmup sweeps and seven measurements. Allocation,
initial packing and cold initialization are excluded. GEMV checks every
output; GEMM checks all rows at sampled columns, against double accumulation
of the same FP16 operands. All checks passed exactly.

CPU 4–7 are fixed at 2.256 GHz, NPU at 1 GHz, DDR at 2.112 GHz. NPU jobs,
initialization, submissions and DMA synchronization are serial. Frequency
samples are recorded every 0.25 seconds and original settings are restored.

| Probe | Useful payload GB/s |
| --- | ---: |
| CPU read, one big core | 17.645 |
| CPU read, four big cores | 26.698 |
| CPU copy, four big cores, read + write payload | 23.712 |
| CPU triad, four big cores, two reads + one write payload | 22.999 |
| NPU GEMV, one core, blocking submission | 10.901 |
| NPU GEMV, three cores, blocking submission | **30.941** |
| NPU GEMV, three cores, packing/sync/readback included | 30.611 |
| NPU GEMV, three cores, K=1024, tile N=2048 | 29.066 |
| NPU GEMV, three cores, K=2048, tile N=352 | 24.519 |
| NPU GEMV, three cores, K=3072, tile N=352 | 27.744 |

Copy/triad byte counts are algorithmic payloads; write allocation can cause
additional DRAM traffic. NPU numbers count useful weights, excluding input,
output and command traffic. They are not bus-counter measurements.

The largest checked native GEMM here uses M=64, K=1024 and N=24576
(three N=8192 tiles) and reaches **1739.44 GFLOP/s** in submission intervals.
Including packing, synchronization and unpacking gives **895.59 GFLOP/s**.
1739.44 is the best observed compute rate here, not a proven hardware maximum.

The empirical weight-based roofline is:

```text
Projection performance <= min(30.94095 * I, 1739.44236) GFLOP/s
I = useful projection FLOPs / useful projection weight bytes
```

FP16 decode contributes two FLOPs and two weight bytes per parameter,
so I is approximately **1 FLOP/byte**. M=64 matmul reuses a weight for 64
rows, giving weight-based I approximately 64; counting input/output traffic
reduces the intensity. The empirical ridge is about 56.22 FLOPs per weight
byte. Full prefill also includes attention, non-matrix CPU operations, layout
conversion, and a single-row vocabulary projection.

The manifest contains **197 dense projections**, totaling
**1,191,968,768 weight bytes per decode token**. The tied matrix is counted
once for the full output head; input embedding reads one row. Using the
entire container or GGUF file size would overcount decode traffic.

```text
Weight-only decode roofline = 30.94095e9 / 1,191,968,768 = 25.958 tokens/s
90% of that stream rate     = 23.362 tokens/s
Observed after 128 tokens   = 20.85 * 1,191,968,768 = 24.853 GB/s = 80.3%
Observed after 256 tokens   = 20.1485 * 1,191,968,768 = 24.016 GB/s = 77.6%
```

The [Rockchip brief datasheet](https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RK3588%20Brief%20Datasheet.pdf)
lists four 16-bit memory channels. If the reported 2.112 GHz is the DDR clock,
64 bits at two transfers per cycle give a nominal 33.792 GB/s. We use the
measured NPU stream as the denominator. The
[NERSC roofline guide](https://docs.nersc.gov/tools/performance/roofline/)
describes the bandwidth/arithmetic-intensity and compute ceilings.

No DDR/DFI byte PMU is exposed by the installed kernel; devfreq `load` is
not treated as a byte counter. KV reads, write allocation, arbitration and
cache reuse prevent inferring actual DRAM utilization from these estimates.
The custom engine's average unique FP32 KV footprint is about 33.0 MB/token
after the 128-token prompt and 62.4 MB after 256; shared KV heads and CPU
caches affect actual transactions.

## Where decode time goes

A separate diagnostic profile of the selected binary, after 128 input tokens,
records per decode token:

| Operation | Time ms |
| --- | ---: |
| NPU blocking submissions | 41.437 |
| DMA synchronization | 0.948 |
| Input packing | 0.354 |
| Output unpacking | 0.410 |
| CPU decode attention | 2.385 |

Weights divided by submission time give **28.766 GB/s**, or **93.0%** of the
stream probe. These intervals include driver/interrupt overhead; they are
not pure hardware execution time. Other CPU operations also remain. The
profile's throughput is diagnostic; the fresh, unprofiled comparisons below
determine final performance claims. The profiler omits streamed FFN prefill
work, so it cannot supply a complete prefill breakdown.

A subsequent [complete profile](PROFILE.md) instruments all paths in an
isolated build, including streamed FFN and attention. It separates packing
from input copies and records driver launch-to-completion timing. Root
operation coverage exceeds 99.96%. See its rendered cost chart and candidate
results; the earlier diagnostic rows above remain preserved measurements.

The 90% whole-decode target requires about **42.8 ms/token**, compared with
approximately 48.0 ms now. CPU attention and packing/sync/unpack are concrete
further profiling targets. The classifier already streams useful weights at
about 30.37 GB/s in its isolated end-to-end test.

## Retained register/task changes

1. **FFN down projection:** keep K=3072 split into three K=1024 partial sums,
   assign one full N=1024 output surface to each NPU core, and submit the
   three independent tasks in one bundle. Previously each K partition
   required its own three-core bundle. Pack all SwiGLU partitions in one
   CPU parallel region and sum FP32 outputs in the original partition order.
   The streamed FFN chunk uses two submissions instead of four.
2. **Classifier N tiles:** balance tile count across three cores. With the
   8192 limit, vocabulary 151936 uses 21 tiles: twenty of N=7264 and one
   of N=6656. Seven bundles keep all cores active. Previously 19 tiles
   left only one active core in the final bundle. Precision is unchanged.

| Checked primitive, ABBA, 14 measurements per variant | Baseline ms | Selected ms | Speed ratio |
| --- | ---: | ---: | ---: |
| Vocabulary projection, all 151936 outputs checked | 10.7164 | 10.2453 | 1.046 |
| M=64 down projection, all 65536 outputs checked | 0.8662 | 0.4555 | 1.902 |

Thirty bounded FEATURE_GRAINS/CBUF-bank variants were also tested across six
shapes. All checks passed, but no consistent win emerged, so none was
retained. DMA bursts were already at the existing maximum. Submission flags,
task counts, domain, precision and serial execution protocol are unchanged.
No model projection falls back to CPU matmul.

Paired immutable-baseline versus selected-binary runs, ten measurements per
variant at each prompt length:

| Input tokens | Baseline TTFT ms | Selected TTFT ms | Prefill speed gain | Baseline decode t/s | Selected decode t/s | Decode gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256.094 | 237.287 | 7.9% | 20.6655 | 20.9395 | 1.3% |
| 256 | 547.402 | 514.899 | 6.3% | 19.8220 | 19.9480 | 0.6% |

Small decode gains have overlapping run ranges. The down-only intermediate
improved prefill but slightly regressed decode; combined changes were
selected after the classifier measurement and full retest.

## Fresh fair comparison with stock RKLLM

| Input tokens | Custom TTFT ms | Stock TTFT ms | Custom decode t/s | Stock decode t/s | Measurements per engine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | **233.849** | 276.233 | **20.8500** | 19.1795 | 10 |
| 256 | **507.688** | 569.537 | **20.1485** | 18.5415 | 8 |

Prefill speed gains are **18.1% / 12.2%**, and decode gains are
**8.7% / 8.7%**, respectively. These are medians for this model, precision,
board, runtime and prompt set.

Both engines use identical input IDs, three NPU cores, CPU affinity 4–7,
four CPU threads, fixed clocks, context 512, IOMMU domain 1, greedy sampling,
EOS 151645 suppressed, empty KV history per request and 32 generated tokens.
TTFT is measured externally through the first token; decode covers the 31
first-to-last token steps. Each fresh process performs two full warmups.
Initialization is excluded. 128-token blocks use custom/stock/stock/custom;
256-token blocks use stock/custom/custom/stock. Every generated ID matches.

Stock uses its default OpenMP waiting policy and `embed_flash=0`; custom uses
`GOMP_SPINCOUNT=1000`. Applying the custom spin setting to stock slowed it in
earlier work and is excluded from the fair protocol. Every sampled clock
held, and saved settings were restored and read back.

## Correctness and provenance

- All five final custom logit arrays for prompts of 1, 24, 73, 128 and 256
  tokens are **bit-identical to the preserved custom baseline**.
- All five checks generate the same 16 tokens as stock, including warmups.
  Unmasked logits are finite; stock's suppressed EOS is recorded and excluded
  from numerical comparisons.
- Relative logit RMSE to stock ranges from `5.70534e-5` to `0.001529543`,
  unchanged by these optimizations.
- A fresh same-FP16 CPU reference with double accumulation has Hello relative
  RMSE `4.686397323e-5`, passing the **unchanged `1e-4` limit**. The older
  failed diagnostic remains preserved.
- `final-audit.json` independently recomputes raw medians and verifies tensor
  identity, source/binary hashes, IDs, warmups, resources, clock samples and
  restored sysfs settings.

The down-only intermediate's source snapshot raced with the next classifier
edit, while its measured binary stayed unchanged. The snapshot is preserved;
reversing only that later edit and rebuilding reproduced the measured binary's
exact SHA256. `source-provenance-repair.json` records the repair and
`down-bundle-source` holds the matching source/binary. The selected combined
binary and final measurement snapshots match.

## Supplied references and local benchmark

The [2024 article](https://clehaxze.tw/gemlog/2024/02-14-benchmarking-rk3588-npu-matrix-multiplcation-performance-ep2.gmi)
is headed EP3 despite its URL. It reports about 900 GFLOP/s for native FP16
M=128 K=1024 N=8192, lower normal-layout performance, and input/output reorder
costs. Our persistent native layouts and separate submission/end-to-end
timings address that distinction. Its core/clock conditions do not establish
a current three-core ceiling; our 1739 GFLOP/s probe uses M=64 and three
separate N=8192 tiles.

The [2023 API article](https://clehaxze.tw/gemlog/2023/09-02-benchmarking-rk3588-npu-matrix-multiplcation-performance-ep2.gmi)
explains normal/native layout and rebinding updated inputs. Its older SDK
fields are not substituted into the installed API without checking headers.
The [original article](https://clehaxze.tw/gemlog/2023/08-26-benchmarking-rk3588-npu-matrix-multiplcation-performance.gmi)
acknowledges FP16 NPU versus FP32 CPU/GPU arithmetic; those timings do not
establish a matched-precision accelerator comparison.

The supplied [gist](https://gist.github.com/marty1885/a939e3cda146e333195bf8f62fde7a95)
is a historical square-matmul benchmark, rather than a complete roofline
measurement. Its lambda-local static `first` is shared across matrix shapes,
timing includes result allocation/copying, CPU operands are FP32, and no
output correctness check is performed. It cannot determine decode bandwidth
efficiency.

`~/rk3588-matmul-bench` is an NPU-only sweep, revision
`267930d8c6468a9a576dc3922468a643bf725494`. It initializes FP16 through an
integer `uint16_t` distribution instead of numeric half conversion, uses
`memset(&ctx, 0, sizeof(info))`, copies SDK sizes without ensuring host
allocations include padding, and has no output checks or excluded warmups.
The full sweep has 6144 configurations and 184320 timed calls. We inspected
and archived it; it was not executed unchanged. Current bounded probes use
checked data, explicit core selection, persistent packing and fixed clocks.

## GPU comparison

The later [CPU / Mali GPU / NPU experiment](HYBRID.md) runs a complete request
with NPU dense projections and GPU decode attention. Its matched decode
throughput regresses 11.7–15.1%; GPU prefill fails the unchanged logit gate.
The selected CPU+NPU runner is retained. That report includes rendered
performance and OpenCL event costs; a model running entirely on GPU remains untested.

The ARM OpenCL driver works on Mali-LODX r0p0 and supports FP16. Direct
OpenCL probes through installed tinygrad 0.11.0 give:

| GPU at 1 GHz, DDR 2.112 GHz | Wall throughput | OpenCL event throughput |
| --- | ---: | ---: |
| Read 256 MiB with checksum output | 24.645 GB/s | 24.934 GB/s |
| FP16 GEMV, same K/N and four weight sets as NPU | 17.464 GB/s | 18.203 GB/s |

Matching NPU blocking-submission GEMV reaches 30.941 GB/s. GPU operands stay
FP16 and products/accumulation/output are FP32. Every GEMV output and sampled
GEMM columns in every row match exactly before and after timing. GPU clocks
held and were restored. Wall intervals compare to NPU submission intervals;
GPU event time excludes host overhead. These are prototype implementation
measurements, without a GPU hardware-limit claim.

The shared-array GPU GEMM prototype performs poorly, at 12.665 GFLOP/s;
the driver reports local memory backed by global memory. Generated kernels
reach 52.394 GFLOP/s while materializing FP32 operands. Neither establishes
optimized GPU prefill. Both remain as diagnostics. The newer `~/tinygrad`
checkout requires Python >=3.11; its initial attempt failed on Python 3.10.
Successful tests use the installed package with recorded hashes. No
dependency was installed.

**llama.cpp is a suitable full-model harness once it has a working Mali
backend.** The pinned checkout
`b92761a515ea31e852e7fbc1fad5f874b46f3718` has a CPU-only build. Its OpenCL
device filter accepts Adreno/Qualcomm or Intel and rejects Mali; disabling
Adreno kernels does not change that filter. The
[upstream OpenCL documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)
describes Adreno and certain Intel support. No Vulkan loader/ICD was found
on this board. No full-model GPU tokens/s is claimed.

An FP16 GGUF of the exact shared checkpoint is ready at
`roofline/gpu/Qwen3-0.6B-matched-f16.gguf`, SHA256
`6782193cfc9dc18b3ab1db4479893ef2d554650df7b33b460aeefe349e4ce035`.
All 311 effective tensors are verified: matrix tensors FP16, norms exact
FP32 expansions, no integer quantization. A wrapper selects the upstream
BPE tokenizer fallback because `tokenizer.model` is absent; upstream files
are unchanged.

For full GPU comparison, use identical 128/256 token IDs, context 512,
32-token greedy output/EOS rule, empty history, two full warmups, fixed clocks
and external TTFT/31-step decode timing. Verify actual GPU placement and
record CPU fallbacks and activation/KV precision. A standalone
`llama-bench tg32` at empty history does not match these decode workloads.
The protocol and raw evidence are in
`/home/orangepi/qwen3-bench/matched/roofline/gpu/COMPARISON.md`.

## Reproduction and evidence

```bash
cd /home/orangepi/qwen3.c
make fp16
cd /home/orangepi/qwen3-bench/matched
python3 roofline/analyze.py
python3 optimization/audit_final.py --quality roofline/final-quality \
  --experiments roofline/final-stock128 roofline/final-stock256 \
  --out roofline/RECHECK.json
python3 roofline/gpu/audit_micro.py
```

New hardware measurements must run serially with fresh output directories:
`roofline/run_micro.py --suite baseline|registers|projections --out NEW_DIR`
and `roofline/gpu/run_micro.py --implementation coalesced --out NEW_GPU_DIR`.
Recorded source/binary snapshots reproduce the actual measured versions.

Evidence: `measurements-baseline`, `register-sweep`, `projection-abba`,
`balanced-abba`, `final-stock128`, `final-stock256`, `final-quality`,
`baseline-profile128`, `final-profile128`, `gpu/micro-coalesced`.
Failed/intermediate measurements remain labeled in their own directories.
The earlier unexecuted plotting scratch file was removed. The follow-up now
renders and visually checks roofline PNG/JPG/SVG and complete phase costs,
with plotting source/data hashes retained. See [PROFILE.md](PROFILE.md).
No commits or pushes were made.

INTENT: code executes all projections on the NPU using shared FP16 weights; the task expects faster register-driven matmul, prefill and decode with a measured bandwidth roofline; fp16/README.md specifies native persistent tasks, no CPU projection fallback, matched checkpoint precision and numerical verification.
