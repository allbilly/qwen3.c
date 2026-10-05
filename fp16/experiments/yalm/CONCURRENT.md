# Concurrent CPU / Mali / NPU FFN experiment

An isolated runner now executes disjoint gate/up output channels concurrently.
The NPU processes a balanced prefix on its three cores. Mali receives an
asynchronous upload, GEMM/GEMV and readback; one persistent CPU worker runs a
separate output slice. The main thread joins both branches and assembles the
full gate/up tensors. SwiGLU and merging are CPU operations; down projection
stays on the native NPU. All NPU allocation, sync and submit calls remain
serial on the main thread. The native register/task generator is unchanged.

Each auxiliary device receives 96 of the 3,072 hidden channels (3.125%).
CPU+NPU and GPU+NPU use a 2,976-channel NPU prefix; the three-device split
uses 2,880. Every configuration has three equally sized native tasks, with
the original submission flags, masks, domain and timeout.
`FFN_MODE=1/2/3` enables prefill/decode/both respectively.

## Precision and quality

The model is the existing pinned Qwen3-0.6B FP16 container, SHA256
`c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879`.
Projection inputs are rounded to FP16 on every branch; outputs and
accumulation are FP32. CPU weight slices are expanded once from those FP16
values into resident FP32 arrays; this is a lossless cast, with no additional
quantization. Reduction order can differ across devices.

- Nine primitive cases check all 1,640,448 gate/up outputs against double
  accumulation of identical FP16 operands. Largest relative RMSE is
  `1.32879464819e-7`, below the unchanged `1e-5` limit.
- Thirty full-model cases cover 1/24/73/128/256-token prompts, three device
  combinations and both-phase versus decode-only splitting. Each checks one
  complete warmup and one measured request, including all 16 predictions.
- Both-phase splitting fails the unchanged `0.001` first-logit gate for
  24/73-token prompts on all three combinations: six retained failures,
  relative RMSE `0.0010973–0.0015912`. All tested predictions still match.
  These prefill candidates remain diagnostics and are not selected.
- Decode-only splitting passes all fifteen cases. First-token logits are
  bit-identical because the selected prefill path is used. All decode
  predictions match. Finite prompt coverage does not prove arbitrary inputs.

The [independent quality audit](concurrent-ffn/concurrent-quality-audit.json)
checks source/binary/model hashes, inputs, full vocabulary logits and actual
CPU/GPU/NPU execution counters in every warmup and measured phase. The
[primitive records](evidence/concurrent-primitive/summary.json) and all
quality logs/configurations are frozen under `evidence/concurrent-*`.
A Python environment-key error prevented the first attempted quality job
from starting; that setup attempt is preserved separately and supplies no
measurements.

## Native packing and additional checks

The separate [native-packing source](concurrent-native/concurrent_ffn.c)
processes 64-row blocks and reads NPU gate/up outputs in their native layout.
It applies the selected stream's vector SwiGLU directly into the existing
native down-input buffers, joining CPU/GPU slices before packing. Down
partial sums retain their original addition order. It avoids full gate/up
unpacking and assembly. The original split remains frozen.

The candidate passes all 1,640,448 linear primitive outputs again, plus
270,336 complete FFN outputs compared with the selected native stream;
maximum full-FFN relative RMSE is `4.63321290521e-6`, below `1e-4`.
All fifteen full-vocabulary first-token dumps are bit-identical between
the original split and native packing. The combined [sixty-case quality
audit](concurrent-native/concurrent-quality60-audit.json) retains twelve
short-prefill failures, six per layout. All thirty decode-only cases pass
with bit-identical selected first-token logits and matching predictions.

## Branch profiles before request comparisons

Eight fixed-clock, 256-token branch-profile jobs are independently audited:
two full warmups, one measurement, 32 outputs, <=51C request starts, fixed
and restored CPU/GPU/NPU/DDR clocks. Every tested prediction matches. These
event-enabled timings diagnose costs; final throughput comes from the
unprofiled request sweeps, which are running at this milestone.

| Layout | Split | Prefill ms | Prefill tokens/s | Decode tokens/s |
|---|---|---:|---:|---:|
| Original | Selected NPU | 540.398 | 473.72 | 20.235 |
| Original | CPU+NPU | 671.060 | 381.49 | 20.004 |
| Original | GPU+NPU | 667.976 | 383.25 | 19.343 |
| Original | CPU+GPU+NPU | 683.260 | 374.67 | 19.523 |
| Native packing | Selected NPU | 537.599 | 476.19 | 19.787 |
| Native packing | CPU+NPU | 594.597 | 430.54 | 19.664 |
| Native packing | GPU+NPU | 1044.231 | 245.16 | 19.197 |
| Native packing | CPU+GPU+NPU | 1044.113 | 245.18 | 19.427 |

CPU+NPU gate/up assembly falls from 24.897ms to 0.014ms. Native packing
still exposes 40.127ms waiting for the single CPU worker, with 22.972ms
SwiGLU/native activation packing and 49.778ms down work. It improves this
profile's CPU+NPU prefill while remaining slower than its NPU reference.

GPU kernel time for the prefill slice rises from 96.639ms with full-prompt
jobs to 534.538ms with 64-row jobs. Their gate/up join wait rises from
20.981ms to 491.723ms. Smaller GPU jobs lose throughput and multiply queue
round trips; saved assembly costs do not offset that regression. This is
why each device layout needs a complete request measurement.

The [branch audit](concurrent-native/concurrent-branch-profiles-audit.json)
checks all execution counts, original quality limits and clock samples.
Numerical-check timings are not performance claims. The split records
host spans for rounding, dispatch, the complete
NPU branch, joining and assembly; CPU worker spans and their overlap with
the NPU branch; and optional OpenCL kernel/copy event times. An OpenCL
completion-status transition across the NPU branch independently records
GPU work completing while that branch is active. Branch timings overlap and
must not be added to the exclusive host decomposition.

These scripts are snapshots of the external comparison workspace's tools;
run its harnesses in `/home/orangepi/qwen3-bench/matched/roofline/yalm`.
The exported `build.py` builds directly from the repository source copy.

The runner keeps the original full NPU FFN plans for the phase that uses the
selected path. It therefore adds 330–341 MB of native FP16 weight payload,
plus 22 MB of CPU FP32 slices and/or 11 MB of GPU FP16 slices, outside request
timing. This is a correctness-first placement experiment with resident
weights, not a minimum-memory implementation or fully resident GPU model.

## Reproduce

```bash
python3 fp16/experiments/yalm/concurrent-ffn/build.py --output /tmp/runq-concurrent
FFN_CPU_CHANNELS=96 FFN_GPU_CHANNELS=96 FFN_MODE=2 \
OMP_NUM_THREADS=4 GOMP_SPINCOUNT=1000 \
NPU_CORES=3 NPU_FUSED=1 NPU_DOMAIN_ID=1 NPU_ATTENTION=1 \
NPU_SPLIT_DOWN=1 NPU_STREAM_FFN=1 NPU_CLS_TILE=8192 \
LD_LIBRARY_PATH=/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs \
taskset -c 4-7 /tmp/runq-concurrent \
  /home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16 \
  /home/orangepi/qwen3-bench/matched/roofline/final-quality/prompt128.tokens 32 2
```

The build uses the frozen vendor headers recorded in its original command;
`--npu-include` can specify another header checkout, in which case an exact
binary reproduction is not assumed. The sample inference command does not
fix clocks. Performance uses the external comparison harness instead.

INTENT: the router waits for each device projection; the user requests mixed CPU/GPU/NPU inference and transfer-inclusive phase and request measurements; fp16/README.md requires the shared FP16 checkpoint, numerical checks and serial NPU submissions.

INTENT: the first split assembles and repacks full FFN tensors; the user requests profiling followed by concrete prefill improvements; fp16/README.md describes direct activation packing between FFN projections with the verified FP16 operands.

AUTH: user said "wip commit per milestone".
