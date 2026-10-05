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

## Profiling and request measurements

Fixed-clock branch profiling and unprofiled complete-request measurements
are pending this milestone. Numerical-check timings are not performance
claims. The split records host spans for rounding, dispatch, the complete
NPU branch, joining and assembly; CPU worker spans and their overlap with
the NPU branch; and optional OpenCL kernel/copy event times. An OpenCL
completion-status transition across the NPU branch independently records
GPU work completing while that branch is active. Branch timings overlap and
must not be added to the exclusive host decomposition.

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

AUTH: user said "wip commit per milestone".
