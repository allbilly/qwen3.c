# CPU / GPU / NPU phase-routing experiment — WIP

This is the exact source export of the isolated `11d2f9b7...` benchmark build.
Six CPU and six GPU projection cases check every output against double
accumulation of identical FP16 operands, including the vocabulary head.
All pass the unchanged `1e-5` relative RMSE limit. Check logs and source hashes
are included. The [20-row phase sweep](PHASES.md) now passes clock, placement
and output audits, including a rerun of the rejected thermal CPU256 case.

It supports all nine CPU/GPU/NPU projection-backend pairs for prefill and
decode, plus an explicit placement with CPU classifier, GPU prefill projections
and decode attention, and NPU decode projections. Device labels describe
projections/attention; **embedding, norms, RoPE, SwiGLU, residuals and sampling
remain CPU operations**. A fully GPU-resident transformer is not implemented.
There is no concurrent partition of the same projection across devices.

CPU keeps FP16 model storage and rounds projection inputs to FP16, with FP32
accumulation. Prefill uses temporary FP32 expansions and the installed ILP64
OpenBLAS; decode uses NEON FP16 conversion and FP32 accumulation. GPU keeps
persistent FP16 weights in a coalesced layout and uses FP32 arithmetic/output.
Prefill attention uses FP16 Q/K/V/probabilities; decode cache and attention are
FP32. Original NPU source, submission flags and task metadata are retained.

```bash
cd /home/orangepi/qwen3.c
make fp16-routes
LD_LIBRARY_PATH=/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 GOMP_SPINCOUNT=1000 WARMUP_RUNS=2 \
DENSE_PREFILL=cpu DENSE_DECODE=npu COOL_REQUEST_C=55 \
NPU_CORES=3 NPU_FUSED=1 NPU_DOMAIN_ID=1 NPU_ATTENTION=1 \
NPU_SPLIT_DOWN=1 NPU_STREAM_FFN=1 NPU_CLS_TILE=8192 \
taskset -c 4-7 ./runq-fp16-routes \
  /home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16 \
  /home/orangepi/qwen3-bench/matched/roofline/final-quality/prompt128.tokens 32 2
```

`DENSE_PREFILL` / `DENSE_DECODE`: `cpu`, `gpu`, or `npu`.
`GPU_ATTENTION`: unset, `prefill`, `decode`, or `both`.
`CPU_CLASSIFIER=1` assigns the vocabulary projection to CPU.
`YALM_BLAS` can select another compatible ILP64 OpenBLAS library; its dependencies
must be discoverable by the dynamic linker. `COOL_REQUEST_C=55` waits between
requests, outside timing spans, to avoid the observed thermal throttling.
`TEACHER_IDS` accepts exactly one reference ID per requested output token, so
timed decode receives identical inputs even if a route predicts differently.
Every prediction is still recorded and compared to the reference.

Only the supported shared FP16 Qwen3-0.6B/context-512 container is accepted.
Run hardware processes serially. This command does not fix clocks. The
comparison harness in `/home/orangepi/qwen3-bench/matched/roofline/yalm` fixes,
samples and restores CPU/GPU/NPU/DDR clocks and records exact inputs and hashes.

The [phase report](PHASES.md) includes TTFT milliseconds, effective prefill
tokens/s, decode tokens/s, full min/max ranges and PNG/JPG/SVG plots. Two thermal
failures and an unreachable 45°C setup target are preserved; accepted rows
retain the original strict clock and numerical gates. The final complete-request
comparison will use one common successful thermal target.

`make fp16-routes` was built locally and reproduced the exact external benchmark
binary hash `11d2f9b7...`. The [independent candidate patches](patches/README.md)
reproduce every compiled source hash in isolated application checks. GPU primitive
checks pass; full-model candidate tests, detailed cost profiles and independent
complete-request measurements are the next milestones. The [complete cost
profiles](COSTS.md) now cover all 14 stages and identify actual packing, device,
synchronization and host wait costs. Additional short-prompt CPU/GPU prefill
logit checks fail the unchanged gate; those routes remain WIP diagnostics. The selected `runq-fp16`
remains unchanged.

The `source/` copy preserves the code used in the experiment independently of
future selected-engine edits. This is a WIP checkpoint, not a claim that every
route or optimization is complete or faster.
