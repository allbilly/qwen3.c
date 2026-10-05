# Independent experiment patches — WIP

Each patch applies to the committed phase-routing source in the parent directory.
Generated OpenCL C-string headers must be rebuilt from the `.cl` files. Use an
isolated copy for a candidate and retain the original for matched comparisons.
Do not apply several alternative attention patches together.

| Patch | Change | Observed verification so far |
|---|---|---|
| `parallel-softmax.patch` | One cooperative 64-lane group per decode head | Compiled; 8 primitive cases / 993,280 outputs pass unchanged limits |
| `fused-attention.patch` | Cooperative softmax and P×V in one kernel; contiguous feature loads | Compiled; 8 primitive cases / 993,280 outputs pass unchanged limits |
| `gpu-tile16.patch` | 16 regular GEMM rows reuse weights in one FP32 accumulator set; original split-down order retained | Compiled; 6 actual-weight projection cases / 383,360 outputs pass `1e-5` rRMSE |
| `kv-pack-reuse.patch` | Pack 8 shared KV heads once and copy into the original per-core maps | Compiled; full-model correctness and performance pending |
| `complete-profile.patch` | All 14 root stages plus CPU/GPU/NPU packing, math, transfer and wait details | Compiled externally; hardware profiles pending |
| `request-timer.patch` | Independent prompt-to-final-output monotonic wall span | Compiled externally; full-request measurements pending |

These are independent candidates, with no claimed performance gain yet. The
shared FP16 operands and NPU submission metadata are unchanged. Compiler resource
and hardware execution time do not expose actual DRAM traffic or separate pure
ALU compute from memory stalls.

From an isolated copy of the phase-routing directory:

```bash
patch -p1 < /home/orangepi/qwen3.c/fp16/experiments/yalm/patches/fused-attention.patch
python3 embed.py attention.cl source/fp16/gpu_source.h gpu_source
python3 embed.py linear.cl linear_source.h linear_source
```

Compile with the same source order and flags as `make fp16-routes`, using the
absolute NPU include path `/home/orangepi/npu/include`. For `complete-profile`,
add `source/fp16/phase_profile.c` to the sources and set `ROUTE_PROFILE=1`;
`GPU_PROFILE=1` and `LINEAR_PROFILE=1` enable OpenCL event timing. Root-stage
profiling changes timing overhead, so final performance is measured separately.

`request-timer` reports resident-model request wall time including transfer,
packing, synchronization, transition/counter-reset/interphase diagnostic work.
Model initialization, thermal waits before a request, tokenization/input-ID file
reads and post-request reporting are excluded and identified separately. Cache
layout preparation inside the first decode call remains in the decode span.

`manifest.json` records the external compiled binaries and patch hashes. The
selected production runner remains the exact original comparison reference.
