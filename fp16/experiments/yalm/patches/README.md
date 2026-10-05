# Independent experiment patches — WIP

Each patch applies to the committed phase-routing source in the parent directory.
Generated OpenCL C-string headers must be rebuilt from the `.cl` files. Use an
isolated copy for a candidate and retain the original for matched comparisons.
Do not apply several alternative attention patches together.

| Patch | Change | Observed verification so far |
|---|---|---|
| `parallel-softmax.patch` | One cooperative 64-lane group per decode head | Compiled; 8 primitive cases / 993,280 outputs pass unchanged limits; all five full-model prompt cases pass |
| `fused-attention.patch` | Cooperative softmax and P×V in one kernel; contiguous feature loads | Compiled; 8 primitive cases / 993,280 outputs pass unchanged limits; all five full-model prompt cases pass |
| `gpu-tile16.patch` | 16 regular GEMM rows reuse weights in one FP32 accumulator set; original split-down order retained | Compiled; 6 projection cases pass `1e-5`; full-model 24/73-token logits fail, all tested IDs match |
| `kv-pack-reuse.patch` | Pack 8 shared KV heads once and copy into the original per-core maps | Compiled; all five full-model prompt cases pass with bit-identical first logits; performance pending |
| `attention-row128.patch` | Batch 128 attention rows for K <= 512 within four data banks; retain dense row limits | Compiled; original and candidate each check 987,136 primitive outputs at `1e-4`; all five full-model cases pass with bit-identical first logits; performance pending |
| `complete-profile.patch` | All 14 root stages plus CPU/GPU/NPU packing, math, transfer and wait details | Complete baseline profiles audited; candidate profiles pending |
| `cpu-blas-attention.patch` | CPU prefill attention SGEMM with identical FP16-rounded operands | Compiled; 4 primitive cases / 985,088 outputs pass `1e-4`; full-model 24/73-token logit checks fail; 128/256 pass, all tested predictions match |
| `request-timer.patch` | Independent prompt-to-final-output monotonic wall span | Compiled externally; full-request measurements pending |
| `bounded-request-timer.patch` | Independent request timer plus a 180-second idle cooldown bound and normal cleanup | Compiled; initialized CPU/GPU/NPU timeout check exits normally with status 1 before inference; successful complete-request comparison pending |

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

`bounded-request-timer` includes `request-timer`; apply either patch independently
to the baseline. Its impossible-target timeout check took 180.127 seconds,
without a signal or changed clock settings. The original 50°C complete-request
sweep stalled and was rejected in full; a fresh common-51°C sweep is pending.

`manifest.json` records the external compiled binaries and patch hashes. The
selected production runner remains the exact original comparison reference.
All nine independent patches reproduce every compiled source hash when applied
in isolation and their generated OpenCL headers are rebuilt.

`CPU_ATTN_BLAS=1` enables the CPU attention candidate when a CPU projection or classifier route initializes OpenBLAS. These patches preserve the original scalar fallback. Extra short-prompt baseline CPU/GPU prefill checks fail the unchanged logit gate; neither a primitive pass nor a finite long-prompt timing establishes general correctness.
