# Single-runq refactor

Work is on `refactor/single-runq` directly in the `qwen3.c` checkout.
There are no linked worktrees. The core refactor starts at `02caf9d`, before
the Vortex integration. First complete the shared Qwen CPU/RKNPU/ANE core;
then adapt Vortex to the resulting hardware API.

The tested Vortex integration remains saved at `79f36cc` on `main`, and the
earlier setup is preserved on `baseline/refactor-setup-20261026`. Its source
and results remain available in Git. No core-refactor production source has
moved and no Qwen transformer implementation has been removed.

Captured model files, full logit vectors and binaries are under
`build/refactor-baseline/`. Historical receipts retain their original paths;
`build/refactor-baseline/relocation/manifest.json` maps their previous
checkout location to this one and records checksum verification.

## Phase gates

| Phase | Status | Required next evidence |
|---|---|---|
| 1. Baselines | Partial: local CPU captured; original Q8/FP16 containers reproduced | Fresh RK3588 Q8/FP16 and Asahi M1 ANE runs |
| 2. Hardware API | Not started | Phase 1 complete; Q8 CPU and hardware regressions |
| 3. File moves | Not started | Unchanged backend behavior/builds after moves |
| 4. Shared entry point and loaders | Not started | Q8 v1 and FP16 v2 format/CLI regressions |
| 5. Shared prefill/decode | Not started | Backend logits/tokens and matched performance checks |
| 6. Build simplification and cleanup | Not started | All format/backend acceptance gates |

The user's migration rule requires each phase to compile and pass its own
regressions before proceeding. Historical reports are preserved evidence;
they cannot establish current before/after hardware performance.
RK3588 and Asahi M1 SSH targets are needed to complete the live baseline.
macOS Core ML measurements do not substitute for the native Asahi ANE path.

After removing the linked worktree and selecting the pre-Vortex core,
the complete CPU baseline was captured again. All 9,723,904 logit values
and all generated token IDs are bitwise identical to the earlier capture.
The compatibility receipt is
`build/refactor-baseline/relocation/core-compatibility.json`.

## Baseline captured on October 26, 2026

The checkpoint containers reproduce the existing hardware baselines exactly:

- Q8 version 1, 633,495,808 bytes:
  `7a67926eb5be74ab87e4dc29db81f3077d98f686df6a65591eeb333ccfa9ca6e`.
- FP16 version 2, 1,192,231,168 bytes:
  `c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879`.
- Source revision: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.

CPU measurements use GCC 16.2.1, `-O3 -fopenmp -march=native`, four threads
on CPUs 4–7, one complete warmup and three measured trials per prompt.
The 16 output tokens consist of the prefill's first prediction followed by
15 decode calls. All warmup/measured IDs were stable. Every saved vocabulary
logit was finite and its argmax matched the generated token. All CPU
projection counts matched the expected layer/step coverage.

| Prompt tokens | Median prefill ms | Median decode tokens/s | Peak process RSS MiB |
|---|---:|---:|---:|
| 5 | 183.87 | 21.48 | 1205.5 |
| 16 | 561.12 | 21.22 | 1207.8 |
| 32 | 1128.40 | 21.13 | 1211.3 |
| 65 | 2309.86 | 20.77 | 1218.5 |

All 9,723,904 warmup logits are retained under
`build/refactor-baseline/phase1/cpu-q8/`, with per-case commands, source/model/
binary hashes, exit receipts, timing rows, generated IDs and full vectors.
The compact tracked inventory is `tests/baselines/refactor/phase1.json`.
RSS measures this process, rather than total device memory.

The original legacy CPU CLI completed successfully and its stdout/stderr are
retained with the baseline. The original default `make` still fails on this
x86 host because it selects RKNPU and exposes `__fp16`; the CPU-only
`make cpu` target passes. That portability issue remains a required build
acceptance fix after the baseline gate.

The archived ANE report has the same Q8 checkpoint hash. The FP16 container
matches `fp16/fp16-manifest.json` byte for byte. The selected FP16 performance
and accuracy reports remain under `fp16/`; experimental routing evidence
under `fp16/experiments/yalm/` is unchanged. A current RKNPU Q8 baseline still
needs to be captured; withdrawn Q8-versus-W8A8 comparisons are not acceptance
baselines. The native Asahi ANE raw logits and live device memory/performance
must also be refreshed on its original hardware.

## Model and hardware boundary inventory

| Current location | Model logic to retain in runq.c | Hardware logic to isolate |
|---|---|---|
| `runq.c` | v1 loading, tokenizer, sampling, serial forward/prefill, KV/attention/RoPE/SwiGLU | Compile-time backend dispatch and statistics access |
| `ane/prefill.h` | Duplicate Qwen prefill/layer ordering and KV updates | 32-row batches, short-batch CPU policy, projection calls |
| `fp16/run.c` | v2 loading and a second production main/CLI | FP16 initialization and benchmark control |
| `fp16/prefill.h` | Duplicate batched prefill/decode, attention and residual logic | Fused projections, hardware attention, streamed FFN hooks |
| `npu_matmul.c` | CPU fallback belongs to the caller | Q8-to-FP16 preparation and RKNPU register submission |
| `fp16/fp16_backend.c` | No model-layer ordering should migrate here | Persistent plans, three-core tiling, fused operations and transfers |
| `ane/ane_matmul.c` | No model-layer ordering should migrate here | M1 ABI, packed resident plans, batch/decode scheduling |

Keep `fp16/vector_math.h` optimizations when sharing CPU model operations.
FP16 and Q8 have different weight/activation arithmetic. The v2 loader must
retain FP16 weights and populate format-aware CPU fallback views; casting its
weights into Q8 or relying on the old null Q8 pointers is invalid.

Current public `NpuMatmulContext` leaks RK3588 dimensions and `__fp16` arrays.
Its boolean result merges unsupported operations and failures. The replacement
must distinguish success, unsupported and hardware error; errors must not
silently enter CPU fallback. The shared model needs to preserve single
quantization of reused Q/K/V and gate/up inputs.

Inventory these compile-time controls during migration:
`QWEN3_DISABLE_NPU`, `QWEN3_USE_ANE`, `QWEN3_USE_VORTEX`,
`MATCHED_CPU_REFERENCE`, and `__ARM_FEATURE_DOTPROD`.
Retain runtime ANE prefill/decode policy and RKNPU fusion, core, attention,
streamed-FFN and classifier settings. Preserve the FP16 benchmark's EOS
selection policy separately from normal inference semantics.

`macos/` is a separate Core ML experiment. The `tinystories/` Llama/Vortex
frontend belongs to the deferred `79f36cc` integration, outside the core
refactor branch for now. Preserve its frozen results when adapting Vortex
later. Test/experiment entry points are not additional production Qwen mains.
Do not delete archived routing source snapshots or rewrite their recorded
hashes during folder reorganization.

## Reproduce the local baseline

From the refactor worktree:

```bash
uv venv build/refactor-baseline/.venv
uv pip install --python build/refactor-baseline/.venv/bin/python numpy jinja2
build/refactor-baseline/.venv/bin/python -m tools.refactor_models \
  --snapshot build/refactor-baseline/phase1/hf-snapshot \
  --out build/refactor-baseline/phase1
make cpu
gcc -O3 -fopenmp -march=native -D_FILE_OFFSET_BITS=64 -DQWEN3_DISABLE_NPU -I. \
  tests/refactor_bench.c npu_matmul.c -lm -o build/refactor-baseline/phase1/cpu-bench
python3 tools/benchmark_queue.py -- python3 tools/refactor_baseline.py \
  --binary build/refactor-baseline/phase1/cpu-bench \
  --checkpoint build/refactor-baseline/phase1/Qwen3-0.6B.bin \
  --out build/refactor-baseline/phase1/cpu-q8-NEW \
  --backend cpu --trials 3 --tokens 16 --cpus 4-7 --threads 4
```

The benchmark utility calls the existing `runq.c` implementation; it does
not copy the transformer forward pass. The shared queue now unwinds SIGTERM,
stops its owned process group before releasing locks and keeps timeout
separate from a child handling termination successfully.
The queue/collector/export tests passed 13 cases on Linux; three native
macOS process-argument cases remain skipped. The FP16 exporter has a
separate precision-preservation and no-overwrite regression.

Compare future candidates to the preserved executable under matching model,
host, CPU affinity, thread count, warmups, inputs, context and backend policy.
Investigate changes exceeding 5%; preserve the baseline and existing
format-specific numerical gates rather than editing them to make a run pass.
