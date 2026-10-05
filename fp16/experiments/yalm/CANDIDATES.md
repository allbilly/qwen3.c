# Candidate numerical coverage — WIP checkpoint

All five candidates completed full-model checks at 1, 24, 73, 128 and 256
input tokens using the shared Qwen3-0.6B FP16 checkpoint. Each case has one
complete warmup and one measured request, with all 16 predictions compared
against the reference under identical teacher-forced inputs. Every prediction
matches. The unchanged first-token logit gate is relative RMSE < 0.001 over
all 151,936 vocabulary entries, with finite-value and shape checks.

| Candidate | 1 token | 24 tokens | 73 tokens | 128 tokens | 256 tokens |
|---|---:|---:|---:|---:|---:|
| Cooperative GPU decode softmax | 0 | 0 | 0 | 0 | 0 |
| Fused GPU decode softmax/P×V | 0 | 0 | 0 | 0 | 0 |
| Shared NPU prefill KV packing | 0 | 0 | 0 | 0 | 0 |
| GPU prefill tile 16 | 0.000128526 | **0.001487014 FAIL** | **0.001926848 FAIL** | 0.000610766 | 0.000577791 |
| CPU prefill BLAS attention | 0.000136200 | **0.001312401 FAIL** | **0.001496000 FAIL** | 0.000822982 | 0.000558366 |

Zeros mean bit-identical **first prefill logits**. They do not claim that every
decode logit is identical. GPU decode attention also passes the separate
double-accumulation primitive checks; the full-model checks cover all predicted
IDs on these finite cases. GPU tile 16 reproduces the original GPU prefill
logits, including its short-prompt failures. CPU BLAS attention fails the
quality gate; no qualified performance gain is claimed for it.

## Recorded evidence

Each directory retains source/binary provenance, complete raw logs, exact
inputs, teacher IDs, environment and numerical summaries:

- [GPU cooperative softmax](evidence/candidate-parallel-softmax-quality/quality-summary.json)
- [GPU fused attention](evidence/candidate-fused-attention-quality/quality-summary.json)
- [NPU shared KV packing](evidence/candidate-kv-pack-reuse-quality/quality-summary.json)
- [GPU tile 16, including failures](evidence/candidate-gpu-tile16-quality/quality-summary.json)
- [CPU BLAS attention, including failures](evidence/candidate-cpu-blas-attention-quality/quality-summary.json)

These additional checks did **not** fix frequencies. Their timing fields are
numerical-check logs, not qualified performance measurements. Full vocabulary
logit binaries remain in the external comparison workspace; their inclusion in
the next independent audit is pending. The new `benchmark/audit_quality.py`
recomputes the gates and checks device counters, source hashes and inputs; it
has been written but has not yet been run while the hardware sweep is active.

## Pending complete-request experiments

The independent request-timer build is running all ten CPU/GPU/NPU placements
at both 128 and 256 input tokens, two full warmups plus two measurements per
row, 32 outputs, common 50°C request-start target and fixed/sampled/restored
clocks. The partial sweep is not a completed or audited result. Several
cooldowns have taken minutes; they are outside measured spans. No device
process has been killed.

The next timer builds add a 180-second bound to the idle cooldown, followed
by normal CPU/GPU/NPU buffer cleanup on failure. The bounded builds have been
prepared but are **not yet compiled or tested**. The proposed counterbalanced
iteration protocol uses the same 51°C target for both engines within each
comparison; it has not run. Its results must be audited separately from the
50°C placement sweep, without mixing measurements between protocols.

The exported tools prepare frozen timer builds, run serial ABBA/BAAB
comparisons, aggregate all four individual measurements per engine/prompt,
and plot transfer-inclusive prefill, decode and independently measured request
time. Those iteration tools and their figures are pending execution. The
benchmark scripts are workspace source snapshots: their original location is
`/home/orangepi/qwen3-bench/matched/roofline/yalm`, not a claim of a portable
standalone package in this directory.

The selected production `runq-fp16` remains the original reference; no candidate
has been promoted. Existing independent patch reproduction and phase-cost
audits are recorded in [PHASES.md](PHASES.md) and [COSTS.md](COSTS.md).

INTENT: the selected runner has NPU projections and a tested GPU-attention route but no full CPU/GPU projection comparison; the user requests device combinations and stepwise profiling/optimization; fp16/README.md specifies the shared FP16 checkpoint, native NPU projection reference and numerical verification.

AUTH: user said "wip commit".
