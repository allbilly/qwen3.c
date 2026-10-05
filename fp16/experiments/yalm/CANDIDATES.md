# Candidate numerical coverage — WIP checkpoint

All six candidates completed full-model checks at 1, 24, 73, 128 and 256
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
| NPU attention batch 128 | 0 | 0 | 0 | 0 | 0 |
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
- [NPU attention batch 128](evidence/candidate-attention-row128-quality/quality-summary.json)

These additional checks did **not** fix frequencies. Their timing fields are
numerical-check logs, not qualified performance measurements. Full vocabulary
logit binaries remain in the external comparison workspace. The independent
[42-case audit](candidate-checks/quality-audit42.json) recomputed every gate and
checked source/binary/model hashes, exact inputs and teacher IDs, actual device
counts, and all predictions in both the warmup and measured request. All IDs
match; 12 first-logit gates fail and remain marked as failures. The audit includes
the earlier short-prompt baseline CPU/GPU cases as well as these candidates.

## NPU attention register candidate

`attention-row128.patch` batches 128 attention rows when K <= 512, within the
existing four-data-bank guard. Dense projection row limits stay at 64/32/16;
NPU submission flags, core masks, domain and task metadata are unchanged.
For the 128- and 256-token cases, attention submissions per layer decrease
from 24 to 12 and 48 to 24 respectively. This does not establish a speed gain.

The original 64-row and candidate 128-row primitive checks each compare all
987,136 outputs over five prompt lengths against the same double-accumulation
reference. Both pass the unchanged `1e-4` limit; the maximum relative RMSE is
`6.99521059651e-7`. The complete-model checks have bit-identical first logits
and matching 16-token predictions. [Raw checks and source provenance](candidate-checks/npu-attention-row128-provenance.json)
are retained alongside the original checker and checker sources.
`build_attention_check.py` regenerates the model helper header from the recorded
entrypoint prefix; the generated header hash is retained in checker provenance.

## FP16 arithmetic observation

The [native attention probe](candidate-checks/npu-fp16-probe.jsonl) preserves
the tested positive FP16 subnormal values, including the smallest value,
subnormal Q/K operands and a subnormal probability. All seven printed outputs
equal the IEEE FP16 operand reference. This finite probe gives no evidence for
flushing those values to zero; it does not explain the full-model CPU/GPU logit
differences or establish exhaustive floating-point equivalence. Its source,
compiled binary and backend hashes are [recorded](candidate-checks/npu-fp16-probe-provenance.json).

## Pending complete-request experiments

The original independent request-timer sweep completed only the ten 128-token
placements. The common 50°C target stalled while GPU/NPU buffers were resident
at fixed clocks. Recovery restored the original governors during an idle wait;
the native process then completed normally. The Python controller was stopped
only after that native process exited. All original settings were independently
verified against live sysfs and immutable before-run snapshots. The entire
[partial sweep is rejected](evidence/rejected-request-cool50/rejected.json),
including earlier rows; none of its timings qualify as final speed evidence.

Four bounded timer builds now compile: baseline, cooperative GPU softmax,
fused GPU attention and shared NPU KV packing. A deliberately impossible
cooldown target exercised the baseline timeout after 180.127 seconds with
CPU/GPU/NPU buffers initialized. It exited normally with status 1, before
inference, without a signal; clock settings were unchanged. The
[timeout evidence](evidence/guarded-request-timeout/summary.json) and compiled
source/binary provenance are retained. A successful complete-request sweep of
the bounded builds remains pending.

The next complete placement sweep and counterbalanced candidate comparisons
will use a common 51°C target and the 180-second idle-wait bound. They have not
run. Their measurements must be audited without mixing the rejected 50°C
partial sweep into the new protocol.

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
