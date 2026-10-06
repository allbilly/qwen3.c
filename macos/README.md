# macOS Qwen3 CPU, ANE and GPU benchmarks

This harness converts a **qwen3.c version-1 Q8 checkpoint** into one Core ML
decoder graph and runs it with `CPU_AND_NE` and `CPU_AND_GPU` by default.
An optional `--modes cpu gpu ane` also measures the `CPU_ONLY` control.
It uses the same checkpoint, coefficients, prompt IDs, context capacity,
warmups and fixed decode inputs for all three configurations. The existing
Asahi C backend and its measurements remain separate.

Requires Apple silicon, macOS 15 or later, Xcode command-line tools, and Python 3.11 with the packages
in `requirements.txt`. From the repository root:

```sh
python3.11 -m venv macos/.venv
macos/.venv/bin/python -m pip install -r macos/requirements.txt
macos/.venv/bin/python -m macos.export macos/local-models/Qwen3-0.6B.bin \
  --output macos/local-models/qwen3-0.6b-coreml --context 512
macos/.venv/bin/python -m macos.benchmark \
  --model macos/local-models/qwen3-0.6b-coreml \
  --checkpoint macos/local-models/Qwen3-0.6B.bin \
  --output macos/local-results/run-001
macos/.venv/bin/python -m macos.audit macos/local-results/run-001 \
  --output macos/local-results/run-001/audit.json
```

Conversion and benchmarking each require a new output directory; the audit
requires a new output file. Reuse existing converted packages by starting at
the benchmark command. Conversion needs temporary disk
space for the separate prefill/decode graphs; the final multifunction packages
share their identical weights. Model weights, converted packages and raw
validation logits stay in ignored local directories. A checkpoint can be
created from a local Hugging Face snapshot with:

```sh
macos/.venv/bin/python ane/export_checkpoint.py /path/to/hf-snapshot \
  macos/local-models/Qwen3-0.6B.bin --context 512
```

All decoder layers run through Core ML, in four-layer blocks by default.
Embedding lookup, final RMS normalization, the vocabulary projection and
argmax run on CPU in every configuration. Q8 coefficients are decoded once
and rounded to FP16. The host vocabulary projection accumulates in FP32;
Core ML uses FP16 static projections with FP32 attention and pointwise math.
Projection coefficients are scaled per row by exact powers of two (1–32),
then restored in FP32 after the matrix product, preserving the original FP16
coefficients while avoiding small-coefficient loss. These operations and their
device handoffs are timed. This differs
from the original C CPU backend's Q8 activation quantization, so an OS or
hardware speedup must not be inferred by comparing the two implementations.

The primary configuration labels mean **permitted compute devices**. Each
worker saves the preferred device for every Core ML operation and fails if
the requested ANE/GPU has no preferred operations for prefill or decode.
These plans describe compiler placement, not runtime utilization or exclusive
execution on one accelerator. CPU operations and device handoffs are included.

Prefill processes the entire prompt and returns its first next-token choice.
The default request has 16 output tokens and **15 subsequent decode calls**.
Prefill throughput is prompt tokens divided by prefill seconds; decode
throughput is 15 divided by the total seconds for the subsequent calls.
Tables report the median of each trial's throughput in **tokens/s**.
Each decode call uses one real token in the same 32-row graph, with masked
padding. On this M1, Core ML preferred CPU for the one-row graph even when
GPU or ANE was permitted. Fixed-width decode exercises the accelerator and
its extra computation and cache traffic are included in every configuration.
These rates are not optimized single-token decode throughput.
Every timed decode receives the same reference-generated token, even if a
configuration predicts differently. Prefill uses fixed 32-token buckets;
the host final normalization and vocabulary projection also execute for each
bucket. Padding writes future cache positions, which remain causally masked
until real tokens overwrite them. Extra cache slots keep the last padded call
within the allocated capacity. Each request begins with an empty cache.

Timers include completed synchronous predictions, explicit KV cache handoff,
host embedding/head math, finite checks and greedy selection. They exclude
model loading, compilation, cache initialization, warmups, validation,
tokenization and detokenization. Initialization is recorded separately.
Each backend/trial uses a fresh process, with two complete warmups per prompt.
The default three rounds rotate the selected device order, keeping one configuration
resident at a time. All workers are serialized behind `~/ane.lock` and
`~/gpu.lock`, and the queue also checks for live benchmark processes.
On macOS the scanner reads native process arguments, preserving quotes,
spaces and empty arguments. It ignores `macos.benchmark` coordinators that
acquire these same locks and detects their `--worker-mode` processes, avoiding
a deadlock between a lock holder and another coordinator waiting for it.
Run `macos.benchmark` directly: it acquires these locks itself, so wrapping
it in `tools/benchmark_queue.py` would wait on locks already held by its parent.

An independent PyTorch FP32 arithmetic replay on the same FP16 coefficients
creates the reference logits and teacher tokens. Every vocabulary logit is
checked outside timing: finite values, per-prediction NRMSE ≤0.005, KL ≤0.01,
and zero top-1 mismatches. Failed checks are retained and their timings are
labeled unqualified. Every warmup and timed request must also reproduce the
reference token choices; a mismatch fails the prompt and worker qualification
while retaining its predictions and timings.

`results.json` contains the protocol, source/model hashes, job order,
initialization costs, raw timings and medians. Adjacent worker receipts retain
placement plans, validation logits/hashes, and host load, thermal and paging
snapshots. The graph and cache strategy are benchmark implementations; the
numbers do not represent optimized device peaks. Clocks are not fixed.

Workers atomically save their receipts after each completed prompt and remain
`RUNNING` until all prompts finish. A prediction error records `ERROR` and a
finish timestamp while retaining completed measurements and logit files.
Lock acquisition and process-scan failures also finalize the coordinator as
`ERROR` and release any locks it acquired.

The auditor hashes the embedded model manifest using the exporter's JSON
format, checks the physical input widths against its batch size, and
recomputes the numerical gates and timing summaries. New audit receipts
record the auditor's source hash; historical reports retain the source hashes
captured when their measurements were taken.

Run the portable numerical and queue checks from the repository root:

```sh
macos/.venv/bin/python -m unittest discover -s tests -p 'test_macos*.py' -v
macos/.venv/bin/python tests/test_benchmark_queue.py -v
```

Queue tests use temporary lock files. Production benchmarks use the shared
`~/ane.lock` and `~/gpu.lock` files.
Regressions cover concurrent coordinators completing without overlapping
work, active-worker detection, and script paths or arguments containing spaces,
quotes, empty values and Unicode. The coordinator regression substitutes model
workers while exercising the real coordinator, native process scanner and locks.

Apple's [compute-unit API](https://apple.github.io/coremltools/source/coremltools.models.html)
and [compute-plan API](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw)
describe the configuration and placement distinction.

## Measured Qwen3-0.6B run on M1

[Saved results and audit](benchmarks/m1-qwen3-0.6b.json) use a MacBook Air M1,
8 GiB RAM, macOS 27.0.1, Core ML Tools 9.0 and PyTorch 2.5.1. The Q8 checkpoint
SHA-256 is `7a67926eb5be74ab87e4dc29db81f3077d98f686df6a65591eeb333ccfa9ca6e`,
identical to the saved Asahi Qwen3 checkpoint. Three fresh workers per device,
two full warmups per prompt, and 16 output tokens produced these medians:

| Prompt tokens | CPU+GPU prefill tokens/s | CPU+ANE prefill tokens/s | CPU+GPU decode tokens/s | CPU+ANE decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 36.81 | 33.95 | 8.44 | 6.37 |
| 16 | 137.54 | 112.73 | 8.99 | 6.98 |
| 32 | 250.00 | 205.06 | 8.29 | 6.92 |
| 65 | 168.13 | 152.83 | 8.14 | 5.80 |

**Decode uses one real token padded to 32 rows**, with all padding work and
CPU operations timed. These measurements describe this harness, with explicit
cache transfers and a CPU vocabulary head. They do not describe an optimized
single-token runtime or an OS speedup over Asahi's different arithmetic.
Clocks were not fixed, host snapshots recorded swap usage, and raw timings
varied across trials; the saved file retains all samples.

The independent audit passed **384 full-logit vectors and 384 timed token
choices**, with zero top-1 mismatches. Worst NRMSE was 0.001433 for GPU and
0.002091 for ANE, below the fixed 0.005 gate. Each ANE decoder block preferred
the ANE for 28 projection operations and CPU for 355 operations; each GPU block
preferred GPU for 383 operations. The 382 constants per block have no assigned
device. These are compiler preferences, rather than utilization measurements.

The optional `CPU_ONLY` control failed the same gate: worst NRMSE 0.051632,
despite zero top-1 mismatches. Its 64 logit checks were independently recomputed
and retained as an **unqualified diagnostic** in the saved file. Its numerical
error needs investigation before using it for CPU speed comparisons. GPU and
ANE passed without relaxing the gate.

The published JSON retains the full report, audit, raw timing samples,
per-prediction quality checks, source/model hashes, reference token IDs and
per-block placement counts. Full operation lists and binary validation logits
remain in `macos/local-results/qwen3-0.6b-m1-001/`; rerunning `macos.audit`
requires these ignored local files. The CPU diagnostic's raw run is
`macos/local-results/qwen3-0.6b-pilot-001/`.

The stricter auditor verified this run and the completed long sweep:
**672 full-logit comparisons and 672 timed token checks passed**.
[Review audit receipts](benchmarks/m1-qwen3-0.6b-review-audit.json) record the
auditor hash, report hashes and published-result hashes. Historical partial
audits are retained separately from these totals.

## Longer prompts: 512, 1024 and 2048 tokens

The fixed inputs in [long-prompts.json](benchmarks/long-prompts.json) repeat
the [technical source passage](benchmarks/long-prompt-source.txt) and take
prefixes of exactly 512, 1024 and 2048 token IDs. This is a synthetic scaling
workload. [Prompt provenance](benchmarks/long-prompt-provenance.json) records
the tokenizer revision, hashes and construction method.

[Saved long-prompt results](benchmarks/m1-qwen3-0.6b-long.json) contain these
medians on the same M1 MacBook Air. **Three trials per device completed**,
with two full warmups per prompt and 16 output tokens per request:

| Prompt tokens | CPU+GPU prefill tokens/s | CPU+ANE prefill tokens/s | CPU+GPU decode tokens/s | CPU+ANE decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 62.78 | 40.26 | 2.42 | 1.33 |
| 1024 | 71.89 | 43.54 | 2.52 | 1.29 |
| 2048 | 74.72 | 22.74 | 3.03 | 0.75 |

The completed sweep and its independent audit are **PASS**. GPU trial 2
initially failed before validation or timing: one block's compute plan
returned only unknown device preferences. Three later attempts could not
compile under the managed sandbox. Two unrestricted attempts then ran out of
disk space during compilation. Removing the verified, regenerable Hugging
Face source download from this task's cache freed 1.4 GiB but was insufficient.
After additional disk space was freed, the sixth retry's fresh GPU worker
completed all three prompt lengths using the exact, hash-verified
source snapshot from the original sweep. Its model execution matches the
fixed working runner. No measured sample was discarded.

The [earlier partial result](benchmarks/m1-qwen3-0.6b-long-partial.json), original
failed sweep, failed initialization receipts, successful retry, launchers and
source snapshot are retained. The missing GPU trial is now complete. Source
hashes describe the historical runner; the current stricter auditor records
its own hash separately. Preferred devices describe compiler placement.

The independent audit passed **288 full-logit vectors and 288 timed token
choices**, with zero top-1 mismatches. Worst NRMSE was 0.001305 for GPU and
0.001758 for ANE, below the unchanged 0.005 gate. Every GPU block preferred GPU
for 383 operations; every ANE block preferred ANE for 28 projections and CPU
for 355 operations.

All timings include the CPU embedding and vocabulary head, explicit cache
transfers, and decode padded to 32 rows. Trial variation was substantial:
2048-token ANE prefill ranged from 17.02 to 42.83 tokens/s. Host snapshots
record load and swap activity; clocks were not fixed. All completed samples
remain in the published JSON.

These requests use a logical context of **2080** positions, allowing a
2048-token prompt and 15 subsequent decode inputs. The padded cache has 2111
positions for every prompt length in this sweep. The checkpoint differs from
the earlier 512-context checkpoint only in its context header; the complete
coefficient and norm payload is identical. All seven converted coefficient
blobs and the embedding also match the earlier conversion byte for byte.

The longer-context conversion uses `--single-function` to export just the
32-row function used by both phases. `--reuse-weights-from` shares identical,
immutable coefficient files after checking their complete hashes. This
reduces temporary conversion storage and avoids duplicating the weights.

```sh
# When creating the assets on another machine:
macos/.venv/bin/python ane/export_checkpoint.py /path/to/hf-snapshot \
  macos/local-models/Qwen3-0.6B-context2080.bin --context 2080
macos/.venv/bin/python -m macos.export \
  macos/local-models/Qwen3-0.6B-context2080.bin \
  --output macos/local-models/qwen3-0.6b-coreml-context2080 \
  --context 2080 --batch 32 --block-layers 4 --single-function \
  --reuse-weights-from macos/local-models/qwen3-0.6b-coreml

# Reuse existing assets and choose a new output directory:
macos/.venv/bin/python -m macos.benchmark \
  --model macos/local-models/qwen3-0.6b-coreml-context2080 \
  --checkpoint macos/local-models/Qwen3-0.6B-context2080.bin \
  --prompts macos/benchmarks/long-prompts.json \
  --output macos/local-results/long-run-001 --modes gpu ane \
  --trials 3 --warmups 2 --new-tokens 16
macos/.venv/bin/python -m macos.audit macos/local-results/long-run-001 \
  --output macos/local-results/long-run-001/audit.json
```

The arithmetic, accuracy gates and timing boundaries follow the earlier
protocol. The larger fixed cache changes the amount of attention work and
cache traffic, so short-prompt results from the 512-context graph are a
separate configuration.

The completed sweep can be audited again using the retained local binary
logits and operation lists. Choose an unused output filename for each repeat:

```sh
macos/.venv/bin/python -m macos.audit \
  macos/local-results/qwen3-0.6b-m1-long-qualified-001 \
  --output macos/local-results/qwen3-0.6b-m1-long-qualified-001/audit-rerun-001.json
```
