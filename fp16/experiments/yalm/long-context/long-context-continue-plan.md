# Continue bare-board long-context sweep

INTENT: the long-context harness aborts on cooldown timeout and marks clock drops as failed screens; the user permits continuing with documented clock drops; fp16/experiments/yalm/LONG-CONTEXT.md currently requires <=55°C starts and strict clock checks.

User direction: "even clock drop u just mention it in doc and continue run it is ok ?"
Scope: comparison harness, continuation controller, auditors and rendered report/evidence. Shared FP16 checkpoint, isolated compiled source/binary, selected executable and NPU submission metadata stay frozen. No kernel thermal protection changes. No overwritten result directories.

- [x] Add explicit harness flags for accepting documented clock drops and disabling software cooldown. Record both in configs/summaries; keep quality checks unchanged.
- [x] Build the missing-six-job continuation plan with the same checked NPU references and fresh output namespace; dry-run and inspect it.
- [ ] Run GPU4K, mixed4K placements, CPU2K and CPU4K serially. Two complete warmups and two measurements, 32 outputs/31 decode steps; record actual clock/temperature samples and normal restoration.
- [ ] Independently audit every completed request, retained numerical failures, actual devices, source/model/binary hashes and clock restoration. Clock stability stays a separate observation, not a stop condition.
- [ ] Export evidence losslessly; render all 18 slots with thermal/protocol labels, prefill ms and tokens/s, decode tokens/s and independent request ms. Distinguish both sessions and avoid fixed-clock causal speedup claims.
- [ ] WIP commit each milestone locally; no push.
