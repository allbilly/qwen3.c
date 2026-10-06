# Continue bare-board long-context sweep

INTENT: the long-context harness aborts on cooldown timeout and marks clock drops as failed screens; the user permits continuing with documented clock drops; fp16/experiments/yalm/LONG-CONTEXT.md currently requires <=55°C starts and strict clock checks.

User direction: "even clock drop u just mention it in doc and continue run it is ok ?"
Scope: comparison harness, continuation controller, auditors and rendered report/evidence. Shared FP16 checkpoint, isolated compiled source/binary, selected executable and NPU submission metadata stay frozen. No kernel thermal protection changes. No overwritten result directories.

- [x] Add explicit harness flags for accepting documented clock drops and disabling software cooldown. Record both in configs/summaries; keep quality checks unchanged.
- [x] Build the missing-six-job continuation plan with the same checked NPU references and fresh output namespace; dry-run and inspect it.
- [ ] Run GPU4K, mixed4K placements, CPU2K and CPU4K serially. Two complete warmups and two measurements, 32 outputs/31 decode steps; record actual clock/temperature samples and normal restoration.
- [x] Independently audit every completed request, retained numerical failures, actual devices, source/model/binary hashes and restoration evidence. Explicitly record the interrupted restoration as unverified. Clock stability stays a separate observation, not a stop condition.
- [x] Export evidence losslessly; render all18 slots (15 measured,3 missing) with thermal/protocol labels, prefill ms and tokens/s, decode tokens/s and independent request ms. Distinguish sessions and record actual clock ranges.
- [x] Preserve this local WIP milestone in a separate Git directory and a portable bundle; original .git stays read-only. No push.


## Restricted-session recovery

The previous continuation process handle is gone. GPU4K contains all four complete requests, but no summary or clock/fan restoration artifacts. Its captured clock stream stopped several hours before the resumed session. Live CPU/NPU/DDR governor/limit settings remain locked; GPU governor/limits match its original snapshot. This is not recorded as successful restoration.

The current managed sandbox exposes no /dev/dri/card1 or Mali node. The known ~/rk3588/examples/simple_add.py check fails with FileNotFoundError for /dev/dri/card1; the unchanged inference runner also fails during NPU initialization. This does not establish absent hardware on the board; these devices are not exposed to this session.

Continue CPU2K/4K in `/tmp/qwen3-long-context-cpu-sample-only-20261006`, observing existing clocks without Docker, sysfs writes or changes to device settings. Use the same frozen runner/model, teacher IDs, two warmups/two measurements and 32 outputs. The new sample-only mode records actual before/after limit snapshots and explicitly does not claim to restore the earlier locked clocks. The remaining three mixed-device4K rows need a session exposing the board devices. New git commits are also restricted by the current read-only .git permission profile.


### WIP artifact under current permissions

If the protected original .git remains read-only, preserve a local WIP commit in a separate Git directory under /tmp, reading the original objects and saving a portable bundle in the writable workspace. This does not update the protected original branch or expose hardware devices. Report the bundle commit and that the original branch remains at744158b. Do this after timed CPU work and verified evidence exports.

## Observed milestone

CPU2K and CPU4K each finished two warmups/two measured requests, and their controller exited0. GPU4K raw recovery plus the original12 rows produces15/18 measured placements. CPU2K/4K match every actual generated ID but fail the unchanged0.001 logit gate. All completed raw requests/device counts/timers and frozen hashes rechecked;261 lossless exports in20 directories verified. PNG/JPG/SVG and the table show both phases, independent resident request spans and actual clock/temperature ranges. The three mixed4K jobs remain unmeasured because current device nodes are hidden. The first execution checkbox stays open.
