# Long-context benchmark status

## Original 55°C session (historical)

The original common 1.8GHz bare-board sweep exited normally with a cooldown guard failure. At that checkpoint its inference, controller and fan workers had ended. All 11 started controllers restored their original CPU/NPU/DDR/GPU governors and limits, independently verified against live sysfs; the original fan command was restored. This restoration applies to that session, not to the later interrupted continuation. No native process was killed.

- Requested matrix: six placements × 1024/2048/4096 input tokens; prefill ms and tokens/s, decode tokens/s and independently timed resident request ms.
- Measured: 12/18 rows. All 12 pass the unchanged first-logit/actual-token checks; 11/12 also pass the strict sampled-clock check.
- CPU 1K is a diagnostic: CPU held 1.8GHz but idle NPU/GPU targets dropped; sampled peak 85.888°C.
- CPU 2K completed one warmup, then the 55°C cooldown guard failed after 180053.191ms at 55.461°C. Zero measured requests. Warmup generated IDs match, but first-logit relative RMSE 0.0011133907476657428 fails the unchanged 0.001 gate. Retain it as a failed diagnostic.
- Other five 4K placements never started. The NPU 4K reference has two complete warmups and two measured requests.
- All six 1K placements are measured, so best qualified values receive an asterisk. No winners are assigned to the incomplete 2K/4K groups.
- Same pinned Qwen3-0.6B FP16 bytes, isolated runtime capacity 4128, unchanged selected binary and 21 compiled source hashes. Four host threads, two warmups/two measurements, 32 outputs/31 decode steps, common <=55°C starts and CPU/NPU/GPU/DDR targets 1.800/1.000/1.000/2.112GHz.
- All 514048 wide QK/PV primitive outputs and six short regression routes passed before the long sweep. All 151936 short first logits are bit-identical to previous corresponding route dumps.

## Cooling and remaining work

Actual board model: original Orange Pi 5 / RK3588S. User reports no heatsink. Recommend a fitted heatsink with a fan; manufacturer example: https://wiki.52pi.com/index.php?title=EP-0167 . PWM commands do not prove a physical fan exists or cools the board. Kernel thermal protection remained active throughout.

The user subsequently permitted continuing despite clock drops, with the drops documented. That instruction governs the continuation below. Keep all current bare-board outputs immutable, retain the CPU 2K numerical failure, and label starting conditions separately. A future cooled comparison would use a fresh output namespace and a common protocol.

Historical failed 51°C, one-time PWM 55°C and feedback 2.256GHz attempts remain separate diagnostics. Their raw logs and clock/fan restoration are exported in previous WIP milestones. All NPU init/alloc/sync/submits were serial; no driver unbinding or kernel register changes.

## Evidence

- long-context-audit.json: independent audit of 12 measured rows, source/model/binary/input/logit checks, actual predictions, device counts, phase/request timers, sampled clocks and restoration snapshots.
- long-context-cpu2048-abort-audit.json: separate zero-measurement warmup diagnostic.
- long-context-cooling-restoration.json: all 11 started controllers and live limits restored; five not-started jobs explicitly listed.
- long-context-execution-plan-screen55-cpu1800-fanheld.json: original 16-controller/18-row plan and order.
- long-context-fan-screen55-cpu1800-fanheld-* evidence: all feedback overrides, maximum polling gap, before/restored command and summary. Does not measure cooling effectiveness.

INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.


## Interrupted continuation (historical)

Controller exec session 32092 started `resume_long_context.py` with six missing rows planned serially. Software cooldown was disabled by removing COOL_REQUEST_C from the environment. Same compiled runner/model, requested CPU1.8/NPU1/GPU1/DDR2.112GHz, two warmups/two measurements, 32 outputs and unchanged numerical checks. New directories have suffix continue-throttle-cpu1800-fanheld, using existing checked references55-cpu1800-fanheld. Order: GPU4K, CPU+NPU4K, GPU+NPU4K, all4K, CPU2K, CPU4K. Only GPU4K produced four complete raw requests; its parent summary and restoration files are absent. The old process handle is unavailable. Do not infer a controller exit cause or successful restoration from that missing handle.


## Completed restricted-session CPU continuation

Old exec session32092 is unavailable; recovered GPU4K contains four complete requests and passes the numerical/device/timer/clock audit (20.00580676 prefill tokens/s,2.8215 decode tokens/s,215726.8356755ms request). Parent summary/restoration artifacts are absent; live CPU/NPU/DDR governors/limits and fan command are still locked, and only GPU limits match its saved baseline. Do not claim successful restoration. The current sandbox exposes no DRM/Mali nodes; the known add check fails before any ioctl.

CPU-only exec session 91919 exited with status0 after completing CPU2K and CPU4K serially via run_cpu_remaining.py. Each completed two full warmups and two measured requests.

| Input | Prefill tokens/s | Decode tokens/s | Resident request ms | First-logit relative RMSE | Numerical gate |
|---|---:|---:|---:|---:|---|
|2048|6.50304445|3.706|323395.670166|0.0011133907476657428|fail|
|4096|3.48255044|3.017|1186426.3957415|0.001009178565721269|fail|

Every generated ID matches. Both failures remain diagnostics at the unchanged0.001 gate; this comparison does not identify which route is closer to an independent full-model oracle. CPU2K first-logit bytes equal the earlier aborted warmup. Both CPU clusters sampled408–1800MHz. Temperatures reached85.888°C for2K and85.0°C for4K. Existing governor/min/max limits stayed equal before/after. Median CPU prompt attention accounts for91.40% of2K prefill and95.82% of4K prefill.

Outputs are /tmp/qwen3-long-context-cpu-sample-only-20261006/p{T}-cpu. Same frozen runner/model/input IDs/teacher IDs, four threads, two warmups/two measurements,32 outputs; software cooldown disabled. The controller read clocks and captured before/after limit snapshots without writing sysfs, fan commands or Docker state. It did not restore the earlier interrupted settings. New CPU and recovered GPU first-logit files are exported losslessly as.f32.gz with raw logs and captured harness/snapshots.

The remaining three mixed4K rows require a session exposing board device nodes. Independent offline audit passes for15 completed rows (13 numerical passes),60 complete requests including warmups, matching inputs, device/phase/request counts and unchanged model/binary/21 compiled sources. All261 files in20 long-context export directories match their originals. The table and PNG/JPG/SVG now show15/18 measured slots, separate starting conditions and actual frequency/temperature ranges. Complete-mode rendering correctly rejects the missing rows.

New git writes remain restricted by the read-only original.git profile. This WIP milestone is preserved in a separate Git directory under/tmp and [a portable bundle](../long-context-wip.bundle) in the writable workspace; the protected original branch stays at744158b. The bundle records its commit in refs/heads/wip-long-context and uses744158b as its prerequisite. No push.
