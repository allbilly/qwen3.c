# Long-context benchmark status

The common 1.8GHz bare-board sweep has exited normally with a cooldown guard failure. No inference, controller or fan worker remains active. All 11 started controllers restored their original CPU/NPU/DDR/GPU governors and limits, independently verified against live sysfs; the original fan command was restored. No native process was killed.

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

A fresh cooled benchmark session is needed to complete the same protocol. Keep all current bare-board outputs immutable. Do not mix them with cooled measurements. Investigate CPU 2K numerical error before accepting a future row. The optional user preference between installing cooling and a new lower-clock protocol has not been answered; silence is not a pause request or approval. The current sweep stopped on its programmed guard.

Historical failed 51°C, one-time PWM 55°C and feedback 2.256GHz attempts remain separate diagnostics. Their raw logs and clock/fan restoration are exported in previous WIP milestones. All NPU init/alloc/sync/submits were serial; no driver unbinding or kernel register changes.

## Evidence

- long-context-audit.json: independent audit of 12 measured rows, source/model/binary/input/logit checks, actual predictions, device counts, phase/request timers, sampled clocks and restoration snapshots.
- long-context-cpu2048-abort-audit.json: separate zero-measurement warmup diagnostic.
- long-context-cooling-restoration.json: all 11 started controllers and live limits restored; five not-started jobs explicitly listed.
- long-context-execution-plan-screen55-cpu1800-fanheld.json: original 16-controller/18-row plan and order.
- long-context-fan-screen55-cpu1800-fanheld-* evidence: all feedback overrides, maximum polling gap, before/restored command and summary. Does not measure cooling effectiveness.

INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.
