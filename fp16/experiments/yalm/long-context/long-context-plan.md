# Long-context benchmark extension

Scope: isolated e2e/long-context sources and long-context preparation/check/run/audit/report tools in the comparison workspace, exported source/evidence and tables in qwen3.c/fp16/experiments/yalm. Selected engine and shared FP16 container stay frozen.

INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.

- [x] Add checked runtime context override, dynamic host/GPU scratch and bounded NPU attention query blocks; keep original submission metadata and the four-data-bank guard.
- [x] Compile source snapshots and check large NPU QK/PV primitives against identical FP16 operands; verify short-prompt regression logits and actual device execution.
- [ ] Prepare immutable exact-length prefix prompts and long NPU golden logits/IDs; preserve source/model/binary/input hashes.
- [ ] Run six placements × three prompt lengths serially: CPU only, Mali OpenCL, NPU, CPU+NPU FFN decode, GPU+NPU FFN decode, all three FFN decode. Same precision, four host threads, clocks, two complete warmups, two measurements, 32 outputs and common <=55C starts.
- [ ] Independently audit every prediction/logit, phase counts, request timers, sampled clocks and restoration. Retain failed quality or clock jobs as diagnostics.
- [ ] Generate tables with best-observed asterisks, export small evidence/source, render charts and locally WIP commit each completed milestone. No push requested.

Milestone7cedff6: all514048 primitive outputs pass; all six128-token cases pass quality/device/clocks and first logits are bit-identical to previous routes. Exported source reproduces exactf8b4f9eb binary;31 evidence files are lossless. Long controller started, with18 total benchmark rows.

The first51C attempt completed1K/2K but aborted normally before the second4K warmup: after180s cooling it remained51.769C. All clocks were restored. Keep the entire attempt as rejected diagnostics. Rerun all18 rows freshly at a common55C start limit; all other clock, numerical and request-count gates stay unchanged.

Raw51C logs also show five GPU300MHz samples at4K, with a peak84.076C. Board fan PWM was observed at0. Set the existing fan PWM to255 for all fresh55C controllers, record it in every sampled clock row and restore the original value in the outer finally block. Do not disable thermal protection or accept clock drops as fair winners.

The attempted one-time fan255 setting was overridden by the kernel temperature notifier (rockchip,temp-trips). Retain references55 as diagnostics. The final fresh fanheld dataset uses a common full-speed PWM255 feedback setpoint, reasserting observed overrides every10ms; kernel thermal protection and the temperature notifier remain active. Record all reassertions/poll gaps plus the raw sampled PWM, and restore the original PWM after the controller exits. No driver unbinding or kernel register changes. All numerical and exact sampled-clock gates remain unchanged.

For the reference controller containing three prompt lengths, audit clock targets over each complete row (its model initialization, four requests and intervening cooloffs, plus cooling-before-row). Preserve the whole-controller held/failure flag as well. A later4K clock drop must be attributed to the4K row, rather than changing a previously completed1K row. The exact clock targets and all-samples criterion are unchanged.

The full-speed-feedback CPU1K row thermally dropped from2.256GHz to2.208GHz; idle NPU/GPU clocks also dropped. Preserve the complete2.256GHz sweep attempt as diagnostics (three NPU rows passed all clocks; CPU1K failed, GPU1K is finishing normally). For a fair final table, freshly rerun all18 rows at a common CPU1.800GHz target, with NPU/GPU/DDR still1.000/1.000/2.112GHz. Same binary/model,55C starts, fan feedback, four host threads, warmups, output lengths and numerical/all-sampled-clock gates. Use a separate clock helper without changing the existing global script; all original settings must restore.
