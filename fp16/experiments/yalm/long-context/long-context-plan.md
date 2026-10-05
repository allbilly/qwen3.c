# Long-context benchmark extension

Scope: isolated e2e/long-context sources and long-context preparation/check/run/audit/report tools in the comparison workspace, exported source/evidence and tables in qwen3.c/fp16/experiments/yalm. Selected engine and shared FP16 container stay frozen.

INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.

- [ ] Add checked runtime context override, dynamic host/GPU scratch and bounded NPU attention query blocks; keep original submission metadata and the four-data-bank guard.
- [ ] Compile source snapshots and check large NPU QK/PV primitives against identical FP16 operands; verify short-prompt regression logits and actual device execution.
- [ ] Prepare immutable exact-length prefix prompts and long NPU golden logits/IDs; preserve source/model/binary/input hashes.
- [ ] Run six placements × three prompt lengths serially: CPU only, Mali OpenCL, NPU, CPU+NPU FFN decode, GPU+NPU FFN decode, all three FFN decode. Same precision, four host threads, clocks, two complete warmups, two measurements, 32 outputs and common <=51C starts.
- [ ] Independently audit every prediction/logit, phase counts, request timers, sampled clocks and restoration. Retain failed quality or clock jobs as diagnostics.
- [ ] Generate tables with best-observed asterisks, export small evidence/source, render charts and locally WIP commit each completed milestone. No push requested.
