# 1K / 2K / 4K prompt benchmark

15/18 placements were measured. 13/15 pass the unchanged numerical gate; 12/15 also pass the exact sampled-clock check.

The user permits continuing with documented clock drops. The original 55°C sweep stopped during CPU 2K; its 12 measured rows are preserved. Continuation rows use the same model, executable, thread count and measured request protocol, with software cooldown disabled. Actual frequencies and clock-control modes are recorded below. Kernel thermal protection remains active. These are observed bare-board results from two starting-temperature protocols, not a fixed-clock causal speedup comparison.

The continuation process handle is unavailable. GPU 4K has four complete raw requests and passing numerical/device/timer checks, but no parent summary or restoration artifacts. The later managed session exposes no DRM/Mali nodes: the known add check fails on missing /dev/dri/card1. CPU 2K/4K run with the same frozen executable and passive clock sampling, using existing limits without changing sysfs or fan settings. The earlier CPU/NPU/DDR limits and fan command remain locked; successful restoration is not claimed. The three mixed-device 4K rows still need a session exposing board devices.

The aborted CPU 2K warmup matched all generated IDs but had first-logit relative RMSE 0.00111339, above the unchanged 0.001 gate. It is retained as a failed diagnostic, with zero measured requests. Installing cooling does not establish numerical correctness; investigate this difference before accepting a future CPU 2K row.

## Same-model screening comparison

| Input tokens | Placement | Prefill ms ↓ | Prefill tokens/s ↑ | Decode tokens/s ↑ | Request ms ↓ | Audit | Start condition |
|---|---|---:|---:|---:|---:|---|---|
| 1024 | CPU only | 63751.00 | 16.06 | 12.43 | 66245.42 | pass; clock drop | <=55°C |
| 1024 | GPU OpenCL | 23820.43 | 42.99 | 7.35 | 28037.46 | pass | <=55°C |
| 1024 | NPU | 4246.48 | 241.14 | 13.19 | 6597.71 | pass | <=55°C |
| 1024 | CPU+NPU | 4229.70 | 242.10 | **13.24**\* | **6572.43**\* | pass | <=55°C |
| 1024 | GPU+NPU | 4256.27 | 240.59 | 13.20 | 6604.64 | pass | <=55°C |
| 1024 | CPU+GPU+NPU | **4225.21**\* | **242.35**\* | 13.05 | 6600.76 | pass | <=55°C |
| 2048 | CPU only | 314929.42 | 6.50 | 3.71 | 323395.67 | quality fail; target clocks differ | no forced cooldown |
| 2048 | GPU OpenCL | 62036.77 | 33.01 | 5.22 | 67978.29 | pass | <=55°C |
| 2048 | NPU | 12501.28 | 163.82 | **9.35**\* | 15816.94 | pass | <=55°C |
| 2048 | CPU+NPU | 12096.94 | 169.30 | 9.22 | 15457.58 | pass | <=55°C |
| 2048 | GPU+NPU | 12139.49 | 168.71 | 9.21 | 15505.91 | pass | <=55°C |
| 2048 | CPU+GPU+NPU | **12082.94**\* | **169.50**\* | 9.26 | **15430.26**\* | pass | <=55°C |
| 4096 | CPU only | 1176149.51 | 3.48 | 3.02 | 1186426.40 | quality fail; target clocks differ | no forced cooldown |
| 4096 | GPU OpenCL | 204740.56 | 20.01 | 2.82 | 215726.84 | pass; restoration unverified | no forced cooldown |
| 4096 | NPU | 45443.88 | 90.13 | 5.28 | 51320.79 | pass | <=55°C |
| 4096 | CPU+NPU | — | — | — | — | not run: device nodes hidden | — |
| 4096 | GPU+NPU | — | — | — | — | not run: device nodes hidden | — |
| 4096 | CPU+GPU+NPU | — | — | — | — | not run: device nodes hidden | — |

\* Best observed value among numerical passes, only where all six placements were measured. Clock drops are permitted and explicitly recorded. Asterisks do not denote a controlled fixed-clock winner or statistical significance. Numerical failures remain diagnostics and cannot win.

Mixed rows split FFN gate/up channels during decoding only (CPU 96 and/or GPU 96 channels, with the remaining channels on NPU). Their prefill follows the same NPU path. Small prefill differences among these rows are run variation. CPU host operations remain present in every accelerated placement.

CPU only runs projections and attention on CPU with no NPU/GPU execution. GPU OpenCL runs both projections and attention on Mali using the custom kernels, with CPU embedding, normalization, RoPE, SwiGLU, residuals and sampling. The NPU route offloads dense matrices and prompt QK/PV to native registers, with CPU decode attention. These GPU results do not represent a tuned, GPU-resident llama.cpp baseline.

![Long-context phase and request comparison](long-context-comparison.png)

[PNG](long-context-comparison.png) · [JPG](long-context-comparison.jpg) · [SVG](long-context-comparison.svg)

## Protocol and correctness

The device tree identifies Orange Pi 5 / RK3588S, and the user reports no heatsink installed. These are current bare-board measurements, with sampled temperatures reaching 85.888°C. PWM commands do not establish physical fan presence or cooling effectiveness. A heatsink with a fan is recommended; [52Pi EP-0167](https://wiki.52pi.com/index.php?title=EP-0167) is specified by its manufacturer for the original Orange Pi 5. A future cooler installation needs a separate benchmark session.

All rows use the same pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` and shared FP16 container `c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879`. Effective weight tensors and weight quantization match. Intermediate precision and reduction order can differ: CPU prompt attention uses FP32, whereas NPU/Mali prompt attention uses FP16 operands. These are checked by the unchanged logit gate. The original 512-token container header is unchanged; the isolated runner expands runtime KV/scratch capacity to 4128. The model configuration permits 40960 positions with RoPE base 1e6. The selected executable `100716de...` remains unchanged.

Original-session requests start at <=55°C; continuation requests have no forced cooldown. CPU/NPU/Mali/DDR targets are 1.800/1.000/1.000/2.112GHz, with four inference threads on CPU4–7. Clocks are sampled every 250ms. The original 12-row session restored governors/limits and the fan command. Its PWM255 feedback setpoint was reasserted when the kernel temperature notifier changed it, with 10ms polling on CPU0–3. Raw PWM samples, all reassertions and maximum polling gap are retained. The interrupted continuation has no verified restoration. Later CPU-only jobs sample existing settings without sysfs or fan writes. Kernel thermal protection remains active.

Two complete warmups precede two measured requests per measured row. Each request has 32 output tokens and 31 decode steps. Prefill tokens/s is input tokens divided by the measured span through the first classifier/argmax. Decode tokens/s is 31 divided by the decode span. Request time is independently measured from prefill start through the last argmax. Activation packing, memory movement, DMA synchronization, host work, queue waits and handoffs are included. Loading, tokenization, ID-file reading, cooldown and warmups are excluded.

CPU 2K/4K in the restricted session use passive sampling of existing governors/limits. The comparison targets are not reapplied, and the GPU uses its existing ondemand governor. An idle GPU at 300MHz in those CPU-only rows is a target mismatch, not evidence of active GPU throttling. Before/after snapshots record whether limits stayed equal; the CPU harness does not write them. Those snapshots do not establish restoration of the earlier interrupted session.

All routes use identical input IDs and identical teacher IDs during decoding. The NPU references run without teacher forcing, and every actual prediction from all four requests is retained and checked. Full-vocabulary first logits must be finite and have relative RMSE<0.001 against the checked long NPU reference. This reference is not an independent Hugging Face full-model oracle. A failed comparison identifies disagreement with that reference; it does not establish which route is closer to an independent oracle.

The [wide QK/PV checks](long-context/wide-attention-check.jsonl) compare 514048 outputs against double accumulation of identical FP16 operands. Maximum relative RMSE 2.02905781962e-7 passes the unchanged 1e-5 gate. All six short regressions pass, and [all 151936 first logits are bit-identical to the prior corresponding routes](long-context/long-context-short-parity.json). Native PV submissions retain the four-data-bank limit and use 64/32/16 query rows at 1K/2K/4K.

The [immutable prompts](long-context/prompts/provenance.json) repeat the existing 128-token benchmark prompt and take nested prefixes of exactly 1024/2048/4096 tokens. They measure synthetic throughput, rather than natural long-document task quality.

## Raw measured ranges and checks

| Input | Placement | Prefill min–max ms | Decode min–max tokens/s | Request min–max ms | First-logit relative RMSE | All actual IDs match |
|---|---|---:|---:|---:|---:|---|
| 1024 | CPU only | 63698.372–63803.618 | 12.308–12.550 | 66168.475–66322.372 | 0.00096904935 | True |
| 1024 | GPU OpenCL | 23814.192–23826.678 | 7.348–7.355 | 28029.321–28045.593 | 0.0007862517 | True |
| 1024 | NPU | 4222.575–4270.392 | 13.043–13.329 | 6596.130–6599.295 | 0 | True |
| 1024 | CPU+NPU | 4189.175–4270.224 | 12.983–13.492 | 6486.914–6657.951 | 0 | True |
| 1024 | GPU+NPU | 4231.611–4280.939 | 13.113–13.290 | 6564.291–6644.989 | 0 | True |
| 1024 | CPU+GPU+NPU | 4187.703–4262.719 | 12.961–13.140 | 6547.000–6654.527 | 0 | True |
| 2048 | CPU only | 291213.869–338644.966 | 3.301–4.111 | 300605.718–346185.622 | 0.0011133907 | True |
| 2048 | GPU OpenCL | 62031.626–62041.911 | 5.204–5.231 | 67968.300–67988.277 | 0.00098512171 | True |
| 2048 | NPU | 12292.580–12709.983 | 9.293–9.407 | 15628.563–16005.315 | 0 | True |
| 2048 | CPU+NPU | 12063.078–12130.810 | 9.197–9.252 | 15433.843–15481.325 | 0 | True |
| 2048 | GPU+NPU | 12071.907–12207.081 | 9.129–9.290 | 15409.020–15602.791 | 0 | True |
| 2048 | CPU+GPU+NPU | 11936.147–12229.724 | 9.252–9.271 | 15280.035–15580.495 | 0 | True |
| 4096 | CPU only | 1174682.872–1177616.154 | 2.963–3.071 | 1185143.547–1187709.244 | 0.0010091786 | True |
| 4096 | GPU OpenCL | 204618.522–204862.590 | 2.807–2.836 | 215661.853–215791.818 | 0.00088939649 | True |
| 4096 | NPU | 45378.277–45509.487 | 5.269–5.281 | 51248.645–51392.943 | 0 | True |

Min–max spans contain only two observations; they are not confidence intervals. The run order is retained in the execution plan. No counterbalanced long-context speedup or fresh stock-RKLLM comparison is claimed.

## Phase findings and next checks

In the second measured CPU 1K request, prompt attention took 52939.918ms of 63803.618ms prefill (about 83%). Its linear calls took 10405.982ms, including 162.843ms packing. CPU prompt attention is the main measured bottleneck in that row. At 1K the mixed placements keep the same NPU prefill path, and their small differences overlap the measured NPU range; this screen does not establish a repeatable mixed-device gain.

The resident request span includes packing, transfers, synchronization and host work. It is separately timed; a faster individual phase is insufficient evidence of an end-to-end gain. Existing [transfer and compute profiles](COSTS.md) and [roofline measurements](../../ROOFLINE.md) remain separate artifacts. Wall spans and logical byte counts do not measure physical DDR traffic. The GPU baseline uses custom OpenCL kernels and is not a tuned GPU-resident llama.cpp baseline.

Finish the remaining mixed-device 4K rows once board devices are exposed, preserving these recorded sessions. A later cooled comparison would use fresh outputs and all placements under one protocol. Numerical checks remain unchanged, and neither passive clock sampling nor cooling establishes that the CPU logit checks pass.


## CPU phase profile

Medians of the two measured requests, in milliseconds. Packing is included in linear wall time. These host timer spans do not measure physical DDR bandwidth. Failed numerical rows remain diagnostics.

| Input | Prefill linear wall | Included pack | Linear compute | Prompt attention | Attention / prefill | Decode linear wall | Included pack | Linear compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 10424.050 | 162.912 | 10261.050 | 52863.660 | 82.92% | 1643.517 | 6.028 | 1636.391 |
| 2048 | 26109.960 | 377.753 | 25732.122 | 287851.125 | 91.40% | 4241.012 | 9.390 | 4230.339 |
| 4096 | 47154.413 | 692.352 | 46461.962 | 1126976.467 | 95.82% | 3186.793 | 8.193 | 3177.386 |

The attention percentage is the ratio of median attention time to median prefill time. Decode attention and other CPU host operations are included in the request/decode spans, outside the reported linear timer.

## Reproducibility

[Independent 15-row audit](long-context/long-context-observed-audit.json) · [original execution plan](long-context/long-context-execution-plan-screen55-cpu1800-fanheld.json) · [original cooling-control audit](long-context/long-context-cooling-restoration.json) · [aborted warmup audit](long-context/long-context-cpu2048-abort-audit.json) · [source/checklist](long-context/long-context-plan.md)

The first 51°C attempt stopped normally at the bounded cooldown guard before the second 4K warmup; GPU clock drops were also observed. A subsequent 55°C attempt showed that a one-time fan setting is overridden by the kernel. These attempts are retained as rejected diagnostics and excluded from this table. The later 2.256GHz CPU 1K row also thermally dropped to 2.208GHz; this partial comparison uses a common 1.800GHz CPU target for all measured rows. The max-clock NPU/GPU timings remain separate diagnostics. The [board kernel source](https://github.com/orangepi-xunlong/linux-orangepi/blob/orange-pi-6.1-rk35xx/drivers/hwmon/pwm-fan.c) implements that temperature notifier.

The exported `long-context/build.py` reproduces isolated runner `f8b4f9eb...`. Full source, model, binary, input, raw-log and logit hashes are recorded in provenance/audits. Small raw evidence is exported losslessly; original checkpoint, executables and binary logits remain in `/home/orangepi/qwen3-bench/matched/roofline/yalm/e2e/long-context`. The passive CPU outputs remain in `/tmp/qwen3-long-context-cpu-sample-only-20261006`; the new CPU and recovered GPU first-logit files are also exported losslessly as `.f32.gz` alongside raw logs and clock snapshots. Preserve the original output directories to rerun the full audit directly. All NPU initialization/sync/submits ran serially. No push was requested.


## Recorded clocks and temperatures

Ranges cover each complete job, including initialization, two warmups, two measured requests and any original-session cooldown. They are not isolated measured-phase frequency averages. Differences include thermal clamps and idle accelerators using ondemand scaling in sample-only CPU runs.

| Input | Placement | CPU4 MHz | CPU6 MHz | NPU MHz | GPU MHz | DDR MHz | Temperature °C | Dropped samples / total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | CPU only | 1800.000–1800.000 | 1800.000–1800.000 | 800.000–1000.000 | 300.000–1000.000 | 2112.000–2112.000 | 54.538–85.888 | 29/2223 |
| 1024 | GPU OpenCL | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–69.307 | 0/1603 |
| 1024 | NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 44.384–61.000 | 0/127 |
| 1024 | CPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–67.461 | 0/668 |
| 1024 | GPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–68.384 | 0/537 |
| 1024 | CPU+GPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–68.384 | 0/506 |
| 2048 | CPU only | 408.000–1800.000 | 408.000–1800.000 | 800.000–1000.000 | 300.000–300.000 | 2112.000–2112.000 | 49.923–85.888 | 4736/4736 |
| 2048 | GPU OpenCL | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–73.923 | 0/2313 |
| 2048 | NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–67.461 | 0/309 |
| 2048 | CPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–73.923 | 0/966 |
| 2048 | GPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–74.846 | 0/966 |
| 2048 | CPU+GPU+NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–73.923 | 0/879 |
| 4096 | CPU only | 408.000–1800.000 | 408.000–1800.000 | 1000.000–1000.000 | 300.000–300.000 | 2112.000–2112.000 | 77.615–85.000 | 19341/19341 |
| 4096 | GPU OpenCL | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 41.615–80.384 | 0/3442 |
| 4096 | NPU | 1800.000–1800.000 | 1800.000–1800.000 | 1000.000–1000.000 | 1000.000–1000.000 | 2112.000–2112.000 | 54.538–78.538 | 0/1025 |

[Continuation plan](long-context/long-context-execution-plan-continue-throttle-cpu1800-fanheld.json) · [session recovery observations](long-context/long-context-session-recovery.json) · [recovered GPU audit](long-context/long-context-gpu4096-recovered-audit.json) · [CPU passive-sampling plan](long-context/long-context-cpu-sample-plan.json) · [continuation checklist](long-context/long-context-continue-plan.md)

INTENT: the long-context harness aborts on cooldown timeout and marks clock drops as failed screens; the user permits continuing with documented clock drops; fp16/experiments/yalm/LONG-CONTEXT.md currently requires <=55°C starts and strict clock checks.

AUTH: user said "wip commit per milestone".
