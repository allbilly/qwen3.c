# Requested investigation: evidence and limits

The requested architecture, roofline, profiling, measured optimizations and
Rockchip hybrid experiments are complete. The selected CPU-host/direct-register
NPU engine is retained. New concurrent FFN runners are isolated experiments.
No guaranteed hybrid gain or full-device transformer is claimed.

| Request | Observed result | Evidence |
|---|---|---|
| Matmul offload or full register inference? | CPU schedules the transformer. Native NPU tasks execute projections and prompt QK/PV matmuls; CPU handles embedding, norms, RoPE, softmax, SwiGLU, residuals, sampling and decode attention. This does not call RKLLM or the RKNN matmul API. | [Architecture](../../README.md), [selected native backend](../../fp16_backend.c), [complete profile](../../PROFILE.md) |
| Roofline like Martin, with PNG/JPG? | An empirical roofline uses observed large-working-set streaming bandwidth and checked FP16 matrix rates, with logical weight traffic and end-to-end points. PNG/JPG/SVG and source/data provenance are retained. | [Report](../../ROOFLINE.md), [PNG](../../roofline.png), [JPG](../../roofline.jpg), [profile and accounting](../../PROFILE.md) |
| Achievable memory bandwidth; decode at 90%? | Four-core CPU streaming read: 26.698 GB/s. Three-core native NPU weight streaming: 30.941 useful GB/s. Best observed native GEMM: 1739.44 GFLOP/s. Weight-only decode ceiling: 25.958 tokens/s; 90% is 23.362. Whole-model decode reaches about 78–80%, driver-only about 93–94%. There is no physical DDR byte counter, so these are algorithmic-payload estimates. | [Roofline measurements](../../ROOFLINE.md), [later complete cost profile](COSTS.md) |
| Fair stock comparison | Same pinned Qwen3-0.6B revision and all 311 audited FP16 tensors; stock uses `do_quantization=False`. Custom prefill is 18.1% / 12.2% faster and decode 8.7% / 8.7% faster in that matched comparison. Different reduction order is explicit. This is the earlier certified phase comparison, not a fresh stock complete-request experiment. | [Matched protocol and raw results](../../ROOFLINE.md) |
| Compute, memory movement and communication costs | Complete 14-stage coverage: 99.9441–99.9992%. Separate host packing/copy/unpack, DMA synchronization, submission residual, blocking NPU driver span, CPU work and OpenCL kernel/copy/API-wait spans. Driver timing includes memory stalls and completion work; pure ALU and physical DDR time cannot be isolated by these counters. | [Complete cost profiles](COSTS.md), [earlier detailed profile](../../PROFILE.md) |
| Register and layout optimization | Checked three-core down bundles and balanced classifier tiles improve primitives and selected prefill. Thirty bounded register candidates retained without unsupported speed claims. Later attention batch128 improves prefill by 4.71–4.84%, with overlapping complete-request ranges. Native concurrent packing removes gate/up assembly but does not beat the selected route. | [Selected changes](../../ROOFLINE.md), [paired iterations](ITERATIONS.md), [concurrent costs](CONCURRENT.md), [requests](CONCURRENT-REQUESTS.md) |
| GPU comparison and combinations | All nine CPU/GPU/NPU projection-backend pairs, plus an explicit three-device placement, are measured for prefill/decode and independently timed requests. CPU remains the transformer host. These custom Mali prototypes are not a fully GPU-resident baseline. The installed llama.cpp OpenCL build rejects Mali; no Vulkan ICD is available. An exact FP16 GGUF is prepared, but no unsupported llama.cpp GPU FPS is reported. | [Placement phase results](PHASES.md), [complete requests](REQUESTS.md), [GPU limitations](../../ROOFLINE.md), [hybrid attention](../../HYBRID.md) |
| Try concurrent CPU/GPU/NPU like oMLX | Disjoint 96-channel auxiliary FFN slices run asynchronously with a three-core NPU prefix. Original and native-packing layouts: 24 primitive cases / 3,551,232 checked outputs, sixty full-model cases, eight branch profiles, forty complete-request jobs and eight ABBA decode jobs. Raw calibrated traces independently reproduce 84/112 overlapping intervals in each of two warmups and one measured diagnostic. Scheduling instrumentation is excluded from throughput claims. | [Implementation and checks](CONCURRENT.md), [phase/request results](CONCURRENT-REQUESTS.md), [raw trace audit](concurrent-overlap-trace-audit.json) |
| Measure phases before complete inference | Event-enabled phase/branch costs are diagnosed before unprofiled independently timed requests. Prefill ms **and tokens/s**, decode tokens/s and complete-request ms/ranges are shown. CPU-assisted decode's apparent screen gain does not repeat in ABBA; no split is promoted. | [Costs](COSTS.md), [placements](REQUESTS.md), [paired changes](ITERATIONS.md), [concurrent requests](CONCURRENT-REQUESTS.md) |
| WIP commit per milestone | Local commits retain code, failures, source/model hashes, small raw evidence, auditors and rendered figures. The final concurrent export verifies 267 additional evidence files losslessly; all three observer exports reproduce exact tested binaries offline. Selected engine SHA remains `100716de...`. | Commits `386f5cd`, `cf13ceb` and the following request milestone; [export verification](concurrent-final-export-verification.json), [observer rebuild verification](concurrent-observer-rebuild-verification.json) |

## Quality and remaining limitations

All thirty concurrent decode-only cases pass the unchanged checks; first-token
logits are bit-identical to the selected prefill and every tested prediction
matches. Both-phase splitting fails the unchanged first-logit gate at 24/73
tokens in each layout: twelve failures are retained. The forty long-prompt
request jobs pass their local gates, but that does not qualify the prefill
splits across arbitrary inputs. No tolerance was relaxed.

The counterbalanced CPU-assisted decode check has overlapping ranges and
median decode regressions of 2.01% / 1.49%. GPU branch jobs and inter-device
movement can erase local savings. Results apply to this FP16 model, context,
board, clocks and tested inputs.

A fully GPU-resident transformer, larger tuned concurrent partitions, KV
packing reuse and cooperative GPU softmax are future implementations. They
are not prerequisites for the completed investigation or promised speedups.
No claim of 90% physical DDR utilization, isolated ALU timing, Apple-equivalent
precision or GPU-resident performance is made. Hardware jobs exited normally;
all performance controllers restored the original clocks. No push was requested.

INTENT: the router waits for each device projection; the user requests mixed CPU/GPU/NPU inference and transfer-inclusive phase and request measurements; fp16/README.md requires the shared FP16 checkpoint, numerical checks and serial NPU submissions.

INTENT: the first split assembles and repacks full FFN tensors; the user requests profiling followed by concrete prefill improvements; fp16/README.md describes direct activation packing between FFN projections with the verified FP16 operands.

AUTH: user said "wip commit per milestone".
