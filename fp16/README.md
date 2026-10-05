# RK3588 FP16 engine

Work in progress: the selected CPU+NPU engine, roofline, complete phase
profile and GPU-attention comparison are tested. Full CPU/GPU projection
backends, broader device placements and phase-specific prefill/decode
comparisons are being developed in the comparison workspace. Their results
are not claimed by this checkpoint.

The engine uses persistent direct-register NPU tasks, three-core matrix
projections, batched prefill, fused Q/K/V and gate/up projections, prompt
attention on the NPU, and activation packing directly between feed-forward
projections. Norms, rotary position encoding, softmax, SwiGLU, and decode
attention use CPU operations. Every model projection uses the NPU; allocation
or submission errors fail the run.

Supported checkpoint: the version-2 FP16 container for `Qwen/Qwen3-0.6B`,
revision `c1899de289a04d12100db370d81485cdf75e47ca`, context 512. The local
prepared artifact is `/home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16`.
`fp16-manifest.json` records the container and all effective FP16 tensor hashes.
This executable requires that FP16 container; the normal export tool produces
a different Q8_0 format.

## Build and generate

Requires GCC with OpenMP, libdrm development headers, and the register headers
from the sibling `~/npu/include` checkout. The Python prompt helper uses the
installed `tokenizers` package and the checkpoint's `tokenizer.json`.

```bash
cd /home/orangepi/qwen3.c
make fp16
python3 fp16/generate.py /home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16 \
  --prompt 'Hello' --new-tokens 32
```

The helper uses greedy sampling, suppresses EOS token 151645, starts each run
with an empty logical KV history, and excludes loading and two complete
warmups from the printed timings. It does not fix board clocks. The output is
a raw completion; supply explicit chat template tokens in the prompt when
needed.

The C executable also accepts an input file containing space-separated token
IDs, a generated token count, and a measured run count:

```bash
OMP_NUM_THREADS=4 GOMP_SPINCOUNT=1000 WARMUP_RUNS=2 \
NPU_CORES=3 NPU_FUSED=1 NPU_DOMAIN_ID=1 NPU_ATTENTION=1 \
NPU_SPLIT_DOWN=1 NPU_STREAM_FFN=1 NPU_CLS_TILE=8192 \
taskset -c 4-7 ./runq-fp16 /home/orangepi/qwen3-bench/matched/Qwen3-0.6B.fp16 \
  /home/orangepi/qwen3-bench/matched/results/prompt128.tokens 32 3
```

Run NPU processes serially, including initialization. All custom submissions,
allocation, and DMA synchronization are serial within this engine. CPU packing
may use OpenMP. IOMMU domain 1 is used for both comparison engines because the
board's domain 0 accumulated allocation failures during earlier experiments.

## Fair comparison and numerical limits

The comparison workspace is `/home/orangepi/qwen3-bench/matched`. It contains
the same pinned source exported through the official RKLLM toolkit with
`do_quantization=False`, a three-core stock artifact, tensor audits, GDB
register captures, clock samples, and raw benchmark logs. Stock's runtime
confirms `model_dtype: FP16` and `npu_core_num: 3`. Weight tensors match exactly;
intermediate arithmetic and reduction order can differ.

The current Hello CPU-reference diagnostic passes the unchanged `1e-4`
relative RMSE limit (observed `4.6864e-5`). The older failed diagnostic is
preserved. Read numerical errors and generated token comparisons with the
timings; token agreement on a finite prompt set is not a guarantee for every
prompt. See [ROOFLINE.md](ROOFLINE.md) for the current register optimizations,
bandwidth measurements, matched stock results and GPU comparison. The earlier
matched baseline remains in [RESULTS.md](RESULTS.md).

The [complete CPU/NPU profile](PROFILE.md) includes rendered roofline and
timing PNG/JPG/SVG, full streamed-FFN coverage, driver timing limitations,
and checked optimization candidates.

The [CPU / Mali GPU / NPU experiment](HYBRID.md) uses the same FP16 checkpoint
and direct NPU backend. GPU decode attention matches tested tokens but is
11.7–15.1% slower; GPU prefill fails the unchanged numerical gate. The selected
CPU+NPU runner is retained. PNG/JPG/SVG comparisons and GPU event costs are included.

See the comparison workspace's README for the current final results and
reproduction command. Earlier CPU Q8_0 versus stock W8A8 numbers were withdrawn
as an unmatched comparison.
