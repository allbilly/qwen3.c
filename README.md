## qwen3.c

<p align="center">
  <img src="assets/qwen3_c.jpg" width="300" height="300" alt="Cute Llama">
</p>

**Run inference for frontier models based on the Qwen3 architecture, like Qwen3-4B or DeepSeek-R1-0528-Qwen3-8B, on your local Linux/macOS/Windows machine. No complicated configuration
required, just follow the steps below and enjoy.**

**Understand the basics of transformers but want to learn in-depth how LLM inference works? qwen3.c runs LLMs using one easy-to-understand (relatively speaking!) file of C source with no dependencies. Once you've
digested it and understand the data flow, you're there.**

This project's starting point was Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), which does single-file
inference for LLaMA 2-compatible models. The LLaMA 2 architecture is now 2 years old (a lifetime in the field of AI) and is
long superseded. This project aims to maintain the simplicity of llama2.c while supporting a frontier
model architecture, with the goal of being both an up-to-date learning resource and also a great way to run the latest models locally.

Despite being only around 1000 lines of C code with no dependencies, qwen3.c supports everything you need to
enjoy running leading Qwen3-architecture LLMs on standard hardware (no GPUs needed), including multi-CPU core operation, support for Unicode/multi-language input and output, and thinking/reasoning models.

qwen3.c includes a Python tool to process any Qwen3-architecture HuggingFace model, converting to qwen3.c's model format which uses Q8_0 quantization for a good trade-off between quality
and performance.

## Step 1: checkout and build

For the tested RK3588 direct-register FP16 engine and the comparison against
stock RKLLM using the same model and precision, see [fp16/README.md](fp16/README.md).
Current bandwidth, register optimizations and GPU comparison are in
[fp16/ROOFLINE.md](fp16/ROOFLINE.md).

First, checkout this repo and build it. I recommend the OpenMP version if your toolchain supports it, as it supports multiple CPU
cores for dramatically improved performance:

```aiignore
git clone https://github.com/adriancable/qwen3.c
cd qwen3.c
make openmp
```

(To build without OpenMP, just run `make` without the `openmp` argument.)

## Apple M1 ANE on Asahi Linux

The native C ANE backend uses the driver ABI and verified linear primitives
from `~/ane`. It supports **base M1 (T8103)** with the `ane` driver exposed at
`/dev/accel/accel*`. Install the driver following `~/ane/kmod/README.md`.
Other Apple chips require their own verified register streams.

```bash
make ane cpu
OMP_NUM_THREADS=4 ANE=1 ./runq-ane Qwen3-0.6B.bin -c 512 -i 'Explain gravity.' -n 32 -t 0
```

The existing version-1 Q8 checkpoints work unchanged. Prompts use batches of
up to 32 tokens with resident FP16 projection weights on ANE. Batches shorter
than 16 tokens and single-token decode use CPU Q8, which is faster on this M1.
Attention, normalization, sampling, and the vocabulary head also run on CPU.
`ANE_DECODE=1` additionally offloads decode projections, using the same plans
and weights. `ANE_SERIAL_PREFILL=1` disables prompt batching for comparisons.
FP16 activations can change logits and greedy choices near ties.
`ANE=0 ./runq-ane ...` and `./runq-cpu ...` select CPU.
An explicitly requested ANE backend fails if the device or plans cannot be
prepared. `QWEN3_MATMUL_VERBOSE=1` enables per-operation diagnostics.

For a local Hugging Face safetensors snapshot, a small exporter avoids loading
a full PyTorch model into memory:

```bash
python -m pip install numpy jinja2
python ane/export_checkpoint.py /path/to/Qwen3-0.6B Qwen3-0.6B.bin --context 512
```

Run correctness checks and benchmarks through the shared queue:

```bash
make ane-test
python3 tools/benchmark_queue.py -- env OMP_NUM_THREADS=4 ANE=1 ./runq-ane Qwen3-0.6B.bin -c 512 -i 'Explain gravity.' -n 32 -t 0
```

The queue waits on `~/ane.lock` and `~/gpu.lock`, then checks for live
benchmark processes and processes holding ANE devices. Use the same wrapper
in other Codex sessions. Performance measurements should keep model, prompt,
thread count, CPU affinity, warmups, and generated-token count fixed.

Build and run the paired full-model benchmark (requires NumPy):

```bash
make runq-ane-bench
python3 tools/paired_ane.py Qwen3-0.6B.bin --output /tmp/ane-results.json
python3 tests/test_benchmark_queue.py -v
```

The paired runner queues itself, alternates CPU/ANE trials, uses four threads
on CPUs 4–7, and compares every vocabulary logit on fixed decode inputs.
Prompt token IDs, checkpoint SHA-256, individual trials, timing ranges, and
accuracy checks are saved in its JSON output.

On this base M1 with Qwen3-0.6B Q8, four threads on CPUs 4–7 and five queued
paired trials gave these medians after one full warmup per process. The
default hybrid mode uses ANE prefill and CPU Q8 decode; its five-token prompt
also uses CPU prefill.

| Prompt tokens | CPU prefill (ms) | Hybrid prefill (ms) | Prefill speedup | CPU decode (tokens/s) | Hybrid decode (tokens/s) |
| --- | --- | --- | --- | --- | --- |
| 5 | 65.1 | 65.4 (CPU) | 1.00× | 59.1 | 56.8 |
| 16 | 202.1 | 102.7 | 1.97× | 58.7 | 57.2 |
| 32 | 397.8 | 184.8 | 2.15× | 58.9 | 56.4 |
| 65 | 843.5 | 355.2 | 2.37× | 55.0 | 56.7 |

For a 32-token prompt and 16 output tokens, the decode routing comparison is:

| Mode | Trials | Prefill (ms) | Decode (tokens/s) | Prompt + 16 output tokens (ms) |
| --- | --- | --- | --- | --- |
| CPU Q8 | 5 | 397.8 | 58.9 | 649.7 |
| ANE prefill + CPU decode (default) | 5 | 184.8 | 56.4 | 450.9 |
| ANE prefill + ANE decode (`ANE_DECODE=1`) | 3 | 176.5 | 24.4 | 783.5 |

With 16 output tokens, prompt plus decode was 1.26–1.80× faster in hybrid mode
for the 16–65-token prompts. Initialization is excluded from both tables:
resident weight preparation took about 0.5–0.6 seconds, versus
0.15–0.23 seconds for CPU loading. CPU can be faster for a single short request.
These are shared-desktop measurements; the raw ranges include occasional
background-load outliers. See [the full results](ane/benchmarks/m1-qwen3-0.6b.json).

Minimum CPU/ANE logit cosine similarity was 0.997055, with 63 of 64 matching
top-1 choices on identical decode inputs. The 32-token prompt's first choice
changed near a tie. `ANE_DECODE=1` measured about 25 tokens/s; see
[the ANE decode comparison](ane/benchmarks/m1-qwen3-0.6b-ane-decode.json).

### Other M1 GPU measurements: GPT-2 124M

The sibling `~/applegpu` project now has verified full GPT-2 inference through
omarchy-mlx Vulkan and tinygrad OpenCL. Its 2026-10-05 prefill/decode results
are retained here as GPU context. These use **GPT-2 124M FP32**, batch 1,
context capacity 1024, stock Mesa 26.2.3 and 64 cached decode calls. The Qwen3
tables above use Qwen3-0.6B Q8 with different prompts and output lengths;
these workloads do not establish a GPU-versus-ANE speedup for Qwen3.

Each entry is the median of three fresh workers with backend order rotated
between rounds. Prefill consumes the entire prompt and returns the first
next-token choice. Both phases include completed GPU execution, GPU argmax
and a blocking selected-token read. Loading, compilation, BEAM tuning and
full-logit comparison are excluded.

| GPT-2 FP32 backend | Prompt tokens | Prefill ms | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| omarchy-mlx Vulkan | 32 | 63.55 | 503.53 | 23.05 |
| omarchy-mlx Vulkan | 128 | 175.95 | 727.46 | 22.85 |
| omarchy-mlx Vulkan | 256 | 329.01 | 778.09 | 21.19 |
| tinygrad OpenCL, BEAM=0 | 32 | 47.09 | 679.50 | 24.21 |
| tinygrad OpenCL, BEAM=0 | 128 | 113.95 | 1123.31 | 22.58 |
| tinygrad OpenCL, BEAM=0 | 256 | 190.09 | 1346.70 | 22.62 |
| tinygrad OpenCL, BEAM=2 | 32 | 585.86 | 54.62 | 16.71 |
| tinygrad OpenCL, BEAM=2 | 128 | 1920.51 | 66.65 | 16.68 |
| tinygrad OpenCL, BEAM=2 | 256 | 2147.73 | 119.20 | 16.29 |

All 1,755 full-logit checks and 1,755 timed token checks passed. The import
recomputed the medians and checked the original report's audit hash.
[Imported results, runtime versions, source hashes and validation](ane/benchmarks/applegpu-gpt2-m1.json)
are retained. The original receipts are in
`~/applegpu/gpt2/results/prefill-decode-float32.json` and its validation file;
that project's GPT-2 feature is still uncommitted. Clocks were not fixed and
background CPU activity was present. Untuned tinygrad improved longer-prompt
prefill in these runs; BEAM=2 regressed with this build's default estimate
setting. A matched Qwen3 GPU benchmark is still needed for a Qwen3 comparison.

## Step 2: download and convert a model

Install any needed Python dependencies for the HuggingFace export utility:

```aiignore
pip install -r requirements.txt
```

Then, pick any dense (no Mixture-of-Experts) unquantized (not GGUF) Qwen3-architecture model from HuggingFace.
Unless you have lots of RAM, start with smaller models. `Qwen/Qwen3-4B` is great, so we'll start with that.

Run the Python 3 export tool (will take around 10 minutes) to download the model from HuggingFace and convert to qwen3.c's quantized checkpoint format, storing in
a file called `Qwen3-4B.bin`:

```aiignore
python export.py Qwen3-4B.bin Qwen/Qwen3-4B
```

## Step 3: run and enjoy

```aiignore
./runq Qwen3-4B.bin
```

Fun things you can try asking:

> Tell me a surprising fact about an animal of your choice.

> Write a short story for a 5 year old girl, featuring Sobieski the dog and Pepe the cat.

> Write a C program which sorts a list using the bubble sort algorithm.

> Write a poem about a little boy who builds a rocket to fly to the moon. In Japanese, please.

> Translate into English: 我希望您喜欢使用 qwen3.c 学习 LLM。

## Step 4: experiment with reasoning mode

qwen3.c also supports reasoning/thinking, if the model used supports it. Enable thinking with the `-r 1` command line parameter:

```aiignore
./runq Qwen3-4B.bin -r 1
```

Then try:

> Solve the quadratic equation x^2 - 5x + 6 = 0.

> What is 19673261 * 1842.64?

## Step 5: explore other models

Try for example `DeepSeek-R1-0528-Qwen3-8B`:

```aiignore
python export.py DeepSeek-R1-0528-Qwen3-8B.bin deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

Then:

```aiignore
./runq DeepSeek-R1-0528-Qwen3-8B.bin
```

## Advanced options

qwen3.c lets you configure model settings via the command line including setting a system prompt, setting temperature, sampling parameters and so forth.
To show available settings, run qwen3.c without any command-line parameters:

```aiignore
./runq
```

## License

MIT
