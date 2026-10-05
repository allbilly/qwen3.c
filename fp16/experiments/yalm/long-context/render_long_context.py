"""Render the independently audited long-context screening measurements."""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
audit = json.loads(args.audit.read_text())
routes = ['cpu', 'gpu', 'npu', 'cpu_npu_dec', 'gpu_npu_dec', 'all_dec']
names = dict(zip(routes, ['CPU only', 'GPU OpenCL', 'NPU', 'CPU+NPU', 'GPU+NPU', 'CPU+GPU+NPU']))
prompts = [1024, 2048, 4096]
records = audit['records']
assert len(records) == 18
assert {(r['prompt'], r['route']) for r in records} == {(t, r) for t in prompts for r in routes}
row = {(r['prompt'], r['route']): r for r in records}
qualified = lambda r: r['quality_passed'] and r['clocks_held']
metrics = [('prefill_ms', False), ('prefill_tps', True), ('decode_tps', True), ('request_ms', False)]
best = {}
for t in prompts:
    eligible = [r for r in records if r['prompt'] == t and qualified(r)]
    for key, maximize in metrics:
        if eligible:
            best[t, key] = (max if maximize else min)(r[key] for r in eligible)

def value(r, key):
    text = f"{r[key]:.2f}"
    return '**' + text + '**\\*' if qualified(r) and r[key] == best.get((r['prompt'], key)) else text

lines = [
    '# 1K / 2K / 4K prompt benchmark', '',
    f"All 18 placements were measured. {sum(qualified(r) for r in records)}/18 pass both the unchanged numerical gate and the exact sampled-clock check.", '',
    '## Same-model screening comparison', '',
    '| Input tokens | Placement | Prefill ms ↓ | Prefill tokens/s ↑ | Decode tokens/s ↑ | Request ms ↓ | Audit |',
    '|---|---|---:|---:|---:|---:|---|'
]
for t in prompts:
    for route in routes:
        r = row[t, route]
        status = 'pass' if qualified(r) else ('quality fail' if not r['quality_passed'] else 'clock fail')
        lines.append('| ' + ' | '.join([str(t), names[route], *(value(r, k) for k, _ in metrics), status]) + ' |')
lines += [
    '', '\\* Best observed **qualified** value for that prompt length. Two measured requests per row constitute a screen; asterisks do not denote statistical significance. Failed rows are diagnostics and cannot win.', '',
    'Mixed rows split FFN gate/up channels during decoding only (CPU96 and/or GPU96 channels, with the remaining channels on NPU). Their prefill follows the same NPU path. Small prefill differences among these rows are run variation. CPU host operations remain present in every accelerated placement.', '',
    'CPU only runs projections and attention on CPU with no NPU/GPU execution. GPU OpenCL runs both projections and attention on Mali using the custom kernels, with CPU embedding, normalization, RoPE, SwiGLU, residuals and sampling. The NPU route offloads dense matrices and prompt QK/PV to native registers, with CPU decode attention. These GPU results do not represent a tuned, GPU-resident llama.cpp baseline.', '',
    '![Long-context phase and request comparison](long-context-comparison.png)', '',
    '[PNG](long-context-comparison.png) · [JPG](long-context-comparison.jpg) · [SVG](long-context-comparison.svg)', '',
    '## Protocol and correctness', '',
    'All rows use the same pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` and shared FP16 container `c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879`. There is no quantization difference. The original 512-token container header is unchanged; the isolated runner expands runtime KV/scratch capacity to4128. The model configuration permits40960 positions with RoPE base1e6. The selected executable `100716de...` remains unchanged.', '',
    f"Each request starts at <={audit['cooldown_target_c']}°C. CPU/NPU/Mali/DDR targets are2.256/1.000/1.000/2.112GHz, with four inference threads onCPU4–7. Clocks are sampled every250ms and original governors/limits restored. Fan PWM255 is a common feedback setpoint, reasserted when the kernel temperature notifier changes it, with10ms polling onCPU0–3. Raw PWM samples, all reassertions and maximum polling gap are retained. Kernel thermal protection remains active, and the original fan setting is restored afterward.", '',
    'Two complete warmups precede two measured requests per row. Each request has32 output tokens and31 decode steps. Prefill tokens/s is input tokens divided by the measured span through the first classifier/argmax. Decode tokens/s is31 divided by the decode span. Request time is independently measured from prefill start through the last argmax. Activation packing, memory movement, DMA synchronization, host work, queue waits and handoffs are included. Loading, tokenization, ID-file reading, cooldown and warmups are excluded.', '',
    'All routes use identical input IDs and identical teacher IDs during decoding. The NPU references run without teacher forcing, and every actual prediction from all four requests is retained and checked. Full-vocabulary first logits must be finite and have relative RMSE<0.001 against the checked long NPU reference. This reference is not an independent Hugging Face full-model oracle.', '',
    'The [wide QK/PV checks](long-context/wide-attention-check.jsonl) compare514048 outputs against double accumulation of identical FP16 operands. Maximum relative RMSE2.02905781962e-7 passes the unchanged1e-5 gate. All six short regressions pass, and [all151936 first logits are bit-identical to the prior corresponding routes](long-context/long-context-short-parity.json). Native PV submissions retain the four-data-bank limit and use64/32/16 query rows at1K/2K/4K.', '',
    'The [immutable prompts](long-context/prompts/provenance.json) repeat the existing128-token benchmark prompt and take nested prefixes of exactly1024/2048/4096 tokens. They measure synthetic throughput, rather than natural long-document task quality.', '',
    '## Raw measured ranges and checks', '',
    '| Input | Placement | Prefill min–max ms | Decode min–max tokens/s | Request min–max ms | First-logit relative RMSE | All actual IDs match |',
    '|---|---|---:|---:|---:|---:|---|'
]
span = lambda v: f'{v[0]:.3f}–{v[1]:.3f}'
for t in prompts:
    for route in routes:
        r = row[t, route]
        lines.append('| ' + ' | '.join([str(t), names[route], span(r['prefill_range']), span(r['decode_range']), span(r['request_range']), f"{r['relative_rmse']:.8g}", str(r['all_predictions_match'])]) + ' |')
lines += [
    '', 'Min–max spans contain only two observations; they are not confidence intervals. The run order is retained in the execution plan. No counterbalanced long-context speedup or fresh stock-RKLLM comparison is claimed.', '',
    '## Reproducibility', '',
    '[Independent18-row audit](long-context/long-context-audit.json) · [execution plan](long-context/long-context-execution-plan-screen55-fanheld.json) · [cooling-control audit](long-context/long-context-cooling-restoration.json) · [source/checklist](long-context/long-context-plan.md)', '',
    'The first51°C attempt stopped normally at the bounded cooldown guard before the second4K warmup; GPU clock drops were also observed. A subsequent55°C attempt showed that a one-time fan setting is overridden by the kernel. Both attempts are retained as rejected diagnostics and excluded from this table. The [board kernel source](https://github.com/orangepi-xunlong/linux-orangepi/blob/orange-pi-6.1-rk35xx/drivers/hwmon/pwm-fan.c) implements that temperature notifier.', '',
    'The exported `long-context/build.py` reproduces isolated runner `f8b4f9eb...`. Full source, model, binary, input, raw-log and logit hashes are recorded in provenance/audits. Small raw evidence is exported losslessly; checkpoint, executables and binary logits remain in `/home/orangepi/qwen3-bench/matched/roofline/yalm/e2e/long-context`. All NPU initialization/sync/submits ran serially. No push was requested.', '',
    'INTENT: the benchmark runner and attention scratch are limited to 512 tokens; the user requests 1024/2048/4096-token prompts with CPU/OpenCL/NPU combinations; fp16/README.md specifies pinned shared FP16 weights, measured phase/request spans and numerical checks.', '',
    'AUTH: user said "wip commit per milestone".', ''
]
args.out.mkdir(parents=True, exist_ok=True)
(args.out/'LONG-CONTEXT.md').write_text('\n'.join(lines))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
panels = [('prefill_tps', 'Prefill tokens/s', False), ('decode_tps', 'Decode tokens/s', False), ('prefill_ms', 'Prefill latency (s)', True), ('request_ms', 'Complete resident request (s)', True)]
for ax, (key, title, seconds) in zip(axes.flat, panels):
    for route in routes:
        values = [row[t, route][key]/(1000 if seconds else 1) for t in prompts]
        line, = ax.plot(prompts, values, marker='o', label=names[route])
        for t, val in zip(prompts, values):
            r = row[t, route]
            if not qualified(r): ax.plot(t, val, marker='x', markersize=10, color=line.get_color())
            elif r[key] == best.get((t, key)): ax.annotate('*', (t, val), xytext=(4, 4), textcoords='offset points')
    ax.set_xticks(prompts, ['1K', '2K', '4K'])
    ax.set_xlabel('Input tokens'); ax.set_ylabel(title); ax.grid(alpha=.25)
axes[0, 0].legend(fontsize=8)
fig.suptitle('RK3588 · shared Qwen3-0.6B FP16 · 32 outputs\n* best qualified screen value; × failed audit; two measured requests per row')
for ext in ['png', 'jpg', 'svg']:
    fig.savefig(args.out/f'long-context-comparison.{ext}', dpi=180)
plt.close(fig)
print('Rendered18 measured rows and PNG/JPG/SVG charts')
