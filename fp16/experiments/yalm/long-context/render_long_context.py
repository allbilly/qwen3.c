"""Render the independently audited long-context screening measurements."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--partial', action='store_true', help='Render missing measurements explicitly instead of requiring all 18 rows')
p.add_argument('--thermal-observations', action='store_true', help='Include numerical passes with documented clock drops; label the two protocols')
p.add_argument('--restricted-recovery', action='store_true', help='Document recovered GPU requests, sample-only CPU runs and missing device access')
args = p.parse_args()
audit = json.loads(args.audit.read_text())
assert audit['cpu_target_khz'] == 1800000
assert not args.restricted_recovery or args.thermal_observations
assert args.thermal_observations or audit['cooldown_target_c'] == 55
routes = ['cpu', 'gpu', 'npu', 'cpu_npu_dec', 'gpu_npu_dec', 'all_dec']
names = dict(zip(routes, ['CPU only', 'GPU OpenCL', 'NPU', 'CPU+NPU', 'GPU+NPU', 'CPU+GPU+NPU']))
prompts = [1024, 2048, 4096]
records = audit['records']
row = {(r['prompt'], r['route']): r for r in records}
expected = {(t, r) for t in prompts for r in routes}
assert len(row) == len(records) and row.keys() <= expected
assert args.partial or row.keys() == expected, 'Incomplete measurements require --partial'
complete_prompts = {t for t in prompts if all((t, route) in row for route in routes)}
qualified = lambda r: r['quality_passed'] and (args.thermal_observations or r['clocks_held'])
metrics = [('prefill_ms', False), ('prefill_tps', True), ('decode_tps', True), ('request_ms', False)]
best = {}
for t in prompts:
    if t not in complete_prompts:
        continue
    eligible = [r for r in records if r['prompt'] == t and qualified(r)]
    for key, maximize in metrics:
        if eligible:
            best[t, key] = (max if maximize else min)(r[key] for r in eligible)

def value(r, key):
    text = f"{r[key]:.2f}"
    return '**' + text + '**\\*' if qualified(r) and r[key] == best.get((r['prompt'], key)) else text

lines = [
    '# 1K / 2K / 4K prompt benchmark', '',
    f"{len(records)}/18 placements were measured. {sum(r['quality_passed'] for r in records)}/{len(records)} pass the unchanged numerical gate; {sum(r['quality_passed'] and r['clocks_held'] for r in records)}/{len(records)} also pass the exact sampled-clock check.", '',
    ('The user permits continuing with documented clock drops. The original 55°C sweep stopped during CPU 2K; its 12 measured rows are preserved. Continuation rows use the same model, executable, thread count and measured request protocol, with software cooldown disabled. Actual frequencies and clock-control modes are recorded below. Kernel thermal protection remains active. These are observed bare-board results from two starting-temperature protocols, not a fixed-clock causal speedup comparison.' if args.thermal_observations else 'The bare-board sweep stopped normally at the cooling guard during CPU 2K: after the first warmup and 180 seconds of cooling, the board was still 55.461°C, above the common 55°C start limit. No CPU 2K measured request or subsequent non-NPU 4K job ran. The CPU 1K row also failed the strict clock check: its active CPU stayed at 1.8GHz, but the idle NPU/GPU clocks dropped. All clock governors/limits and the original fan command were restored. The full comparison remains incomplete. The user permits continuing with documented clock drops; the six missing rows resume in fresh directories without forced cooldown. Numerical checks remain unchanged, and the two starting protocols will be labeled separately.'), '',
    ('The continuation process handle is unavailable. GPU 4K has four complete raw requests and passing numerical/device/timer checks, but no parent summary or restoration artifacts. The later managed session exposes no DRM/Mali nodes: the known add check fails on missing /dev/dri/card1. CPU 2K/4K run with the same frozen executable and passive clock sampling, using existing limits without changing sysfs or fan settings. The earlier CPU/NPU/DDR limits and fan command remain locked; successful restoration is not claimed. The three mixed-device 4K rows still need a session exposing board devices.' if args.restricted_recovery else ''), '',
    'The aborted CPU 2K warmup matched all generated IDs but had first-logit relative RMSE 0.00111339, above the unchanged 0.001 gate. It is retained as a failed diagnostic, with zero measured requests. Installing cooling does not establish numerical correctness; investigate this difference before accepting a future CPU 2K row.', '',
    '## Same-model screening comparison', '',
    '| Input tokens | Placement | Prefill ms ↓ | Prefill tokens/s ↑ | Decode tokens/s ↑ | Request ms ↓ | Audit | Start condition |',
    '|---|---|---:|---:|---:|---:|---|---|'
]
for t in prompts:
    for route in routes:
        r = row.get((t, route))
        if r is None:
            status = 'not run: device nodes hidden' if args.restricted_recovery else ('not measured: cooldown abort' if (t, route) == (2048, 'cpu') else 'not run')
            lines.append(f'| {t} | {names[route]} | — | — | — | — | {status} | — |')
            continue
        status = 'pass' if r['quality_passed'] else 'quality fail'
        if not r['clocks_held']:status += '; target clocks differ' if r.get('clock_control')=='sample_only' else '; clock drop'
        if r.get('clock_restoration_verified') is False and r.get('clock_control')!='sample_only':status += '; restoration unverified'
        start = '<=55°C' if r.get('cooldown_enforced', True) else 'no forced cooldown'
        lines.append('| ' + ' | '.join([str(t), names[route], *(value(r, k) for k, _ in metrics), status, start]) + ' |')
lines += [
    '', ('\\* Best observed value among numerical passes, only where all six placements were measured. Clock drops are permitted and explicitly recorded. Asterisks do not denote a controlled fixed-clock winner or statistical significance. Numerical failures remain diagnostics and cannot win.' if args.thermal_observations else '\\* Best observed **qualified** value where all six placements were measured. Two measured requests per row constitute a screen; asterisks do not denote statistical significance. Failed rows are diagnostics and cannot win. Missing cells have no extrapolated timing.'), '',
    'Mixed rows split FFN gate/up channels during decoding only (CPU 96 and/or GPU 96 channels, with the remaining channels on NPU). Their prefill follows the same NPU path. Small prefill differences among these rows are run variation. CPU host operations remain present in every accelerated placement.', '',
    'CPU only runs projections and attention on CPU with no NPU/GPU execution. GPU OpenCL runs both projections and attention on Mali using the custom kernels, with CPU embedding, normalization, RoPE, SwiGLU, residuals and sampling. The NPU route offloads dense matrices and prompt QK/PV to native registers, with CPU decode attention. These GPU results do not represent a tuned, GPU-resident llama.cpp baseline.', '',
    '![Long-context phase and request comparison](long-context-comparison.png)', '',
    '[PNG](long-context-comparison.png) · [JPG](long-context-comparison.jpg) · [SVG](long-context-comparison.svg)', '',
    '## Protocol and correctness', '',
    f"The device tree identifies Orange Pi 5 / RK3588S, and the user reports no heatsink installed. These are current bare-board measurements, with sampled temperatures reaching {max(r.get('row_temperature_range_c', [0, 85.888])[1] for r in records):.3f}°C. PWM commands do not establish physical fan presence or cooling effectiveness. A heatsink with a fan is recommended; [52Pi EP-0167](https://wiki.52pi.com/index.php?title=EP-0167) is specified by its manufacturer for the original Orange Pi 5. A future cooler installation needs a separate benchmark session.", '',
    'All rows use the same pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` and shared FP16 container `c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879`. Effective weight tensors and weight quantization match. Intermediate precision and reduction order can differ: CPU prompt attention uses FP32, whereas NPU/Mali prompt attention uses FP16 operands. These are checked by the unchanged logit gate. The original 512-token container header is unchanged; the isolated runner expands runtime KV/scratch capacity to 4128. The model configuration permits 40960 positions with RoPE base 1e6. The selected executable `100716de...` remains unchanged.', '',
    ('Original-session requests start at <=55°C; continuation requests have no forced cooldown. ' if args.thermal_observations else f"Each measured request starts at <={audit['cooldown_target_c']}°C. ") + 'CPU/NPU/Mali/DDR targets are 1.800/1.000/1.000/2.112GHz, with four inference threads on CPU4–7. Clocks are sampled every 250ms. The original 12-row session restored governors/limits and the fan command. Its PWM255 feedback setpoint was reasserted when the kernel temperature notifier changed it, with 10ms polling on CPU0–3. Raw PWM samples, all reassertions and maximum polling gap are retained. ' + ('The interrupted continuation has no verified restoration. Later CPU-only jobs sample existing settings without sysfs or fan writes. ' if args.restricted_recovery else '') + 'Kernel thermal protection remains active.', '',
    'Two complete warmups precede two measured requests per measured row. Each request has 32 output tokens and 31 decode steps. Prefill tokens/s is input tokens divided by the measured span through the first classifier/argmax. Decode tokens/s is 31 divided by the decode span. Request time is independently measured from prefill start through the last argmax. Activation packing, memory movement, DMA synchronization, host work, queue waits and handoffs are included. Loading, tokenization, ID-file reading, cooldown and warmups are excluded.', '',
    ('CPU 2K/4K in the restricted session use passive sampling of existing governors/limits. The comparison targets are not reapplied, and the GPU uses its existing ondemand governor. An idle GPU at 300MHz in those CPU-only rows is a target mismatch, not evidence of active GPU throttling. Before/after snapshots record whether limits stayed equal; the CPU harness does not write them. Those snapshots do not establish restoration of the earlier interrupted session.' if args.restricted_recovery else ''), '',
    'All routes use identical input IDs and identical teacher IDs during decoding. The NPU references run without teacher forcing, and every actual prediction from all four requests is retained and checked. Full-vocabulary first logits must be finite and have relative RMSE<0.001 against the checked long NPU reference. This reference is not an independent Hugging Face full-model oracle. A failed comparison identifies disagreement with that reference; it does not establish which route is closer to an independent oracle.', '',
    'The [wide QK/PV checks](long-context/wide-attention-check.jsonl) compare 514048 outputs against double accumulation of identical FP16 operands. Maximum relative RMSE 2.02905781962e-7 passes the unchanged 1e-5 gate. All six short regressions pass, and [all 151936 first logits are bit-identical to the prior corresponding routes](long-context/long-context-short-parity.json). Native PV submissions retain the four-data-bank limit and use 64/32/16 query rows at 1K/2K/4K.', '',
    'The [immutable prompts](long-context/prompts/provenance.json) repeat the existing 128-token benchmark prompt and take nested prefixes of exactly 1024/2048/4096 tokens. They measure synthetic throughput, rather than natural long-document task quality.', '',
    '## Raw measured ranges and checks', '',
    '| Input | Placement | Prefill min–max ms | Decode min–max tokens/s | Request min–max ms | First-logit relative RMSE | All actual IDs match |',
    '|---|---|---:|---:|---:|---:|---|'
]
span = lambda v: f'{v[0]:.3f}–{v[1]:.3f}'
for t in prompts:
    for route in routes:
        r = row.get((t, route))
        if r is None:
            continue
        lines.append('| ' + ' | '.join([str(t), names[route], span(r['prefill_range']), span(r['decode_range']), span(r['request_range']), f"{r['relative_rmse']:.8g}", str(r['all_predictions_match'])]) + ' |')
lines += [
    '', 'Min–max spans contain only two observations; they are not confidence intervals. The run order is retained in the execution plan. No counterbalanced long-context speedup or fresh stock-RKLLM comparison is claimed.', '',
    '## Phase findings and next checks', '',
    'In the second measured CPU 1K request, prompt attention took 52939.918ms of 63803.618ms prefill (about 83%). Its linear calls took 10405.982ms, including 162.843ms packing. CPU prompt attention is the main measured bottleneck in that row. At 1K the mixed placements keep the same NPU prefill path, and their small differences overlap the measured NPU range; this screen does not establish a repeatable mixed-device gain.', '',
    'The resident request span includes packing, transfers, synchronization and host work. It is separately timed; a faster individual phase is insufficient evidence of an end-to-end gain. Existing [transfer and compute profiles](COSTS.md) and [roofline measurements](../../ROOFLINE.md) remain separate artifacts. Wall spans and logical byte counts do not measure physical DDR traffic. The GPU baseline uses custom OpenCL kernels and is not a tuned GPU-resident llama.cpp baseline.', '',
    ('Finish the remaining mixed-device 4K rows once board devices are exposed, preserving these recorded sessions. A later cooled comparison would use fresh outputs and all placements under one protocol. Numerical checks remain unchanged, and neither passive clock sampling nor cooling establishes that the CPU logit checks pass.' if args.restricted_recovery else 'After installing cooling, create a fresh output namespace and rerun all six placements at each prompt length, including new NPU references. Keep the same numerical gate and measured request protocol. Preserve the current bare-board session; do not combine it with cooled results. Check the CPU 2K logit difference, stable clock samples and restoration before comparing winners.'), '',
    '## Reproducibility', '',
    f"[Independent {len(records)}-row audit](long-context/{'long-context-observed-audit.json' if args.thermal_observations else 'long-context-audit.json'}) · [original execution plan](long-context/long-context-execution-plan-screen55-cpu1800-fanheld.json) · [original cooling-control audit](long-context/long-context-cooling-restoration.json) · [aborted warmup audit](long-context/long-context-cpu2048-abort-audit.json) · [source/checklist](long-context/long-context-plan.md)", '',
    'The first 51°C attempt stopped normally at the bounded cooldown guard before the second 4K warmup; GPU clock drops were also observed. A subsequent 55°C attempt showed that a one-time fan setting is overridden by the kernel. These attempts are retained as rejected diagnostics and excluded from this table. The later 2.256GHz CPU 1K row also thermally dropped to 2.208GHz; this partial comparison uses a common 1.800GHz CPU target for all measured rows. The max-clock NPU/GPU timings remain separate diagnostics. The [board kernel source](https://github.com/orangepi-xunlong/linux-orangepi/blob/orange-pi-6.1-rk35xx/drivers/hwmon/pwm-fan.c) implements that temperature notifier.', '',
    'The exported `long-context/build.py` reproduces isolated runner `f8b4f9eb...`. Full source, model, binary, input, raw-log and logit hashes are recorded in provenance/audits. Small raw evidence is exported losslessly; original checkpoint, executables and binary logits remain in `/home/orangepi/qwen3-bench/matched/roofline/yalm/e2e/long-context`. ' + ('The passive CPU outputs remain in `/tmp/qwen3-long-context-cpu-sample-only-20261006`; the new CPU and recovered GPU first-logit files are also exported losslessly as `.f32.gz` alongside raw logs and clock snapshots. Preserve the original output directories to rerun the full audit directly. ' if args.restricted_recovery else '') + 'All NPU initialization/sync/submits ran serially. No push was requested.', '',
    'INTENT: the long-context harness aborts on cooldown timeout and marks clock drops as failed screens; the user permits continuing with documented clock drops; fp16/experiments/yalm/LONG-CONTEXT.md currently requires <=55°C starts and strict clock checks.', '',
    'AUTH: user said "wip commit per milestone".', ''
]
cpu_profile_rows=[row[t,'cpu'] for t in prompts if (t,'cpu') in row and row[t,'cpu'].get('cpu_phase_profiles')]
if cpu_profile_rows:
    cpu_profile_lines=['', '## CPU phase profile', '',
        'Medians of the two measured requests, in milliseconds. Packing is included in linear wall time. These host timer spans do not measure physical DDR bandwidth. Failed numerical rows remain diagnostics.', '',
        '| Input | Prefill linear wall | Included pack | Linear compute | Prompt attention | Attention / prefill | Decode linear wall | Included pack | Linear compute |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in cpu_profile_rows:
        pre=r['cpu_phase_profiles']['prefill'];dec=r['cpu_phase_profiles']['decode']
        values=[str(r['prompt']), *(f'{pre[k]:.3f}' for k in ['wall_ms','pack_ms','compute_ms','prefill_attention_ms']),
                f"{100*pre['prefill_attention_ms']/r['prefill_ms']:.2f}%", *(f'{dec[k]:.3f}' for k in ['wall_ms','pack_ms','compute_ms'])]
        cpu_profile_lines.append('| '+' | '.join(values)+' |')
    cpu_profile_lines += ['', 'The attention percentage is the ratio of median attention time to median prefill time. Decode attention and other CPU host operations are included in the request/decode spans, outside the reported linear timer.', '']
    index=lines.index('## Reproducibility')
    lines[index:index]=cpu_profile_lines
args.out.mkdir(parents=True, exist_ok=True)
if args.thermal_observations:
    thermal = ['', '## Recorded clocks and temperatures', '',
               'Ranges cover each complete job, including initialization, two warmups, two measured requests and any original-session cooldown. They are not isolated measured-phase frequency averages. Differences include thermal clamps and idle accelerators using ondemand scaling in sample-only CPU runs.', '',
               '| Input | Placement | CPU4 MHz | CPU6 MHz | NPU MHz | GPU MHz | DDR MHz | Temperature °C | Dropped samples / total |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in prompts:
        for route in routes:
            r = row.get((t, route))
            if r is None:continue
            ranges = r['clock_frequency_ranges']
            clocks = [span([v / (1000 if key.startswith('cpu') else 1000000) for v in ranges[key]]) for key in ['cpu4_khz', 'cpu6_khz', 'npu_hz', 'gpu_hz', 'ddr_hz']]
            thermal.append('| ' + ' | '.join([str(t), names[route], *clocks, span(r['row_temperature_range_c']), f"{r['row_clock_drops']}/{r['row_clock_samples']}"]) + ' |')
    thermal += ['', '[Continuation plan](long-context/long-context-execution-plan-continue-throttle-cpu1800-fanheld.json) · ' + ('[session recovery observations](long-context/long-context-session-recovery.json) · [recovered GPU audit](long-context/long-context-gpu4096-recovered-audit.json) · [CPU passive-sampling plan](long-context/long-context-cpu-sample-plan.json)' if args.restricted_recovery else '[continuation restoration audit](long-context/long-context-continuation-restoration.json)') + ' · [continuation checklist](long-context/long-context-continue-plan.md)', '']
    lines[-4:-4] = thermal
(args.out/'LONG-CONTEXT.md').write_text('\n'.join(lines))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
panels = [('prefill_tps', 'Prefill tokens/s', False), ('decode_tps', 'Decode tokens/s', False), ('prefill_ms', 'Prefill latency (s)', True), ('request_ms', 'Complete resident request (s)', True)]
for ax, (key, title, seconds) in zip(axes.flat, panels):
    for route in routes:
        values = [row[t, route][key]/(1000 if seconds else 1) if (t, route) in row else float('nan') for t in prompts]
        line, = ax.plot(prompts, values, marker='o', label=names[route])
        for t, val in zip(prompts, values):
            r = row.get((t, route))
            if r is None:
                continue
            if not r['quality_passed'] or not r['clocks_held']: ax.plot(t, val, marker='x', markersize=10, color=line.get_color())
            if qualified(r) and r[key] == best.get((t, key)): ax.annotate('*', (t, val), xytext=(4, 4), textcoords='offset points')
    ax.set_xticks(prompts, ['1K', '2K', '4K'])
    if args.restricted_recovery and seconds:
        ax.set_yscale('log')
        title += ' · log scale'
    ax.set_xlabel('Input tokens'); ax.set_ylabel(title); ax.grid(alpha=.25)
axes[0, 0].legend(fontsize=8)
fig.suptitle(f'Orange Pi 5 · Qwen3-0.6B FP16 · bare board · {len(records)}/18 measured placements\n' + ('* best numerical pass in complete groups; × clock mismatch or quality failure' if args.thermal_observations else '* best qualified screen value; × failed clock audit; missing points unmeasured'))
for ext in ['png', 'jpg', 'svg']:
    target = args.out / f'long-context-comparison.{ext}'
    fig.savefig(target, dpi=180)
    if ext == 'svg':
        # Normalize matplotlib path-line whitespace before hashing/exporting.
        target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
plt.close(fig)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
artifacts = [args.out / 'LONG-CONTEXT.md', *(args.out / f'long-context-comparison.{ext}' for ext in ['png', 'jpg', 'svg'])]
provenance = {
    'audit_sha256': sha(args.audit), 'render_source_sha256': sha(Path(__file__)),
    'model_sha256': audit['model_sha256'], 'binary_sha256': audit['binary_sha256'],
    'partial': args.partial, 'thermal_observations': args.thermal_observations, 'restricted_recovery':args.restricted_recovery, 'measured_rows': len(records),
    'missing_placements': [{'prompt': t, 'route': r} for t, r in sorted(expected - row.keys())],
    'asterisk_prompt_lengths': sorted(complete_prompts),
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['matplotlib', 'numpy', 'pillow']},
    'outputs': {path.name: sha(path) for path in artifacts},
    'scope': 'Observed medians only; missing measurements remain gaps. Stars require all six placements measured at that prompt length.'
}
(args.out / 'long-context-render-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(f'Rendered {len(records)} measured rows, {18-len(records)} missing rows and PNG/JPG/SVG charts')
