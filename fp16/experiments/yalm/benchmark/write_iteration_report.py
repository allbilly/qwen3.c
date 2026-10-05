"""Write the audited single-change results without promoting a candidate."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
destination = Path('/home/orangepi/qwen3.c/fp16/experiments/yalm/ITERATIONS.md')
experiments = [
    ('NPU attention batch 128', 'iteration-row128-e2e51-report.json'),
    ('Fused GPU decode attention versus CPU', 'iteration-fused-gpu-vs-cpu-e2e51-report.json'),
]
lines = [
    '# Independent phase and complete-request comparisons', '',
    'Both experiments pass independent model/source/binary, numerical, device-count,',
    'clock and restoration audits. Each uses the shared Qwen3-0.6B FP16 checkpoint,',
    'identical teacher-forced inputs, 32 outputs and every actual predicted ID.',
    'There are two full warmups and two measurements per prompt per block, with',
    'ABBA/BAAB blocks and reversed prompt order in the second half. The tables',
    'recompute medians from all four individual measurements per engine/prompt.',
    'Every request starts at <=51°C. Warmup, initialization and cooling are excluded.',
    '',
    'Complete-request time is independently captured from prefill start through',
    'the last output. Packing, activation transfers, cache preparation, DMA sync,',
    'device/queue waits and interphase work are included. Input-ID file reading and',
    'tokenization are excluded. Effective prefill = input tokens / TTFT, including',
    'classifier and sampling; decode covers 31 subsequent tokens.',
    '',
    '![Audited independent changes](../../yalm-iterations.png)', '',
    '[PNG](../../yalm-iterations.png) · [JPG](../../yalm-iterations.jpg) · [SVG](../../yalm-iterations.svg)',
]
for title, file in experiments:
    report = json.loads((root / file).read_text())
    lines.extend(['', '## ' + title, '',
        '| Input tokens | Engine / decode attention | TTFT ms | Prefill t/s | Decode t/s | Request ms | Request min–max ms |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for row in report['results']:
        for label in ['baseline', 'candidate']:
            engine = row[label]
            median, ranges = engine['median'], engine['ranges']
            description = 'Original / CPU' if label == 'baseline' else ('Batch 128 / CPU' if title.startswith('NPU') else 'Original / fused GPU')
            lines.append(f"| {row['prompt']} | {description} | {median['ttft_ms']:.3f} | {median['effective_prefill_tps']:.2f} | {median['decode_tps']:.3f} | {median['request_ms']:.3f} | {ranges['request_ms'][0]:.3f}–{ranges['request_ms'][1]:.3f} |")
    lines.extend(['', '| Input tokens | Prefill throughput change | Decode throughput change | Complete-request throughput change | Overlapping ranges: prefill / decode / request |',
                  '|---|---:|---:|---:|---|'])
    for row in report['results']:
        overlap = row['observed_ranges_overlap']
        lines.append(f"| {row['prompt']} | {row['prefill_gain_percent']:+.2f}% | {row['decode_gain_percent']:+.2f}% | {row['request_gain_percent']:+.2f}% | " + ' / '.join('yes' if overlap[key] else 'no' for key in ['ttft_ms', 'decode_tps', 'request_ms']) + ' |')
    lines.extend(['', f"[Audited raw aggregation]({file}) records all phase/request ranges, individual observations and audit hashes."])
    if title.startswith('NPU'):
        lines.extend(['',
            'Attention batch 128 improves prefill throughput by 4.71–4.84%, with',
            'non-overlapping observed TTFT ranges. Request median throughput improves',
            'only 0.27% / 2.12%, with overlapping observed request ranges. Decode ranges',
            'also overlap. Four observations are not confidence intervals; a reliable',
            'complete-request improvement is not established and no candidate is promoted.',
            'The existing five-prompt quality and complete primitive checks are in [CANDIDATES.md](CANDIDATES.md).'])
    else:
        changes = [100 * (r['candidate']['median']['request_ms'] / r['baseline']['median']['request_ms'] - 1) for r in report['results']]
        lines.extend(['',
            f'Fused GPU decode attention lowers decode throughput by 6.07% / 5.69% and increases request latency by {changes[0]:.2f}% / {changes[1]:.2f}%.',
            'Decode and request ranges do not overlap. Prefill ranges overlap; both',
            'routes retain original NPU prefill. CPU decode attention remains selected.',
            'This is a direct CPU-versus-fused-GPU comparison, not a matched measurement',
            'of fusion speedup against the older unfused GPU prototype.'])
lines.extend(['', '## Limits and device placement', '',
    'The host transformer still runs embedding, norms, RoPE, SwiGLU, residuals and',
    'sampling on CPU. Dense projections use the native register NPU backend in',
    'both experiments; the GPU candidate replaces decode attention through Mali',
    'OpenCL. No concurrent FFN channel partition across CPU/GPU/NPU is implemented.',
    'The results cover 128/256-token prompts and context 512, with cooled requests',
    'at fixed clocks; they do not establish sustained hot-loop throughput.',
    'The older entire 50°C partial request sweep remains rejected and is not used.',
    'CPU/GPU prefill short-prompt logit failures remain retained; these two',
    'comparisons use qualified NPU prefill and do not resolve those failures.',
    '',
    'Source/binary and audit records are frozen under `evidence/iteration-*`.',
    'The shared model, full vocabulary logits and compiled binaries remain in',
    '`/home/orangepi/qwen3-bench/matched/roofline/yalm` with hashes retained.',
    '',
    'INTENT: the selected runner has NPU projections and a tested GPU-attention route but no full CPU/GPU projection comparison; the user requests device combinations and stepwise profiling/optimization; fp16/README.md specifies the shared FP16 checkpoint, native NPU projection reference and numerical verification.',
    '', 'AUTH: user said "wip commit per milestone".', '',
])
destination.write_text('\n'.join(lines))
print('Wrote', destination)
