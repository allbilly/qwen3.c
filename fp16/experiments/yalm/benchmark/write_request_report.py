"""Render the fresh common-temperature placement data after independent audit."""
from pathlib import Path
import json

root=Path(__file__).resolve().parent
repo=Path('/home/orangepi/qwen3.c/fp16/experiments/yalm')
report=json.loads((root/'request-matrix51-audit.json').read_text())
records=report['records']
assert len(records)==20 and all(r['quality_passed'] for r in records)
order=['cpu','gpu','npu','cpu_gpu','gpu_cpu','cpu_npu','npu_cpu','gpu_npu','npu_gpu','all']
for row in records:
    config=json.loads((root/row['directory']/(row['label']+'.config.json')).read_text())
    assert config['environment']['COOL_REQUEST_C']=='51'
for prompt in [128,256]:
    assert min((r for r in records if r['prompt']==prompt),key=lambda r:r['request_ms'])['route']=='npu'
lines=['# Complete resident-model request placement comparison', '',
    'All twenty 128/256-token rows pass independent model/source/binary, numerical,',
    'actual-device-count, clock and restoration audits. All nine CPU/GPU/NPU',
    'projection pairs and the explicit three-device route were tested. Every',
    'request starts at <=51°C with a bounded idle cooldown. The entire older',
    '50°C partial request sweep remains rejected; none of its rows are reused.', '',
    'All rows use the same pinned Qwen3-0.6B FP16 checkpoint and exact input IDs,',
    'context 512, 32 outputs, two complete warmups and two measured requests.',
    'Decode inputs are the same teacher IDs and every actual greedy prediction',
    'is checked, including warmups. Effective prefill = prompt tokens / TTFT;',
    'TTFT includes the classifier and sampling. Decode covers 31 subsequent tokens.', '',
    'Request time is independently captured from prefill start through the last',
    'output, including activation packing, copies, cache preparation, DMA sync,',
    'device/queue waits and interphase work. Model initialization, pre-request',
    'cooling, tokenization/input-ID reading and post-request reporting are excluded.',
    'This is resident-model inference latency, not cold-start process latency.', '',
    'CPU/GPU/NPU/DDR clocks are fixed, sampled and restored. Warmups and measured',
    'requests must hold all recorded targets. The common start limit does not',
    'establish sustained hot-loop throughput. No fan or thermal policy is changed.', '',
    'Route names describe projection placement; [PHASES.md](PHASES.md) records',
    'attention and classifier placements. Embedding, norms, RoPE, SwiGLU,',
    'residuals and sampling remain CPU operations. GPU measurements use the',
    'current custom Mali OpenCL implementation, not a fully GPU-resident',
    'transformer or a fastest-possible llama.cpp claim. Execution dependencies',
    'are serial; no concurrent partition of one FFN matrix is implemented.', '',
    '![Complete placement requests](../../yalm-request-matrix.png)', '',
    '[PNG](../../yalm-request-matrix.png) · [JPG](../../yalm-request-matrix.jpg) · [SVG](../../yalm-request-matrix.svg)']
for prompt in [128,256]:
    lines.extend(['',f'## {prompt} input tokens', '',
        '| Prefill → decode projections | TTFT ms | Prefill t/s | Decode t/s | Request ms | Output t/s across request | Request min–max ms |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for route in order:
        row=next(r for r in records if r['prompt']==prompt and r['route']==route)
        lo,hi=row['request_range']
        label=route.replace('_',' → ') if '_' in route else route+' → '+route
        if route=='all':label='GPU → NPU + CPU head / GPU decode attention'
        lines.append(f"| {label} | {row['ttft_ms']:.3f} | {row['effective_prefill_tps']:.2f} | {row['decode_tps']:.3f} | {row['request_ms']:.3f} | {row['output_tps']:.3f} | {lo:.3f}–{hi:.3f} |")
lines.extend(['', '## Quality and interpretation', '',
    'Every displayed long-prompt row passes the unchanged first-logit relative',
    'RMSE <0.001 gate, and all 32 predictions match in both complete warmups and',
    'measurements. Earlier CPU/GPU-prefill 24/73-token failures remain recorded:',
    'these long-prompt passes do not qualify those routes for general deployment.',
    'The [42-case numerical audit](candidate-checks/quality-audit42.json) retains',
    'all twelve failed gates. Qualified NPU-prefill candidates and their separate',
    'counterbalanced results are in [ITERATIONS.md](ITERATIONS.md).', '',
    'The selected NPU projection route with CPU decode attention remains the best',
    'complete-request median in this tested placement matrix. Adding devices does',
    'not automatically improve phase or complete-request throughput. Two measured',
    'requests per row and min/max ranges are not confidence intervals.', '',
    '[Independent audit](request-matrix51-audit.json) recomputes every numerical',
    'and timing gate, checks all warmup/measured device counts and predictions,',
    'and verifies live clock settings against the final restored snapshot.',
    'Raw logs/configs/tokens and compressed clock samples are preserved under',
    '`evidence/request-matrix51`; model, full vocabulary logits and compiled',
    'binary remain in the comparison workspace with hashes retained.', '',
    'These custom-route rows are a fresh placement comparison. The earlier paired',
    'same-FP16 stock RKLLM comparison remains in [ROOFLINE.md](../../ROOFLINE.md);',
    'this matrix does not claim a fresh stock complete-request comparison.', '',
    'INTENT: the selected runner has NPU projections and a tested GPU-attention route but no full CPU/GPU projection comparison; the user requests device combinations and stepwise profiling/optimization; fp16/README.md specifies the shared FP16 checkpoint, native NPU projection reference and numerical verification.', '',
    'AUTH: user said "wip commit per milestone".', ''])
(repo/'REQUESTS.md').write_text('\n'.join(lines))
print('Wrote audited request placement report')
