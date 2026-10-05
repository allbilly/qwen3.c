"""Freeze completed request/overlap evidence and generate tables from audited data.

Run only after all hardware jobs and independent audits have finished.
"""
from pathlib import Path
import gzip, hashlib, json, shutil, subprocess

root = Path(__file__).resolve().parent
target = Path('/home/orangepi/qwen3.c/fp16/experiments/yalm')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda name: json.loads((root / name).read_text())

exports = [
    ('e2e/concurrent-ffn/request-matrix51', 'concurrent-original-requests51'),
    ('e2e/concurrent-native/request-matrix51', 'concurrent-native-requests51'),
]
for block in range(1, 5):
    exports.append((f'e2e/concurrent-ffn/iteration-concurrent-cpu-decode-e2e51-abba-{block}',
                    f'concurrent-cpu-decode-abba-{block}'))
for variant in ['concurrent-overlap-proof', 'concurrent-calibrated-proof', 'concurrent-traced-proof']:
    exports.append((f'e2e/{variant}/kernel-overlap51', variant + '-51'))
for directory, name in exports:
    dest = target / 'evidence' / name
    if not dest.exists():
        subprocess.run(['python3', str(root/'export_evidence.py'), directory, name], cwd=root, check=True)

for variant in ['concurrent-overlap-proof', 'concurrent-calibrated-proof', 'concurrent-traced-proof']:
    source = root/'e2e'/variant
    dest = target/variant
    dest.mkdir(exist_ok=True)
    meta = json.loads((source/'source-provenance.json').read_text())
    for name, digest in meta['source_sha256'].items():
        assert sha(source/name) == digest, (variant, name)
        output = dest/name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, output)
        assert sha(output) == digest
    for name in ['source-provenance.json','parent-provenance.json','build-command.json','timing-scope.json']:
        shutil.copy2(source/name, dest/name)
    shutil.copy2(target/'concurrent-native'/'build.py', dest/'build.py')

toolsdir = target/'concurrent-tools'
toolsdir.mkdir(exist_ok=True)
tools = ['plot_concurrent.py','audit_overlap_traces.py','audit_concurrent_matrix.py',
         'concurrent_audit.py','attention_rows.py','analyse_iterations.py','export_evidence.py',
         'prepare_overlap_proof.py','prepare_calibrated_overlap.py','prepare_traced_overlap.py',
         'run_concurrent_matrix.py','run_iterations.py','finalize_concurrent.py']
for name in tools:
    shutil.copy2(root/name, toolsdir/name)
(toolsdir/'source-provenance.json').write_text(json.dumps({
    'workspace':str(root),'source_sha256':{name:sha(toolsdir/name) for name in tools},
    'scope':'Snapshots of comparison-workspace tools. Hardware harness paths refer to that workspace; source variant build.py uses the repository export directly.'
}, indent=2)+'\n')

artifacts = ['concurrent-request40-audit.json','concurrent-request-plan.json',
             'concurrent-cpu-decode-abba-audit.json','concurrent-cpu-decode-abba-report.json',
             'iteration-concurrent-cpu-decode-e2e51-abba-plan.json',
             'concurrent-overlap-device-clock-audit.json','concurrent-overlap-trace-audit.json']
for stem in ['concurrent-original-requests','concurrent-native-requests']:
    artifacts += [stem+ext for ext in ['.png','.jpg','.svg','-provenance.json']]
for name in artifacts:
    shutil.copy2(root/name, target/name)
    assert sha(root/name) == sha(target/name)

verified = 0
for directory, name in exports:
    dest = target/'evidence'/name
    manifest = json.loads((dest/'export-manifest.json').read_text())
    assert Path(manifest['workspace_directory']).resolve() == (root/directory).resolve()
    for original, record in manifest['files'].items():
        src, output = root/directory/original, dest/record['export']
        assert sha(src)==record['original_sha256'] and sha(output)==record['export_sha256']
        payload = gzip.decompress(output.read_bytes()) if output.name.endswith('.gz') else output.read_bytes()
        assert src.read_bytes()==payload, (name,original)
        verified += 1

audit = read('concurrent-request40-audit.json')
paired = read('concurrent-cpu-decode-abba-report.json')
trace = read('concurrent-overlap-trace-audit.json')
assert len(audit['records'])==40 and audit['selected_binary_unchanged']
assert all(r['quality_passed'] and r['all_predictions_match'] and r['measured_runs']==2 for r in audit['records'])
assert all(a['clocks_held_and_restored'] for a in audit['audits'].values())
assert all(r['jobs']==112 and r['valid_clock_maps']==112 and r['all_three_contains_submit']==84 for r in trace['records'])

lines = ['# Concurrent FFN: phase and complete-request results', '',
    'The selected CPU-host/NPU-matrix engine is retained. Concurrent CPU/Mali/NPU FFN slices are implemented and checked, but these measured placements do not establish a reliable complete-request gain.', '',
    '## Measurement scope', '',
    'All rows use the pinned Qwen3-0.6B FP16 operands, 32 outputs (31 decode steps), two full warmups and two measurements. CPU/NPU/Mali/DDR clocks are 2.256/1.000/1.000/2.112 GHz, sampled and restored. Each request starts at <=51°C. Prefill throughput is input tokens / TTFT, including the first classifier and sampling. Request time is independently captured from prefill start through the last argmax and includes movement, DMA synchronization, queue waits and handoffs. Model loading, input-ID file reading, tokenization, warmups and cooldown are excluded.', '',
    'The forty-row sweep screens placements. The original layout ran before the native layout; this order is not a counterbalanced layout comparison. Every long-prompt row passes its quality/device/clock audits. Prefill and both-phase split rows remain **diagnostics** because the separate 24/73-token checks fail the unchanged first-logit gate (six failures per layout). Passing predictions at 128/256 tokens does not resolve those failures.', '',
    'Device labels describe auxiliary **FFN gate/up slices**. QKV, output, down and vocabulary projections stay on NPU; other transformer work stays on CPU. These are not whole-model GPU or NPU-resident graph claims.', '',
    '[Independent forty-job audit](concurrent-request40-audit.json) · [written execution plan](concurrent-request-plan.json) · [architecture, precision and branch costs](CONCURRENT.md)', '']
order=['npu','cpu_npu_pre','gpu_npu_pre','all_pre','cpu_npu_dec','gpu_npu_dec','all_dec','cpu_npu_both','gpu_npu_both','all_both']
labels={'npu':'Selected NPU path','cpu_npu_pre':'CPU+NPU prefill*','gpu_npu_pre':'GPU+NPU prefill*','all_pre':'CPU+GPU+NPU prefill*','cpu_npu_dec':'CPU+NPU decode','gpu_npu_dec':'GPU+NPU decode','all_dec':'CPU+GPU+NPU decode','cpu_npu_both':'CPU+NPU both*','gpu_npu_both':'GPU+NPU both*','all_both':'CPU+GPU+NPU both*'}
for variant, title, stem in [('concurrent-ffn','Original full-prompt split','concurrent-original-requests'),('concurrent-native','Native packing with 64-row blocks','concurrent-native-requests')]:
    lines += ['## '+title,'',f'![{title}]({stem}.png)','',f'[PNG]({stem}.png) · [JPG]({stem}.jpg) · [SVG]({stem}.svg)','',
              '| Input tokens | FFN placement | Prefill ms | Prefill tokens/s | Decode tokens/s | Request ms | Request min–max ms |',
              '|---|---|---:|---:|---:|---:|---|']
    for prompt in [128,256]:
        for route in order:
            r=next(r for r in audit['records'] if r['directory'].split('/')[1]==variant and r['prompt']==prompt and r['route']==route)
            lo,hi=r['request_range']
            lines.append(f"| {prompt} | {labels[route]} | {r['ttft_ms']:.3f} | {r['effective_prefill_tps']:.2f} | {r['decode_tps']:.3f} | {r['request_ms']:.3f} | {lo:.3f}–{hi:.3f} |")
    lines += ['', '* Prefill precision gate failures on separate short prompts; diagnostic placements.', '']

lines += ['## CPU+NPU decode: counterbalanced check', '',
    'The original 128-token screen appeared to improve decode slightly. A same-binary ABBA check isolates the `FFN_MODE=2`, 96-channel CPU slice from the selected NPU path. There are four individual measured requests per route and prompt. All eight jobs pass source/model/binary, quality, actual placement, fixed clocks and restoration audits. The following medians are recomputed from all four raw measurements, not from block medians.', '',
    '| Input tokens | FFN decode route | Prefill ms | Prefill tokens/s | Decode tokens/s | Request ms | Request min–max ms |',
    '|---|---|---:|---:|---:|---:|---|']
for comparison in paired['results']:
    for key,label in [('baseline','Selected NPU path'),('candidate','CPU+NPU decode')]:
        m=comparison[key]['median'];lo,hi=comparison[key]['ranges']['request_ms']
        lines.append(f"| {comparison['prompt']} | {label} | {m['ttft_ms']:.3f} | {m['effective_prefill_tps']:.2f} | {m['decode_tps']:.3f} | {m['request_ms']:.3f} | {lo:.3f}–{hi:.3f} |")
lines += ['',
    'CPU+NPU decode throughput changes by -2.01% / -1.49% for 128/256 tokens; complete-request throughput changes by -1.91% / -1.17%. All observed prefill, decode and request ranges overlap. Four observations are not confidence intervals. The screen gain is not reproduced and this split is not promoted.', '',
    '[ABBA raw aggregation](concurrent-cpu-decode-abba-report.json) · [independent audit](concurrent-cpu-decode-abba-audit.json) · [execution plan](iteration-concurrent-cpu-decode-e2e51-abba-plan.json)', '',
    '## Physical overlap diagnostic', '',
    'An initial OpenCL RUNNING-status probe did not establish overlap. A separate diagnostic then brackets GPU marker END events with host monotonic timestamps before enqueue and after wait, intersects the before/after clock-offset bounds, and widens each endpoint by 10 microseconds. Raw GPU kernel START/END, CPU worker START/END and blocking NPU submit START/END timestamps are retained for all 336 FFN blocks.', '',
    'The independent timestamp auditor reproduces **84/112 blocks with simultaneous CPU worker, GPU kernel event interval and NPU blocking-submit interval** in each of two warmups and the measured request. All 112 clock maps per request are valid; the largest offset interval is 0.042166 ms. It checks conservative interval containment, rather than summing overlapping branch times. The NPU submit span includes driver work and memory stalls, so it does not isolate MAC activity. GPU kernel event spans likewise identify command execution intervals.', '',
    'Marker waits and timestamp instrumentation change scheduling. This establishes concurrent execution in the diagnostic; it does not establish the same overlap fraction or any throughput gain in the unprofiled forty-row sweep. No diagnostic timing is used in those throughput tables.', '',
    '[Independent raw trace audit](concurrent-overlap-trace-audit.json) · [quality/device/clock audit of all three probes](concurrent-overlap-device-clock-audit.json) · [raw traced request log](evidence/concurrent-traced-proof-51/p256-1-all_both.jsonl) · [trace auditor](concurrent-tools/audit_overlap_traces.py) · [frozen diagnostic source](concurrent-traced-proof/concurrent_ffn.c)', '',
    '## Measured changes and next targets', '',
    'Native packing removes almost all full gate/up assembly cost in the CPU+NPU profile (24.897 ms to 0.014 ms), but exposes CPU-worker wait and activation packing. Its CPU split remains slower than the selected NPU reference. On Mali, changing the auxiliary job shape to 64 rows raises profiled GPU kernel time from 96.639 ms to 534.538 ms and join wait from 20.981 ms to 491.723 ms. Saved assembly does not compensate. These branch profiles precede the complete-request measurements; profile timings are event-enabled diagnostics.', '',
    'The selected profile identifies CPU decode attention (4.406 ms/token) and prefill activation movement/layout/SwiGLU as useful next targets. Earlier NPU attention batch128 improves prefill in a paired test but has overlapping complete-request ranges. Earlier fused GPU decode attention loses complete-request throughput. A fully GPU-resident transformer or larger tuned FFN partitions require further implementation and quality checks; the present custom Mali kernels are not an optimal llama.cpp GPU baseline.', '',
    '## Reproducibility and evidence', '',
    'Original and native source exports reproduce their exact tested binary hashes: `a2480d20...` and `2922f251...`. The traced observer is `d3544011...`; the selected engine remains `100716de...`. Full source/binary hashes and compiler commands are frozen in each variant. Use each exported `build.py` to compile offline; hardware processes must be run serially.', '',
    f'All {verified} newly exported configuration/log/token/clock files were checked byte-for-byte against the external workspace, decompressing deterministic gzip clock samples. Model files, full-vocabulary binary logit dumps and compiled executables remain in that workspace; audits record their hashes. Sources and small evidence are committed locally. No push is authorized.', '',
    'INTENT: the router waits for each device projection; the user requests mixed CPU/GPU/NPU inference and transfer-inclusive phase and request measurements; fp16/README.md requires the shared FP16 checkpoint, numerical checks and serial NPU submissions.', '',
    'INTENT: the first split assembles and repacks full FFN tensors; the user requests profiling followed by concrete prefill improvements; fp16/README.md describes direct activation packing between FFN projections with the verified FP16 operands.', '',
    'AUTH: user said "wip commit per milestone".', '']
(target/'CONCURRENT-REQUESTS.md').write_text('\n'.join(lines))
verification={'new_evidence_files_losslessly_verified':verified,'new_evidence_exports':[name for _,name in exports],
              'source_variants_exported':3,'compiled_source_files_verified':63,
              'artifacts_sha256':{name:sha(target/name) for name in artifacts},
              'selected_binary_sha256':sha(Path('/home/orangepi/qwen3.c/runq-fp16')),
              'scope':'Offline immutable export/table verification after all hardware jobs exited normally and all clock restoration audits passed.'}
assert verification['selected_binary_sha256']=='100716de3aca7c4efc7a5b0a5a87178c41adc3bc4049a5f1c009a83fe051ff58'
(target/'concurrent-final-export-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({'new_evidence_files_verified':verified,'request_jobs':40,'abba_jobs':8,'overlap_diagnostics':3},indent=2))
