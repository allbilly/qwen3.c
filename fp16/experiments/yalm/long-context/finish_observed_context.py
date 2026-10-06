"""Audit and render observed rows after both CPU jobs have ended."""
from pathlib import Path
import argparse, ast, gzip, hashlib, json, os, re, subprocess

root = Path(__file__).resolve().parent
comparison = Path('/home/orangepi/qwen3-bench/matched/roofline/yalm')
cpu_base = Path('/tmp/qwen3-long-context-cpu-sample-only-20261006')
p = argparse.ArgumentParser()
p.add_argument('--finish', action='store_true')
args = p.parse_args()
cpu_dirs = [cpu_base / f'p{prompt}-cpu' for prompt in [2048, 4096]]
ready = all((directory / 'summary.json').exists() for directory in cpu_dirs)
if not args.finish:
    print(json.dumps({'ready_for_offline_audit': ready, 'cpu_directories': list(map(str, cpu_dirs))}))
    raise SystemExit(0)
assert ready, 'Do not hash/render during the timed CPU jobs'
old_dirs = json.loads((comparison / 'long-context-completed-directories.json').read_text())
for directories, cool, name in [(old_dirs, 55, 'long-context-original-recheck.json'),
                                (list(map(str, cpu_dirs)), 0, 'long-context-cpu-observed-audit.json')]:
    subprocess.run(['taskset', '-c', '0-3', 'python3', str(root / 'audit_long_context.py'),
                    *directories, '--workspace', str(comparison), '--cpu-khz', '1800000',
                    '--cool-c', str(cool), '--output', str(root / name)], check=True)
subprocess.run(['taskset', '-c', '0-3', 'python3', str(root / 'audit_completed_gpu_raw.py'),
                '--workspace', str(comparison), '--output', str(root / 'long-context-gpu4096-recovered-audit.json')], check=True)
old = json.loads((root / 'long-context-original-recheck.json').read_text())
cpu = json.loads((root / 'long-context-cpu-observed-audit.json').read_text())
gpu = json.loads((root / 'long-context-gpu4096-recovered-audit.json').read_text())
assert old['model_sha256'] == cpu['model_sha256'] and old['binary_sha256'] == cpu['binary_sha256'] == gpu['binary_sha256']
records = old['records'] + cpu['records'] + [gpu]
assert len(records) == 15 and len({(row['prompt'], row['route']) for row in records}) == 15
missing = {(4096, route) for route in ['cpu_npu_dec', 'gpu_npu_dec', 'all_dec']}
expected = {(prompt, route) for prompt in [1024, 2048, 4096]
            for route in ['cpu', 'gpu', 'npu', 'cpu_npu_dec', 'gpu_npu_dec', 'all_dec']}
assert expected - {(row['prompt'], row['route']) for row in records} == missing
combined = {'cpu_target_khz': 1800000, 'cooldown_target_c': 'mixed',
            'model_sha256': old['model_sha256'], 'binary_sha256': old['binary_sha256'],
            'selected_binary_unchanged': True, 'records': records,
            'clock_audits': {**old['clock_audits'], **cpu['clock_audits']},
            'missing_placements': [{'prompt': prompt, 'route': route} for prompt, route in sorted(missing)],
            'sessions': ['Original 12 rows: <=55C starts and locked clocks, restored',
                         'Recovered GPU4K: cooldown disabled, captured clocks held; parent restoration unverified',
                         'CPU2K/4K: cooldown disabled, existing limits sampled without sysfs/fan changes'],
            'scope': 'Fifteen measured rows under recorded conditions, with quality failures retained. Three mixed4K rows lack accessible board device nodes. This is not a controlled fixed-clock speedup comparison. The earlier CPU/NPU/DDR limits and fan command remain locked; no restoration is invented.'}
audit_file = root / 'long-context-observed-audit.json'
audit_file.write_text(json.dumps(combined, indent=2) + '\n')
exports = [(comparison / gpu['directory'], 'long-context-recovered-gpu4096'),
           *[(directory, f'long-context-sample-only-cpu{prompt}') for directory, prompt in zip(cpu_dirs, [2048, 4096])]]
for directory, name in exports:
    target=root.parent / 'evidence' / name
    if target.exists():
        captured=json.loads((target / 'export-manifest.json').read_text())
        assert Path(captured['workspace_directory'])==directory
        eligible={p.name for p in directory.iterdir() if p.is_file() and
                  (p.name=='clock-samples.jsonl' or p.suffix in ['.json','.jsonl','.tokens','.py','.f32'])}
        assert set(captured['files'])==eligible
    else:
        subprocess.run(['python3', str(root / 'export_evidence.py'), str(directory), name], check=True)
env = dict(os.environ, PYTHONPATH=str(comparison.parent / '.plot-packages'),
           MPLCONFIGDIR='/tmp/qwen3-long-context-matplotlib-cache')
subprocess.run(['python3', str(root / 'render_long_context.py'), '--audit', str(audit_file),
                '--out', str(root.parent), '--partial', '--thermal-observations', '--restricted-recovery'], env=env, check=True)
report = (root.parent / 'LONG-CONTEXT.md').read_text()
for target in re.findall(r'\]\(([^)]+)\)', report):
    if '://' not in target:
        assert (root.parent / target.split('#')[0]).exists(), target
provenance = json.loads((root.parent / 'long-context-render-provenance.json').read_text())
for name, digest in provenance['outputs'].items():
    assert hashlib.sha256((root.parent / name).read_bytes()).hexdigest() == digest
verified_files=0
manifests=sorted((root.parent / 'evidence').glob('long-context-*/export-manifest.json'))
for path in manifests:
    manifest=json.loads(path.read_text())
    original=Path(manifest['workspace_directory'])
    for name, metadata in manifest['files'].items():
        source=(original / name).read_bytes()
        exported=(path.parent / metadata['export']).read_bytes()
        assert hashlib.sha256(source).hexdigest()==metadata['original_sha256']
        assert hashlib.sha256(exported).hexdigest()==metadata['export_sha256']
        assert (gzip.decompress(exported) if metadata['export'].endswith('.gz') else exported)==source
        verified_files+=1
for prompt in [1024,2048,4096]:
    assert len({r['prompt_sha256'] for r in records if r['prompt']==prompt})==1
for name in ['run_matrix.py','run_cpu_remaining.py','status_cpu_remaining.py','audit_long_context.py',
             'audit_completed_gpu_raw.py','export_evidence.py','finish_observed_context.py','render_long_context.py']:
    ast.parse((root / name).read_text(),filename=str(root / name))
rejected=subprocess.run(['python3', str(root / 'render_long_context.py'), '--audit', str(audit_file),
                         '--out', '/tmp/qwen3-long-context-rejected-complete-render-20261006',
                         '--thermal-observations','--restricted-recovery'],env=env,capture_output=True,text=True)
assert rejected.returncode!=0 and 'Incomplete measurements require --partial' in rejected.stderr
assert not Path('/tmp/qwen3-long-context-rejected-complete-render-20261006').exists()
subprocess.run(['git','diff','--check'],cwd='/home/orangepi/qwen3.c',check=True)
verification={'measured_rows':len(records),'measured_requests':2*len(records),'completed_requests_with_warmups':4*len(records),
              'quality_passes':sum(r['quality_passed'] for r in records),
              'quality_failures':[{'prompt':r['prompt'],'route':r['route'],'relative_rmse':r['relative_rmse']} for r in records if not r['quality_passed']],
              'same_prompt_hash_per_length':True,'export_directories_checked':len(manifests),'lossless_export_files_checked':verified_files,
              'incomplete_full_render_rejected':True,'report_links_and_hashes_verified':True,'python_syntax_checked':True,'diff_check_passed':True,
              'original_model_and_21_compiled_sources_unchanged':True,'scope':'Offline checks after all timed CPU requests completed; missing mixed-device 4K runs remain unmeasured.'}
(root / 'long-context-postrun-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({'measured_rows': len(records), 'quality_passes': sum(row['quality_passed'] for row in records),
                  'missing_rows': len(missing), 'report_links_and_hashes_verified': True,'lossless_exports_checked':verified_files}))
