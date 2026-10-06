"""Recover observed GPU4K timings without inventing missing restoration records."""
from pathlib import Path
import argparse, hashlib, json, math, statistics
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = args.workspace
parent = root / 'e2e/long-context'
directory = parent / 'p4096-gpu-continue-throttle-cpu1800-fanheld'
label = 'p4096-1-gpu'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
meta = json.loads((parent / 'source-provenance.json').read_text())
config = json.loads((directory / (label + '.config.json')).read_text())
env = config['environment']
assert not (directory / 'summary.json').exists()
assert config['route'] == 'gpu' and config['cpu_target_khz'] == 1800000
assert config['allow_clock_drops'] and not config['cooldown_enforced']
assert env['COOL_REQUEST_C'] is None and env['WARMUP_RUNS'] == '2'
assert env['BENCH_CONTEXT'] == '4128' and env['OMP_NUM_THREADS'] == '4'
assert env['DENSE_PREFILL'] == env['DENSE_DECODE'] == 'gpu' and env['GPU_ATTENTION'] == 'both'
assert all(env[key] is None for key in ['NPU_PROFILE', 'GPU_PROFILE', 'LINEAR_PROFILE', 'FFN_PROFILE'])
assert config['binary_sha256'] == sha(directory / 'runq-routes') == meta['binary_sha256']
model=root.parents[1] / 'Qwen3-0.6B.fp16'
assert sha(model)=='c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879'
assert config['command']==['taskset','-c','4-7',str(directory / 'runq-routes'),str(model),
                          str(directory / (label+'.tokens')),'32','2',str(directory / (label+'.f32'))]
for name, digest in meta['source_sha256'].items():
    assert sha(parent / name) == digest
assert sha(Path('/home/orangepi/qwen3.c/runq-fp16')) == meta['selected_binary_sha256']
golden = parent / config['golden_directory']
golden_label = 'p4096-1-npu'
ids = lambda path: list(map(int, path.read_text().split()))
assert len(ids(directory / (label + '.tokens'))) == 4096
assert ids(directory / (label + '.tokens')) == ids(golden / (golden_label + '.tokens'))
rows = [json.loads(line) for line in (directory / (label + '.jsonl')).read_text().splitlines() if line.startswith('{')]
events = lambda event: [row for row in rows if row.get('event') == event]
routing, = events('routing')
assert not routing['npu_initialized'] and routing['gpu_projections_initialized']
assert not events('cooldown') and not events('cooldown_abort')
runs = events('run')
requests = events('request_wall')
assert [row['run'] for row in runs] == [row['run'] for row in requests] == [-1, 0, 1, 2]
golden_rows = [json.loads(line) for line in (golden / (golden_label + '.jsonl')).read_text().splitlines() if line.startswith('{')]
golden_ids = next(row['generated_ids'] for row in golden_rows if row.get('event') == 'run')
assert ids(directory / (label + '.teacher.tokens')) == golden_ids
predictions = all(row['generated_ids'] == golden_ids for row in runs)
for run, request in zip(runs, requests):
    index = run['run']
    assert run['warmup'] == request['warmup'] == (index <= 0)
    assert run['prefill_tokens'] == request['prompt_tokens'] == 4096
    assert run['new_tokens'] == request['output_tokens'] == 32 and run['decode_steps'] == 31
    assert abs(request['request_ms'] - request['prefill_ms'] - request['decode_ms'] - request['handoff_ms']) < .00001
    assert abs(request['prefill_ms'] - run['first_token_ms']) <= .000501
    assert abs(request['decode_ms'] - run['decode_ms']) <= .000501
    assert abs(31000 / run['decode_ms'] - run['decode_tps']) <= .000501
    for phase, n in [('prefill', 1), ('decode', 31)]:
        device, = [row for row in events('device_phase') if row['run'] == index and row['phase'] == phase]
        projection, = [row for row in events('linear_gpu_profile') if row['run'] == index and row['phase'] == phase]
        attention, = [row for row in events('gpu_profile') if row['run'] == index and row['phase'] == phase]
        assert device['npu_ops'] == device['cpu_ops'] == 0
        assert projection['calls'] == 197 * n and attention['calls'] == 28 * n
        assert attention['kernel_calls'] == (112 if phase == 'prefill' else 2604)
a = np.fromfile(directory / (label + '.f32'), np.float32).astype(np.float64)
b = np.fromfile(golden / (golden_label + '.f32'), np.float32).astype(np.float64)
assert a.shape == b.shape == (151936,) and np.isfinite(a).all() and np.isfinite(b).all()
relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
measured = [row for row in runs if not row['warmup']]
timed = [row for row in requests if not row['warmup']]
clocks = [json.loads(line) for line in (directory / 'clock-samples.jsonl').read_text().splitlines()]
row_clocks = [row for row in clocks if row['phase'] == label or row['phase'] == 'cooling-before-' + label]
assert row_clocks
targets = {'cpu4_khz': 1800000, 'cpu6_khz': 1800000, 'npu_hz': 1000000000, 'gpu_hz': 1000000000, 'ddr_hz': 2112000000}
prefill = statistics.median(row['first_token_ms'] for row in measured)
out = {'directory': str(directory.relative_to(root)), 'label': label, 'prompt': 4096, 'route': 'gpu',
       'reference': False, 'quality_passed': relative < .001 and predictions,
       'relative_rmse': relative, 'all_predictions_match': predictions,
       'clocks_held': all(row[key] == target for row in row_clocks for key, target in targets.items()),
       'row_clock_samples': len(row_clocks),
       'row_clock_drops': sum(any(row[key] != target for key, target in targets.items()) for row in row_clocks),
       'row_temperature_range_c': [min(row['temperature_millidegrees'] for row in row_clocks) / 1000, max(row['temperature_millidegrees'] for row in row_clocks) / 1000],
       'clock_frequency_ranges': {key: [min(row[key] for row in row_clocks), max(row[key] for row in row_clocks)] for key in targets},
       'cooldown_target_c': 0, 'cooldown_enforced': False, 'allow_clock_drops': True,
       'device_counts_checked': True, 'prefill_ms': prefill, 'prefill_tps': 4096000 / prefill,
       'decode_tps': statistics.median(row['decode_tps'] for row in measured),
       'request_ms': statistics.median(row['request_ms'] for row in timed),
       'prefill_range': [min(row['first_token_ms'] for row in measured), max(row['first_token_ms'] for row in measured)],
       'decode_range': [min(row['decode_tps'] for row in measured), max(row['decode_tps'] for row in measured)],
       'request_range': [min(row['request_ms'] for row in timed), max(row['request_ms'] for row in timed)],
       'logit_sha256': sha(directory / (label + '.f32')), 'golden_logit_sha256': sha(golden / (golden_label + '.f32')),
       'raw_log_sha256': sha(directory / (label + '.jsonl')), 'prompt_sha256': sha(directory / (label + '.tokens')),
       'binary_sha256': meta['binary_sha256'], 'clock_restoration_verified': False,
       'scope': 'Four raw requests completed. Parent summary and restoration records are absent. This audit verifies observed request metrics and quality, not normal controller exit or restoration. No summary or restoration artifact is fabricated.'}
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({key: out[key] for key in ['prefill_tps', 'decode_tps', 'request_ms', 'quality_passed', 'relative_rmse', 'clocks_held', 'clock_restoration_verified']}))
