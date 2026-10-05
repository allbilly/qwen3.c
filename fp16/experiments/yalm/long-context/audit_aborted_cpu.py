"""Audit the CPU 2K warmup abort without creating a measured benchmark row."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = args.workspace
parent = root / 'e2e/long-context'
directory = parent / 'p2048-cpu-screen55-cpu1800-fanheld'
label = 'p2048-1-cpu'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert not (directory / 'summary.json').exists()
config = json.loads((directory / (label + '.config.json')).read_text())
env = config['environment']
meta = json.loads((parent / 'source-provenance.json').read_text())
assert config['route'] == 'cpu' and config['cpu_target_khz'] == 1800000
assert config['binary_sha256'] == sha(directory / 'runq-routes') == meta['binary_sha256']
assert json.loads((directory / 'source-provenance.json').read_text()) == meta
for name, digest in meta['source_sha256'].items():
    assert sha(parent / name) == digest
assert sha(Path('/home/orangepi/qwen3.c/runq-fp16')) == meta['selected_binary_sha256']
assert env['BENCH_CONTEXT'] == '4128' and env['WARMUP_RUNS'] == '2'
assert env['OMP_NUM_THREADS'] == '4' and env['COOL_REQUEST_C'] == '55'
assert env['DENSE_PREFILL'] == env['DENSE_DECODE'] == 'cpu'
assert all(env[key] is None for key in ['NPU_PROFILE', 'GPU_PROFILE', 'LINEAR_PROFILE', 'FFN_PROFILE'])
assert config['command'][:3] == ['taskset', '-c', '4-7']
for stem in ['clock', 'gpu-clock']:
    assert (directory / (stem + '-state-before.json')).read_bytes() == (directory / (stem + '-state-restored.json')).read_bytes()
golden = parent / config['golden_directory']
assert golden.name == 'references55-cpu1800-fanheld'
golden_label = 'p2048-1-npu'
ids = lambda path: list(map(int, path.read_text().split()))
assert len(ids(directory / (label + '.tokens'))) == 2048
assert ids(directory / (label + '.tokens')) == ids(golden / (golden_label + '.tokens'))
rows = [json.loads(line) for line in (directory / (label + '.jsonl')).read_text().splitlines() if line.startswith('{')]
events = lambda event: [row for row in rows if row.get('event') == event]
assert events('runtime_context') == [{'event': 'runtime_context', 'checkpoint_context': 512, 'capacity': 4128, 'max_prompt': 4096}]
routing, = events('routing')
assert routing['dense_prefill'] == routing['dense_decode'] == 'cpu'
assert not any(routing[key] for key in ['npu_initialized', 'gpu_projections_initialized'])
run, = events('run')
request, = events('request_wall')
assert run['run'] == request['run'] == -1 and run['warmup'] and request['warmup']
assert run['prefill_tokens'] == request['prompt_tokens'] == 2048
assert run['new_tokens'] == request['output_tokens'] == 32 and run['decode_steps'] == 31
assert abs(request['request_ms'] - request['prefill_ms'] - request['decode_ms'] - request['handoff_ms']) < .00001
assert abs(request['prefill_ms'] - run['first_token_ms']) <= .000501
assert abs(request['decode_ms'] - run['decode_ms']) <= .000501
assert abs(31000 / run['decode_ms'] - run['decode_tps']) <= .000501
golden_rows = [json.loads(line) for line in (golden / (golden_label + '.jsonl')).read_text().splitlines() if line.startswith('{')]
golden_ids = next(row['generated_ids'] for row in golden_rows if row.get('event') == 'run')
assert run['generated_ids'] == ids(directory / (label + '.teacher.tokens')) == golden_ids
for phase, calls in [('prefill', 197), ('decode', 6107)]:
    device, = [row for row in events('device_phase') if row['phase'] == phase]
    profile, = [row for row in events('linear_cpu_profile') if row['phase'] == phase]
    assert device['run'] == profile['run'] == -1
    assert device['npu_ops'] == 0 and device['cpu_ops'] == profile['calls'] == calls
    assert profile['prefill_attention_calls'] == (28 if phase == 'prefill' else 0)
cooldown, = events('cooldown')
assert cooldown['run'] == -1 and cooldown['after_millidegrees'] <= 55000
abort, = events('cooldown_abort')
assert rows[-1] == abort and abort['run'] == 0 and abort['target_millidegrees'] == 55000
assert abort['temperature_millidegrees'] > 55000 and abort['wait_ms'] >= 180000
a = np.fromfile(directory / (label + '.f32'), np.float32).astype(np.float64)
b = np.fromfile(golden / (golden_label + '.f32'), np.float32).astype(np.float64)
assert a.shape == b.shape == (151936,) and np.isfinite(a).all() and np.isfinite(b).all()
relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
clocks = [json.loads(line) for line in (directory / 'clock-samples.jsonl').read_text().splitlines()]
assert clocks
targets = {'cpu4_khz': 1800000, 'cpu6_khz': 1800000, 'npu_hz': 1000000000, 'gpu_hz': 1000000000, 'ddr_hz': 2112000000}
out = {
    'directory': str(directory.relative_to(root)),
    'status': 'aborted during cooldown before second warmup',
    'completed_warmups': 1, 'measured_requests': 0,
    'cooldown_abort': abort,
    'warmup_request_diagnostic': request,
    'warmup_first_logit_relative_rmse': relative,
    'warmup_first_logits_pass_gate': relative < .001,
    'warmup_actual_ids_match': True, 'actual_cpu_only_execution_checked': True,
    'clock_and_gpu_snapshots_restored': True,
    'sampled_temperature_max_c': max(row['temperature_millidegrees'] for row in clocks) / 1000,
    'samples_with_clock_drop': sum(any(row[key] != target for key, target in targets.items()) for row in clocks),
    'binary_sha256': meta['binary_sha256'],
    'artifact_sha256': {name: sha(directory / name) for name in [label + '.config.json', label + '.tokens', label + '.teacher.tokens', label + '.jsonl', label + '.f32', 'clock-samples.jsonl']},
    'scope': 'One warmup is a diagnostic, not a measured benchmark row. No timing from this aborted job enters the comparison. Live governor/limit restoration is checked separately.'
}
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({key: out[key] for key in ['status', 'completed_warmups', 'measured_requests', 'warmup_first_logit_relative_rmse', 'samples_with_clock_drop']}))
