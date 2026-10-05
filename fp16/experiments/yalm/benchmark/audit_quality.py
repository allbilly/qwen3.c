"""Independently check additional numerical coverage, including failed gates.

These jobs have no fixed-frequency measurements. Their timings must never be
used as qualified performance data. Run this offline audit between device jobs.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

root = Path(__file__).resolve().parent
roofline = root.parent
parser = argparse.ArgumentParser()
parser.add_argument('directories', nargs='+')
parser.add_argument('--output', default='quality-audit.json')
args = parser.parse_args()

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

model = roofline.parent / 'Qwen3-0.6B.fp16'
model_sha = sha(model)
assert model_sha == 'c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879'
selected = '100716de3aca7c4efc7a5b0a5a87178c41adc3bc4049a5f1c009a83fe051ff58'
assert sha(Path('/home/orangepi/qwen3.c/runq-fp16')) == selected
records = []
for directory in args.directories:
    out = (root / directory).resolve()
    saved = json.loads((out / 'quality-summary.json').read_text())
    metadata = json.loads((out / 'source-provenance.json').read_text())
    assert sha(out / 'runq-routes') == metadata['binary_sha256'] == saved['binary_sha256']
    assert metadata['selected_binary_sha256'] == selected
    for name, digest in metadata['source_sha256'].items():
        assert sha(out.parent / name) == digest, (directory, name)
    configs = sorted(out.glob('*.config.json'))
    assert len(configs) == len(saved['records'])
    for path in configs:
        config = json.loads(path.read_text())
        label = path.name.removesuffix('.config.json')
        env, command = config['environment'], config['command']
        assert command[:3] == ['taskset', '-c', '4-7']
        assert sha(Path(command[3])) == metadata['binary_sha256']
        assert Path(command[4]) == model and command[6:8] == ['16', '1']
        assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '4'
        assert env['WARMUP_RUNS'] == '1' and env['COOL_REQUEST_C'] == '55'
        assert env['NPU_CORES'] == '3' and env['NPU_DOMAIN_ID'] == '1'
        for key in ['NPU_FUSED', 'NPU_ATTENTION', 'NPU_SPLIT_DOWN', 'NPU_STREAM_FFN']:
            assert env[key] == '1'
        assert env['NPU_CLS_TILE'] == '8192'
        inputs = list(map(int, Path(command[5]).read_text().split()))
        prompt = len(inputs)
        name = {1:'hello', 24:'chat24', 73:'ragged73', 128:'prompt128', 256:'prompt256'}[prompt]
        assert inputs == list(map(int, (roofline / 'final-quality' / f'{name}.tokens').read_text().split()))
        reference_log = roofline / 'final-quality' / f'{name}-custom.txt'
        golden = next(json.loads(s)['generated_ids'] for s in reference_log.read_text().splitlines()
                      if s.startswith('{') and json.loads(s).get('event') == 'run')[:16]
        assert list(map(int, Path(env['TEACHER_IDS']).read_text().split())) == golden
        log = out / f'{label}.jsonl'
        rows = [json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
        runs = [r for r in rows if r.get('event') == 'run']
        assert [r['run'] for r in runs] == [0, 1]
        assert [r['warmup'] for r in runs] == [True, False]
        pre, decode, head = env['DENSE_PREFILL'], env['DENSE_DECODE'], env['CPU_CLASSIFIER'] is not None
        route = next(r for r in rows if r.get('event') == 'routing')
        assert route['dense_prefill'] == pre and route['dense_decode'] == decode
        assert route['cpu_classifier'] == head and route['host_non_linear_ops'] == 'CPU'
        assert route['npu_initialized'] == ('npu' in [pre, decode])
        assert route['gpu_projections_initialized'] == ('gpu' in [pre, decode])
        init = next(r for r in rows if r.get('event') == 'initialization')
        assert init['weights'] == 'shared_FP16' and init['npu_cores'] == (3 if route['npu_initialized'] else 0)
        gpu_init = [r for r in rows if r.get('event') == 'linear_gpu_init']
        assert len(gpu_init) == int(route['gpu_projections_initialized'])
        if gpu_init:
            assert 'Mali' in gpu_init[0]['device'] and gpu_init[0]['weight_bytes'] == 1191968768
            assert gpu_init[0]['projection_count'] == 197
        attention = env['GPU_ATTENTION']
        attention_init = [r for r in rows if r.get('event') == 'gpu_initialization']
        assert len(attention_init) == int(attention is not None)
        if attention_init:
            assert 'Mali' in attention_init[0]['device'] and not attention_init[0]['event_profiling']
            assert attention_init[0]['prefill'] == (attention in ['both', 'prefill'])
            assert attention_init[0]['decode'] == (attention in ['both', 'decode'])
        cooling = [r for r in rows if r.get('event') == 'cooldown']
        assert [r['run'] for r in cooling] == [0, 1]
        assert all(r['after_millidegrees'] <= 55000 for r in cooling)
        for run in runs:
            assert run['prefill_tokens'] == prompt and run['new_tokens'] == 16 and run['decode_steps'] == 15
            index = run['run']
            for phase, device, steps in [('prefill', pre, 1), ('decode', decode, 15)]:
                counts = next(r for r in rows if r.get('event') == 'device_phase' and r['run'] == index and r['phase'] == phase)
                cpu = next(r for r in rows if r.get('event') == 'linear_cpu_profile' and r['run'] == index and r['phase'] == phase)
                expected_cpu = steps * (197 if device == 'cpu' else int(head))
                assert counts['cpu_ops'] == cpu['calls'] == expected_cpu
                gpu_pre = attention in ['prefill', 'both'] and prompt > 1
                expected_npu = (112 + int(not head)) * steps if device == 'npu' else 0
                if device == 'npu' and phase == 'prefill' and prompt > 1 and not gpu_pre:
                    expected_npu += 28 * 12 * math.ceil(prompt / 64)
                assert counts['npu_ops'] == expected_npu
                gpu = [r for r in rows if r.get('event') == 'linear_gpu_profile' and r['run'] == index and r['phase'] == phase]
                assert len(gpu) == int(route['gpu_projections_initialized'])
                if gpu:
                    expected_calls = steps * (197 - int(head)) if device == 'gpu' else 0
                    assert gpu[0]['calls'] == expected_calls
                    kernels = (196 + (0 if head else 2)) if phase == 'prefill' and prompt > 1 else 2 * expected_calls
                    assert gpu[0]['kernels'] == (kernels if device == 'gpu' else 0)
                    payload = steps * (1191968768 - (311164928 if head else 0)) if device == 'gpu' else 0
                    assert gpu[0]['weight_bytes'] == payload
                gpu_attention = [r for r in rows if r.get('event') == 'gpu_profile' and r['run'] == index and r['phase'] == phase]
                assert len(gpu_attention) == int(attention is not None)
                if gpu_attention:
                    profile = gpu_attention[0]
                    enabled = gpu_pre if phase == 'prefill' else attention in ['both', 'decode']
                    calls = 28 * steps if enabled else 0
                    assert profile['calls'] == calls
                    fused = 'decode_softmax_values' in profile['kernels']
                    assert profile['kernel_calls'] == calls * (4 if phase == 'prefill' else (2 if fused else 3))
                details = [r for r in rows if r.get('event') == 'cpu_attention_details' and r['run'] == index and r['phase'] == phase]
                if env.get('CPU_ATTN_BLAS') == '1':
                    assert len(details) == 1 and details[0]['calls'] == (28 if phase == 'prefill' and prompt > 1 else 0)
            assert run['cpu_matmul_ops'] == sum(r['cpu_ops'] for r in rows if r.get('event') == 'device_phase' and r['run'] == index)
            assert run['npu_ops'] == sum(r['npu_ops'] for r in rows if r.get('event') == 'device_phase' and r['run'] == index)
        a = np.fromfile(out / f'{label}.f32', np.float32).astype(np.float64)
        b = np.fromfile(roofline / 'final-quality' / f'{name}-custom.f32', np.float32).astype(np.float64)
        assert a.shape == b.shape == (151936,) and np.isfinite(a).all() and np.isfinite(b).all()
        relative = float(np.sqrt(np.sum((a-b)**2) / np.sum(b**2)))
        predictions = all(r['generated_ids'] == golden for r in runs)
        result = relative < .001 and predictions
        row = next(r for r in saved['records'] if r['prompt'] == prompt and r['route'] == config['route'])
        assert row['unchanged_limit'] == .001 and abs(row['logit_relative_rmse'] - relative) < 1e-12
        assert row['all_predictions_match'] == predictions and row['quality_passed'] == result
        assert row['golden_ids'] == golden and row['raw_log_sha256'] == sha(log)
        records.append({'directory':directory, 'label':label, 'prompt':prompt, 'route':config['route'],
            'relative_rmse':relative, 'unchanged_limit':.001, 'quality_passed':result,
            'all_predictions_match':predictions, 'first_logits_bit_identical':np.array_equal(a,b),
            'device_counts_checked_all_runs':True, 'same_teacher_ids':True,
            'binary_sha256':metadata['binary_sha256'], 'raw_log_sha256':sha(log),
            'logits_sha256':sha(out / f'{label}.f32'), 'config_sha256':sha(path)})

report = {'shared_fp16_model_sha256':model_sha, 'selected_binary_unchanged':True, 'records':records,
          'scope':'Additional numerical coverage only. Original logit limit 0.001 retained; all 16 predictions checked in the full warmup and measured request. No timings qualified as performance data.'}
(root / args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'audited_jobs':len(records), 'failed_logit_gates':sum(not r['quality_passed'] for r in records),
                  'all_predictions_match':all(r['all_predictions_match'] for r in records)}, indent=2))
