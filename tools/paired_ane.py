#!/usr/bin/env python3
"""Queue paired CPU/ANE full-model measurements and compare fixed-input logits.

Requires NumPy. Build runq-ane-bench first. Initialization and one full warmup
are excluded from timings. Every decode uses the same CPU-generated token IDs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_model(args, ids, teacher, dump, ane):
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'OMP_DYNAMIC': 'FALSE',
           'ANE': str(int(ane)), 'ANE_DECODE': str(int(args.ane_decode))}
    env.pop('ANE_SERIAL_PREFILL', None)
    command = ['taskset', '-c', args.cpus, str(ROOT / 'runq-ane-bench'),
               str(args.checkpoint), str(ids), str(args.new_tokens), '1', str(dump)]
    if teacher.exists():
        command.append(str(teacher))
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr or f'benchmark exited {process.returncode}')
    records = [json.loads(line) for line in process.stdout.splitlines()]
    init = next(record for record in records if record['event'] == 'init')
    measured = next(record for record in records if record['event'] == 'run' and not record['warmup'])
    measured['init_ms'] = init['init_ms']
    measured['completion_ms'] = measured['prefill_ms'] + measured['decode_ms']
    return measured


def worker(args):
    import numpy as np
    with args.checkpoint.open('rb') as f:
        header = struct.unpack('<12i', f.read(48))
        f.seek(0)
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    vocab = header[7]
    prompts = json.loads(args.prompts.read_text())
    report = {
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'host': {'system': platform.platform(), 'cpus': args.cpus, 'threads': args.threads},
        'model': {'checkpoint': args.checkpoint.name, 'sha256': digest, 'header': list(header)},
        'protocol': {'trials': args.trials, 'warmups_per_process': 1, 'new_tokens': args.new_tokens,
                     'order': 'CPU/ANE alternates per trial', 'decode_inputs': 'fixed CPU-generated tokens',
                     'initialization_included': False, 'ane_decode': args.ane_decode,
                     'locks': ['~/ane.lock', '~/gpu.lock'], 'min_logit_cosine': args.min_cosine},
        'prompts': [],
    }
    with tempfile.TemporaryDirectory(prefix='qwen3-paired-ane-') as directory:
        root = Path(directory)
        for prompt in prompts:
            ids = root / 'prompt.ids'
            ids.write_text(' '.join(map(str, prompt['ids'])))
            teacher = root / 'teacher.ids'
            teacher.unlink(missing_ok=True)
            dumps = {backend: root / f'{backend}.bin' for backend in ('cpu', 'ane')}
            samples = {'cpu': [], 'ane': []}
            accuracy = None
            for trial in range(args.trials):
                order = ('cpu', 'ane') if trial % 2 == 0 else ('ane', 'cpu')
                for backend in order:
                    record = run_model(args, ids, teacher, dumps[backend], backend == 'ane')
                    if samples[backend] and record['ids'] != samples[backend][0]['ids']:
                        raise RuntimeError(f"{prompt['name']}: nondeterministic {backend} token choices")
                    if backend == 'ane' and len(prompt['ids']) >= 16 and record['ane_ops'] == 0:
                        raise RuntimeError('requested ANE prefill performed no ANE projections')
                    samples[backend].append(record)
                    if not teacher.exists():
                        teacher.write_text(' '.join(map(str, record['ids'])))
                if trial == 0:
                    cpu = np.fromfile(dumps['cpu'], dtype='<f4').reshape(args.new_tokens, vocab).astype('f8')
                    ane = np.fromfile(dumps['ane'], dtype='<f4').reshape(args.new_tokens, vocab).astype('f8')
                    if not np.isfinite(cpu).all() or not np.isfinite(ane).all():
                        raise RuntimeError('nonfinite full-model logits')
                    cosine = np.sum(cpu * ane, axis=1) / np.sqrt(np.sum(cpu * cpu, axis=1) * np.sum(ane * ane, axis=1))
                    accuracy = {'min_cosine': float(cosine.min()), 'cosine_per_step': cosine.tolist(),
                                'max_abs_error': float(np.max(np.abs(cpu - ane))),
                                'top1_matches': int(np.sum(cpu.argmax(1) == ane.argmax(1))),
                                'steps': args.new_tokens, 'teacher_ids': [int(token) for token in teacher.read_text().split()]}
                    if accuracy['min_cosine'] < args.min_cosine:
                        raise RuntimeError(f"{prompt['name']}: logit cosine {accuracy['min_cosine']} below {args.min_cosine}")
            summary = {}
            for backend, records in samples.items():
                summary[backend] = {}
                for metric in ('prefill_ms', 'decode_tps', 'completion_ms', 'init_ms'):
                    values = [record[metric] for record in records]
                    summary[backend][metric] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
            summary['prefill_speedup'] = summary['cpu']['prefill_ms']['median'] / summary['ane']['prefill_ms']['median']
            summary['completion_speedup'] = summary['cpu']['completion_ms']['median'] / summary['ane']['completion_ms']['median']
            report['prompts'].append({**prompt, 'accuracy': accuracy, 'summary': summary, 'samples': samples})
            print(f"{prompt['name']}: prefill {summary['prefill_speedup']:.2f}x; cosine {accuracy['min_cosine']:.6f}; "
                  f"top-1 {accuracy['top1_matches']}/{args.new_tokens}", file=sys.stderr, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('--prompts', type=Path, default=ROOT / 'ane/benchmarks/prompts.json')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--trials', type=int, default=5)
    cli.add_argument('--new-tokens', type=int, default=16)
    cli.add_argument('--threads', type=int, default=4)
    cli.add_argument('--cpus', default='4-7')
    cli.add_argument('--ane-decode', action='store_true')
    cli.add_argument('--min-cosine', type=float, default=0.995)
    cli.add_argument('--queued', action='store_true', help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.trials < 1 or not 2 <= args.new_tokens <= 512 or args.threads < 1:
        cli.error('invalid trial, token, or thread count')
    if not args.queued:
        return subprocess.call([sys.executable, str(ROOT / 'tools/benchmark_queue.py'), '--',
                                sys.executable, str(Path(__file__).resolve()), '--queued', *sys.argv[1:]])
    worker(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
