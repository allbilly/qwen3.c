#!/usr/bin/env python3
"""Benchmark complete Qwen3 requests with CPU/GPU/ANE Core ML placements.

All placements use the same converted decoder, CPU embedding/head, prompt IDs,
and teacher-forced decode inputs. Predictions are synchronous. Validation runs
outside timing, and CPU/GPU/ANE device preferences are recorded per operation.
"""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from .common import Checkpoint, ROOT, load_prompts, quality, sha256, source_hashes, write_json

MODES = {'cpu': 'CPU_ONLY', 'gpu': 'CPU_AND_GPU', 'ane': 'CPU_AND_NE'}


def host_snapshot():
    result = dict(platform=platform.platform(), machine=platform.machine(),
                  load=list(os.getloadavg()), time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if sys.platform == 'darwin':
        for name, command in [('hardware', ['sysctl', 'machdep.cpu.brand_string', 'hw.model', 'hw.ncpu']),
                              ('software', ['sw_vers']),
                              ('memory', ['sysctl', 'hw.memsize', 'vm.swapusage']),
                              ('paging', ['vm_stat']), ('thermal', ['pmset', '-g', 'therm'])]:
            try:
                result[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=5)
            except (OSError, subprocess.SubprocessError) as error:
                result[name] = str(error)
    return result


class Engine:
    def __init__(self, directory, mode, checkpoint=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        self.config = self.manifest['config']
        self.capacity = self.manifest['cache_capacity']
        self.batch = self.manifest['batch']
        self.mode = mode
        self.embedding = np.load(self.directory/'embedding.npy', mmap_mode='r')
        head_name = 'embedding.npy' if self.config['tied'] else 'head.npy'
        # A resident FP32 copy of the same FP16 coefficients; no expansion in timers.
        self.head = np.load(self.directory/head_name, mmap_mode='r').astype('f4')
        self.final_norm = np.load(self.directory/'final_norm.npy')
        self.models, self.plans = {}, []
        self.blocks = list({a['start']:a for a in reversed(self.manifest['artifacts'])}.values())
        self.blocks.sort(key=lambda a:a['start'])
        if mode == 'reference':
            import torch
            from .model import Block
            torch.set_num_threads(4)
            source = Checkpoint(checkpoint)
            for a in self.blocks:
                self.models[(a['start'], 0)] = Block(source, a['start'], a['stop']).eval()
        else:
            import coremltools as ct
            units = getattr(ct.ComputeUnit, MODES[mode])
            self.plan_directory = tempfile.TemporaryDirectory(prefix='qwen3-compute-plan-')
            plan_binary = Path(self.plan_directory.name)/'compute-plan'
            subprocess.run(['swiftc', '-O', '-parse-as-library', str(ROOT/'macos/compute_plan.swift'),
                            '-framework', 'CoreML', '-o', str(plan_binary)], check=True)
            for a in self.manifest['artifacts']:
                if a['width'] != self.batch:
                    continue
                model = ct.models.MLModel(str(self.directory/a['package']), compute_units=units,
                                         function_name=a.get('function'))
                self.models[(a['start'], a['width'])] = model
                plan = json.loads(subprocess.check_output([str(plan_binary), model.get_compiled_model_path(),
                                                           mode, a.get('function', 'main')], text=True))
                self.plans.append(dict(package=a['package'], **plan))
            if mode in ('ane', 'gpu'):
                target = 'MLNeuralEngineComputeDevice' if mode == 'ane' else 'MLGPUComputeDevice'
                if not self.plans or any(not p['preferred_operation_counts'].get(target, 0) for p in self.plans):
                    raise RuntimeError(f'{mode} has a decoder block without preferred target-device operations')
        self.reset()

    def reset(self):
        c = self.config
        self.keys = np.zeros((c['layers'], 1, c['kv_heads'], self.capacity, c['head_dim']), 'f4')
        self.values = np.zeros_like(self.keys)

    def forward(self, tokens, start, width):
        from .model import inputs
        if not tokens or len(tokens) > width or start + len(tokens) > self.manifest['context']:
            raise ValueError('invalid request size or position')
        if start + width > self.capacity:
            raise ValueError('padded request exceeds allocated cache capacity')
        hidden = np.zeros((1, width, self.config['dim']), 'f4')
        hidden[0, :len(tokens)] = self.embedding[tokens].astype('f4')
        auxiliary = inputs(self.config, width, start, self.capacity)
        for a in self.blocks:
            lo, hi = a['start'], a['stop']
            values = dict(hidden=hidden, keys=self.keys[lo:hi], values=self.values[lo:hi], **auxiliary)
            if self.mode == 'reference':
                import torch
                with torch.no_grad():
                    outputs = self.models[(lo, 0)](*(torch.from_numpy(v) for v in values.values()))
                hidden, keys, cache = (v.numpy() for v in outputs)
            else:
                result = self.models[(lo, width)].predict(values)
                hidden, keys, cache = result['hidden_out'], result['keys_out'], result['values_out']
            self.keys[lo:hi], self.values[lo:hi] = keys, cache
        last = hidden[0, len(tokens)-1].astype('f4')
        last = last / np.sqrt(np.mean(last * last) + np.float32(1e-6)) * self.final_norm
        logits = self.head @ last
        if not np.isfinite(logits).all():
            raise RuntimeError('nonfinite logits')
        return logits

    def prefill(self, tokens):
        logits = None
        for start in range(0, len(tokens), self.batch):
            chunk = tokens[start:start+self.batch]
            logits = self.forward(chunk, start, self.batch)
        return logits

    def request(self, prompt, new_tokens, teachers=None, capture=False):
        self.reset()
        begin = time.perf_counter_ns()
        logits = self.prefill(prompt)
        ids = [int(logits.argmax())]
        prefill_ms = (time.perf_counter_ns()-begin)/1e6
        saved = [logits.copy()] if capture else None
        samples = []
        for step in range(1, new_tokens):
            token = teachers[step-1] if teachers is not None else ids[-1]
            begin = time.perf_counter_ns()
            logits = self.forward([token], len(prompt)+step-1, self.batch)
            ids.append(int(logits.argmax()))
            samples.append((time.perf_counter_ns()-begin)/1e6)
            if capture:
                saved.append(logits.copy())
        decode_ms = sum(samples)
        return dict(prefill_ms=prefill_ms, prefill_tps=len(prompt)*1000/prefill_ms,
                    decode_ms=decode_ms, decode_tps=(new_tokens-1)*1000/decode_ms,
                    completion_ms=prefill_ms+decode_ms, decode_step_ms=samples,
                    ids=ids), None if saved is None else np.stack(saved)


def worker(args):
    started = host_snapshot()
    begin = time.perf_counter()
    report = dict(mode=args.worker_mode, status='RUNNING', started=started,
                  compute_plans=[], prompts=[])
    engine = None
    try:
        engine = Engine.__new__(Engine)
        engine.__init__(args.model, args.worker_mode, args.checkpoint)
        report.update(initialization_ms=(time.perf_counter()-begin)*1000,
                      compute_plans=engine.plans, source_sha256=source_hashes(),
                      model_manifest_sha256=sha256(args.model/'manifest.json'),
                      packages={n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'coremltools')})
        write_json(args.output, report)
        _worker_prompts(args, engine, report)
        report['status'] = 'PASS' if all(p.get('qualified', True) for p in report['prompts']) else 'FAIL'
    except BaseException as error:
        report.update(status='ERROR', error=str(error), compute_plans=getattr(engine, 'plans', []))
        raise
    finally:
        try:
            report['finished'] = host_snapshot()
            write_json(args.output, report)
        finally:
            del engine
            gc.collect()


def _worker_prompts(args, engine, report):
    prompts = load_prompts(args.prompts, engine.config['vocab'], engine.manifest['context'], args.new_tokens)
    if args.worker_mode == 'reference':
        for prompt in prompts:
            record, logits = engine.request(prompt['ids'], args.new_tokens, capture=True)
            name = prompt['name']
            np.save(args.output.parent/(name+'.npy'), logits)
            report['prompts'].append(dict(name=name, ids=prompt['ids'], teacher_ids=record['ids'],
                                           logits_file=name+'.npy', logits_sha256=sha256(args.output.parent/(name+'.npy'))))
            write_json(args.output, report)
    else:
        reference = json.loads((args.reference/'reference.json').read_text())
        if len(reference['prompts']) != len(prompts):
            raise ValueError('reference prompt count mismatch')
        for prompt, ref in zip(prompts, reference['prompts']):
            if prompt['name'] != ref['name'] or prompt['ids'] != ref['ids']:
                raise ValueError('reference prompt identity mismatch')
            teachers = ref['teacher_ids']
            if len(teachers) != args.new_tokens or sha256(args.reference/ref['logits_file']) != ref['logits_sha256']:
                raise ValueError('reference token count or logit hash mismatch')
            _, actual = engine.request(prompt['ids'], args.new_tokens, teachers, capture=True)
            expected = np.load(args.reference/ref['logits_file'])
            if expected.shape != actual.shape:
                raise ValueError('reference logit shape mismatch')
            checks = [quality(r, a) for r,a in zip(expected, actual)]
            logit_path = args.output.with_name(args.output.stem+'-'+prompt['name']+'.npy')
            np.save(logit_path, actual)
            passed = all(c['nrmse'] <= args.max_nrmse and c['max_kl'] <= args.max_kl and
                         c['top1_mismatches'] == 0 for c in checks)
            warmups = []
            for _ in range(args.warmups):
                record, _ = engine.request(prompt['ids'], args.new_tokens, teachers)
                warmups.append(record)
                if record['ids'] != teachers:
                    passed = False
            record, _ = engine.request(prompt['ids'], args.new_tokens, teachers)
            record['timed_predictions_match_reference'] = record['ids'] == teachers
            if not record['timed_predictions_match_reference']:
                passed = False
            report['prompts'].append(dict(name=prompt['name'], prompt_ids=prompt['ids'],
                                           teacher_ids=teachers, qualified=passed, accuracy=checks,
                                           validation_logits=logit_path.name, validation_logits_sha256=sha256(logit_path),
                                           warmups=warmups, measurement=record))
            write_json(args.output, report)
            print(f"{args.worker_mode} {prompt['name']}: prefill {record['prefill_ms']:.1f} ms, "
                  f"decode {record['decode_tps']:.2f} tok/s, quality {'PASS' if passed else 'FAIL'}", flush=True)


def verify_artifacts(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['protocol'] != 'qwen3-coreml-v1':
        raise ValueError('unsupported model manifest')
    for name, digest in manifest['file_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f'model artifact hash mismatch: {name}')
    return manifest


def run(args):
    if sys.platform != 'darwin':
        raise RuntimeError('Core ML hardware benchmarks require macOS')
    if args.output.exists():
        raise ValueError('output already exists; use a new result directory')
    if args.trials < 1 or args.warmups < 1 or args.new_tokens < 2 or any(
            not math.isfinite(limit) or limit <= 0 for limit in (args.max_nrmse, args.max_kl)):
        raise ValueError('invalid trial, warmup, token count or numerical limit')
    if not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError('choose at least one mode, without duplicates')
    manifest = verify_artifacts(args.model)
    if sha256(args.checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('reference checkpoint hash differs from converted checkpoint')
    prompts = load_prompts(args.prompts, manifest['config']['vocab'], manifest['context'], args.new_tokens)
    args.output.mkdir(parents=True)
    reference = args.output/'reference'
    reference.mkdir()
    report = dict(protocol='qwen3-coreml-v1', status='RUNNING', model_manifest=manifest,
                  model_manifest_sha256=sha256(args.model/'manifest.json'),
                  prompts=prompts, new_tokens=args.new_tokens, decode_steps=args.new_tokens-1,
                  physical_prefill_width=manifest['batch'], physical_decode_width=manifest['batch'],
                  trials_per_mode=args.trials, warmups_per_worker_prompt=args.warmups,
                  modes=args.modes,
                  numerical_gate=dict(max_nrmse=args.max_nrmse, max_kl=args.max_kl, top1_mismatches=0),
                  source_sha256=source_hashes(),
                  started=host_snapshot(), jobs=[], summaries={},
                  timing_scope='Synchronous complete model request: CPU embedding, all Core ML decoder blocks, '
                               'explicit KV cache handoff, CPU final norm/head, finite check and argmax. '
                               'Prefill returns first output; decode has new_tokens-1 subsequent calls. '
                               'Excludes initialization, compilation, cache reset, numerical validation, tokenization and detokenization.',
                  placement_scope='Compute plans report preferred devices, not runtime utilization or exclusive execution.',
                  comparison_scope='Same checkpoint, FP16 coefficient policy, graph, context, inputs and output count; '
                                   'different permitted Core ML compute devices. Separate from Asahi Q8 activation arithmetic.')
    destination = args.output/'results.json'
    write_json(destination, report)
    locks = ExitStack()

    def launch(mode, output):
        command = [sys.executable, '-u', '-m', 'macos.benchmark', '--worker-mode', mode,
                   '--model', str(args.model.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                   '--prompts', str(args.prompts.resolve()), '--new-tokens', str(args.new_tokens),
                   '--warmups', str(args.warmups), '--max-nrmse', str(args.max_nrmse), '--max-kl', str(args.max_kl),
                   '--reference', str(reference.resolve()), '--output', str(output.resolve())]
        row = dict(mode=mode, output=str(output.relative_to(args.output)), command=command,
                   before=host_snapshot())
        report['jobs'].append(row)
        write_json(destination, report)
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        logpath = output.with_suffix('.log')
        with logpath.open('w') as log:
            row['returncode'] = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        row.update(after=host_snapshot(), log=str(logpath.relative_to(args.output)))
        if output.exists():
            row['output_sha256'] = sha256(output)
        write_json(destination, report)
        if row['returncode']:
            print(logpath.read_text()[-4000:], file=sys.stderr)
            return None
        receipt = json.loads(output.read_text())
        if receipt['source_sha256'] != report['source_sha256'] or receipt['model_manifest_sha256'] != report['model_manifest_sha256']:
            raise RuntimeError('worker source or model identity changed during the run')
        return receipt

    try:
        from tools.benchmark_queue import competing_jobs
        for name in ('ane.lock', 'gpu.lock'):
            stream = locks.enter_context((Path.home()/name).open('a'))
            print(f'Waiting for {stream.name}', flush=True)
            fcntl.flock(stream, fcntl.LOCK_EX)
        while jobs := competing_jobs():
            print(f'Queued behind {jobs}', flush=True)
            time.sleep(2)
        if launch('reference', reference/'reference.json') is None:
            raise RuntimeError('reference preparation failed; log retained')
        measurements = {mode: [] for mode in args.modes}
        for trial in range(args.trials):
            modes = list(args.modes)
            modes = modes[trial % len(modes):] + modes[:trial % len(modes)]
            for mode in modes:
                receipt = launch(mode, args.output/f'{mode}-{trial+1}.json')
                if receipt is not None:
                    measurements[mode].append(receipt)
                    print(f'{mode} trial {trial+1}: {receipt["status"]}', flush=True)
        complete = True
        for mode, receipts in measurements.items():
            rows = []
            if len(receipts) != args.trials or any(r['status'] != 'PASS' for r in receipts):
                complete = False
            for i, prompt in enumerate(prompts):
                samples = [r['prompts'][i]['measurement'] for r in receipts]
                rows.append(dict(name=prompt['name'], prompt_tokens=len(prompt['ids']),
                                 qualified=len(receipts)==args.trials and all(r['prompts'][i]['qualified'] for r in receipts),
                                 median={key:statistics.median(s[key] for s in samples) for key in
                                         ('prefill_ms','prefill_tps','decode_ms','decode_tps','completion_ms')} if samples else {},
                                 samples=samples))
            report['summaries'][mode] = rows
        report['status'] = 'PASS' if complete else 'FAIL'
    except BaseException as error:
        report.update(status='ERROR', error=str(error))
        raise
    finally:
        try:
            report['finished'] = host_snapshot()
            write_json(destination, report)
        finally:
            locks.close()
    return report['status'] == 'PASS'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model', type=Path, required=True)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--prompts', type=Path, default=ROOT/'ane/benchmarks/prompts.json')
    cli.add_argument('--new-tokens', type=int, default=16)
    cli.add_argument('--trials', type=int, default=3)
    cli.add_argument('--warmups', type=int, default=2)
    cli.add_argument('--modes', nargs='+', choices=list(MODES), default=['gpu', 'ane'],
                     help='permitted compute devices to test (default: gpu ane; cpu is an optional control)')
    cli.add_argument('--max-nrmse', type=float, default=.005)
    cli.add_argument('--max-kl', type=float, default=.01)
    cli.add_argument('--worker-mode', choices=['reference', *MODES], help=argparse.SUPPRESS)
    cli.add_argument('--reference', type=Path, help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.worker_mode:
        worker(args)
        return 0
    return 0 if run(args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
