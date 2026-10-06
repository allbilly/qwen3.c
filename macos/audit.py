#!/usr/bin/env python3
"""Independently recompute benchmark quality gates and timing summaries."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .common import quality, sha256, write_json


def local_path(directory, name):
    path = (directory/name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('receipt path escapes the result directory')
    return path


def check_timing(measurement, prompt, teachers, report, timed=True):
    fields = ('prefill_ms', 'prefill_tps', 'decode_ms', 'decode_tps', 'completion_ms')
    if any(not math.isfinite(measurement[k]) or measurement[k] <= 0 for k in fields):
        raise ValueError('timings and throughput must be finite and positive')
    steps = measurement['decode_step_ms']
    count = report['decode_steps']
    if len(steps) != count or count != report['new_tokens']-1:
        raise ValueError('decode timing boundary mismatch')
    if any(not math.isfinite(t) or t <= 0 for t in steps):
        raise ValueError('decode step must be finite and positive')
    decode_ms = sum(steps)
    if decode_ms != measurement['decode_ms'] or count*1000/decode_ms != measurement['decode_tps']:
        raise ValueError('decode total or throughput mismatch')
    if len(prompt)*1000/measurement['prefill_ms'] != measurement['prefill_tps']:
        raise ValueError('prefill throughput mismatch')
    if measurement['ids'] != teachers or (timed and not measurement['timed_predictions_match_reference']):
        raise ValueError('token choices differ')
    if measurement['completion_ms'] != measurement['prefill_ms']+decode_ms:
        raise ValueError('completion timing mismatch')


def audit(directory):
    report = json.loads((directory/'results.json').read_text())
    if report['status'] != 'PASS':
        raise ValueError('the benchmark did not pass all configurations')
    # The exporter writes this exact JSON representation via common.write_json.
    manifest = report['model_manifest']
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if hashlib.sha256(manifest_bytes).hexdigest() != report['model_manifest_sha256']:
        raise ValueError('embedded model manifest hash mismatch')
    if any(report[k] != manifest['batch'] for k in ('physical_prefill_width', 'physical_decode_width')):
        raise ValueError('physical input widths differ from the model manifest')
    expected_modes = report.get('modes', ['cpu','gpu','ane'])
    if not expected_modes or len(set(expected_modes)) != len(expected_modes) or not set(expected_modes) <= {'cpu','gpu','ane'}:
        raise ValueError('invalid benchmark modes')
    if report['trials_per_mode'] < 1 or report['warmups_per_worker_prompt'] < 1 or report['new_tokens'] < 2:
        raise ValueError('invalid trial, warmup or token count')
    if set(report['summaries']) != set(expected_modes):
        raise ValueError('summary modes differ')
    jobs = report['jobs']
    if Counter(j['mode'] for j in jobs) != Counter({'reference':1, **{m:report['trials_per_mode'] for m in expected_modes}}):
        raise ValueError('job counts differ')
    if len({j['output'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate worker receipt')
    reference_job = next(j for j in jobs if j['mode']=='reference')
    reference_path = local_path(directory, reference_job['output'])
    if sha256(reference_path) != reference_job['output_sha256']:
        raise ValueError('reference receipt hash mismatch')
    reference = json.loads(reference_path.read_text())
    if reference_job['returncode'] != 0 or reference['status'] != 'PASS' or reference['mode'] != 'reference':
        raise ValueError('reference failed')
    if reference['source_sha256'] != report['source_sha256'] or reference['model_manifest_sha256'] != report['model_manifest_sha256']:
        raise ValueError('reference provenance differs')
    if len(reference['prompts']) != len(report['prompts']) or not report['prompts']:
        raise ValueError('reference prompt count mismatch')
    gate = report['numerical_gate']
    if gate['top1_mismatches'] != 0 or any(not math.isfinite(gate[k]) or gate[k] <= 0 for k in ('max_nrmse', 'max_kl')):
        raise ValueError('invalid numerical gates')
    receipts = {mode: [] for mode in expected_modes}
    checked_logits = 0
    checked_tokens = 0
    for job in jobs:
        if job['mode'] == 'reference':
            continue
        path = local_path(directory, job['output'])
        if job['returncode'] != 0 or sha256(path) != job['output_sha256']:
            raise ValueError('worker failed or receipt hash mismatch')
        worker = json.loads(path.read_text())
        if worker['status'] != 'PASS' or worker['mode'] != job['mode']:
            raise ValueError('worker mode or status mismatch')
        if worker['source_sha256'] != report['source_sha256'] or worker['model_manifest_sha256'] != report['model_manifest_sha256']:
            raise ValueError('worker provenance differs')
        if len(worker['prompts']) != len(report['prompts']):
            raise ValueError('worker prompt count mismatch')
        for expected, row, ref in zip(report['prompts'], worker['prompts'], reference['prompts']):
            if ref['name'] != expected['name'] or ref['ids'] != expected['ids'] or row['name'] != expected['name'] or row['prompt_ids'] != expected['ids'] or row['teacher_ids'] != ref['teacher_ids']:
                raise ValueError('prompt or teacher-input identity differs')
            if len(ref['teacher_ids']) != report['new_tokens']:
                raise ValueError('reference output count differs')
            ref_path = local_path(reference_path.parent, ref['logits_file'])
            actual_path = local_path(directory, row['validation_logits'])
            if sha256(ref_path) != ref['logits_sha256'] or sha256(actual_path) != row['validation_logits_sha256']:
                raise ValueError('validation logit hash mismatch')
            original, actual = np.load(ref_path), np.load(actual_path)
            if original.shape != actual.shape or actual.shape != (report['new_tokens'], report['model_manifest']['config']['vocab']):
                raise ValueError('validation logit shape mismatch')
            if original.argmax(-1).tolist() != ref['teacher_ids']:
                raise ValueError('reference teachers differ from reference logits')
            checks = [quality(r,a) for r,a in zip(original,actual)]
            if checks != row['accuracy'] or not row['qualified']:
                raise ValueError('recorded numerical checks differ from recomputed values')
            if any(c['nrmse']>gate['max_nrmse'] or c['max_kl']>gate['max_kl'] or c['top1_mismatches'] for c in checks):
                raise ValueError('numerical gate failed')
            measurement = row['measurement']
            check_timing(measurement, expected['ids'], ref['teacher_ids'], report)
            if len(row['warmups']) != report['warmups_per_worker_prompt']:
                raise ValueError('warmup count mismatch')
            for warmup in row['warmups']:
                check_timing(warmup, expected['ids'], ref['teacher_ids'], report, timed=False)
            checked_logits += len(checks)
            checked_tokens += len(measurement['ids'])
        target = {'gpu':'MLGPUComputeDevice','ane':'MLNeuralEngineComputeDevice'}.get(job['mode'])
        artifacts = [a for a in report['model_manifest']['artifacts'] if a['width'] == report['physical_decode_width']]
        if [(p['package'],p['function']) for p in worker['compute_plans']] != [(a['package'],a['function']) for a in artifacts]:
            raise ValueError('decoder compute plans differ')
        for plan in worker['compute_plans']:
            if dict(Counter(op['preferred'] for op in plan['operations'])) != plan['preferred_operation_counts']:
                raise ValueError('compute-plan counts differ')
        if target:
            if not worker['compute_plans'] or any(not p['preferred_operation_counts'].get(target,0) for p in worker['compute_plans']):
                raise ValueError('target device has no preferred operations')
        receipts[job['mode']].append(worker)
    for mode, workers in receipts.items():
        if len(workers) != report['trials_per_mode']:
            raise ValueError('trial count mismatch')
        if len(report['summaries'][mode]) != len(report['prompts']):
            raise ValueError('summary prompt count mismatch')
        for i,row in enumerate(report['summaries'][mode]):
            samples = [w['prompts'][i]['measurement'] for w in workers]
            if samples != row['samples'] or not row['qualified']:
                raise ValueError('summary qualification or samples mismatch')
            if row['name'] != report['prompts'][i]['name'] or row['prompt_tokens'] != len(report['prompts'][i]['ids']):
                raise ValueError('summary prompt identity differs')
            medians = {key:statistics.median(s[key] for s in samples) for key in
                       ('prefill_ms','prefill_tps','decode_ms','decode_tps','completion_ms')}
            if medians != row['median']:
                raise ValueError('summary median mismatch')
    return dict(status='PASS', report_sha256=sha256(directory/'results.json'),
                auditor_sha256=sha256(__file__),
                checked_logit_vectors=checked_logits, checked_timed_tokens=checked_tokens,
                workers=sum(len(v) for v in receipts.values()), medians_recomputed=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('audit output exists; use a new path')
    result = audit(args.directory)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
