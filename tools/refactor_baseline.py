#!/usr/bin/env python3
import argparse
from array import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import time

if __package__:
    from .benchmark_queue import run_owned
else:
    from benchmark_queue import run_owned


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(rows, logits, prompt, steps, trials, vocab, backend):
    if len(rows) != trials + 1 or [row['run'] for row in rows] != list(range(trials + 1)):
        raise ValueError('Incomplete warmup or measurement coverage')
    if [row['warmup'] for row in rows] != [True] + [False] * trials:
        raise ValueError('Warmup classification differs')
    expected = rows[0]['ids']
    if len(expected) != steps or any(not 0 <= value < vocab for value in expected):
        raise ValueError('Invalid generated token coverage')
    for row in rows:
        if row['ids'] != expected or row['prompt_tokens'] != len(prompt) or row['new_tokens'] != steps:
            raise ValueError('Inputs or generated token IDs differ')
        if any(not math.isfinite(row[key]) or row[key] <= 0 for key in ['prefill_ms','decode_ms','decode_tps','peak_rss_bytes']):
            raise ValueError('Invalid timing or memory evidence')
        if backend == 'cpu' and row['npu_ops'] != 0:
            raise ValueError('CPU control used an accelerator')
        if backend != 'cpu' and len(prompt) >= 16 and row['npu_ops'] <= 0:
            raise ValueError('Requested accelerator performed no projections')
    if len(logits) != steps * vocab * 4:
        raise ValueError('Incomplete logit vectors')
    values = array('f')
    values.frombytes(logits)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite baseline logits')
    for step, token in enumerate(expected):
        row = values[step*vocab:(step+1)*vocab]
        if max(range(vocab), key=row.__getitem__) != token:
            raise ValueError('Logit argmax differs from generated token')
    return expected


def main():
    cli = argparse.ArgumentParser(description='Record immutable Q8 model baselines before refactoring')
    cli.add_argument('--binary', type=Path, required=True)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--prompts', type=Path, default=Path(__file__).resolve().parents[1] / 'ane/benchmarks/prompts.json')
    cli.add_argument('--backend', choices=['cpu','rknpu','ane'], required=True)
    cli.add_argument('--trials', type=int, default=3)
    cli.add_argument('--tokens', type=int, default=16)
    cli.add_argument('--threads', type=int, default=4)
    cli.add_argument('--cpus', default='4-7')
    args = cli.parse_args()
    if args.trials < 1 or args.tokens < 2 or args.threads < 1:
        cli.error('Invalid trial/token/thread count')
    binary, checkpoint, out = [path.resolve() for path in [args.binary,args.checkpoint,args.out]]
    with checkpoint.open('rb') as source:
        header = struct.unpack('<12i', source.read(48))
    if header[0] != 0x616a6331 or header[1] != 1:
        cli.error('This collector requires the original Q8 version-1 checkpoint')
    vocab = header[7]
    prompts = json.loads(args.prompts.read_text())
    if (not prompts or len({row['name'] for row in prompts}) != len(prompts)
            or any(re.fullmatch(r'[A-Za-z0-9_-]+', row['name']) is None for row in prompts)):
        cli.error('Prompt matrix is empty or duplicated')
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ['runq.c','qwen3.h','npu_matmul.c','npu_matmul.h','ane/prefill.h','fp16/prefill.h','tests/refactor_bench.c']
    report = {'status':'RUNNING','backend':args.backend,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'host':platform.platform(),'machine':platform.machine(),'cpus':args.cpus,'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'binary_sha256':sha(binary),'header':list(header),
              'trials':args.trials,'output_tokens':args.tokens,'prompts':[],
              'source_sha256':{name:sha(root/name) for name in sources},
              'source_revision':subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()}
    manifest = out/'manifest.json'
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    environment = dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_PROC_BIND='close',
                       NPU='1' if args.backend=='rknpu' else '0',ANE='1' if args.backend=='ane' else '0')
    if args.backend=='rknpu':
        environment.pop('ANE')
    try:
        for prompt in prompts:
            if not prompt['ids'] or len(prompt['ids'])+args.tokens>header[8] or any(not 0<=token<vocab for token in prompt['ids']):
                raise ValueError('Prompt exceeds the selected checkpoint')
            directory = out/prompt['name']
            directory.mkdir()
            ids = directory/'prompt.ids';ids.write_text(' '.join(map(str,prompt['ids']))+'\n')
            logits = directory/'logits.bin'
            argv = ['taskset','-c',args.cpus,str(binary),str(checkpoint),str(ids),str(args.tokens),str(args.trials),str(logits)]
            started = time.monotonic()
            try:
                with (directory/'run.log').open('wb') as stream:
                    code = run_owned(argv,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
            except BaseException as error:
                (directory/'exit.json').write_text(json.dumps({'exit_code':getattr(error,'actual_returncode',None),
                    'timed_out':isinstance(error,subprocess.TimeoutExpired),'interruption':repr(error),
                    'elapsed_seconds':time.monotonic()-started})+'\n')
                raise
            (directory/'exit.json').write_text(json.dumps({'exit_code':code,
                'timed_out':False,'interruption':None,'elapsed_seconds':time.monotonic()-started})+'\n')
            if code:
                raise RuntimeError('Benchmark failed: '+str(directory/'run.log'))
            rows = [json.loads(line) for line in (directory/'run.log').read_text().splitlines() if line.startswith('{') and json.loads(line).get('event')=='run']
            tokens = validate(rows,logits.read_bytes(),prompt['ids'],args.tokens,args.trials,vocab,args.backend)
            measured = rows[1:]
            report['prompts'].append({'name':prompt['name'],'input_ids':prompt['ids'],'generated_ids':tokens,
                'rows':rows,'elapsed_seconds':time.monotonic()-started,'command':argv,
                'logits_sha256':sha(logits),'log_sha256':sha(directory/'run.log'),
                'median_prefill_ms':statistics.median(row['prefill_ms'] for row in measured),
                'median_decode_tps':statistics.median(row['decode_tps'] for row in measured),
                'peak_rss_bytes':max(row['peak_rss_bytes'] for row in rows)})
            manifest.write_text(json.dumps(report,indent=2)+'\n')
            print(prompt['name']+' baseline saved',flush=True)
        if (sha(checkpoint)!=report['checkpoint_sha256'] or sha(binary)!=report['binary_sha256']
                or any(sha(root/name)!=value for name,value in report['source_sha256'].items())):
            raise ValueError('Baseline inputs changed')
        report['status']='BASELINE_CAPTURE_PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        manifest.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
