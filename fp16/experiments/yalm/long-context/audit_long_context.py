"""Independent raw-log, execution, numerical and clock audit of long contexts."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
import numpy as np
import concurrent_audit
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('directories',nargs='+');p.add_argument('--output',required=True);p.add_argument('--cool-c',type=int,default=51);args=p.parse_args()
parent=root/'e2e/long-context';matched=root.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((parent/'source-provenance.json').read_text())
assert sha(parent/'runq-routes')==meta['binary_sha256']
for name,digest in meta['source_sha256'].items():assert sha(parent/name)==digest
assert sha(Path('/home/orangepi/qwen3.c/runq-fp16'))==meta['selected_binary_sha256']
model=matched/'Qwen3-0.6B.fp16'
assert sha(model)=='c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879'
records=[];clock_audits={}
for directory in args.directories:
 folder=root/directory;summary=json.loads((folder/'summary.json').read_text())
 assert summary['binary_sha256']==sha(folder/'runq-routes')==meta['binary_sha256']
 for stem in ['clock','gpu-clock']:
  assert (folder/f'{stem}-state-before.json').read_bytes()==(folder/f'{stem}-state-restored.json').read_bytes()
 clocks=[json.loads(s) for s in (folder/'clock-samples.jsonl').read_text().splitlines()]
 targets={'cpu4_khz':2256000,'cpu6_khz':2256000,'npu_hz':1000000000,'gpu_hz':1000000000,'ddr_hz':2112000000}
 held=all(c[k]==v for c in clocks for k,v in targets.items())
 clock_audits[directory]={'samples':len(clocks),'targets_held':held,'snapshots_restored':True,
                         'temperature_max_c':max(c['temperature_millidegrees'] for c in clocks)/1000}
 for expected in summary['records']:
  label=expected['label'];prompt=expected['prompt'];route=expected['route'];steps=31
  config=json.loads((folder/(label+'.config.json')).read_text());env=config['environment']
  assert config['binary_sha256']==meta['binary_sha256'] and config['route']==route
  assert env['BENCH_CONTEXT']=='4128' and env['WARMUP_RUNS']=='2' and env['OMP_NUM_THREADS']=='4'
  if expected['prompt']>512:assert all(c['fan_pwm']==255 for c in clocks)
  assert env['COOL_REQUEST_C']==str(args.cool_c) and env['NPU_CORES']=='3' and env['NPU_DOMAIN_ID']=='1'
  assert all(env[k] is None for k in ['NPU_PROFILE','GPU_PROFILE','LINEAR_PROFILE','FFN_PROFILE'])
  ids=list(map(int,(folder/(label+'.tokens')).read_text().split()));assert len(ids)==prompt
  rows=[json.loads(s) for s in (folder/(label+'.jsonl')).read_text().splitlines() if s.startswith('{')]
  ctx=[r for r in rows if r.get('event')=='runtime_context'];assert len(ctx)==1 and ctx[0]['capacity']==4128 and ctx[0]['checkpoint_context']==512
  runs=[r for r in rows if r.get('event')=='run'];requests=[r for r in rows if r.get('event')=='request_wall']
  assert len(runs)==len(requests)==4 and [r['run'] for r in runs]==[-1,0,1,2]
  reference=not expected['teacher_forced_identical_decode_inputs']
  if prompt>512:
   golden_folder=parent/config.get('golden_directory','references');golden_label=f'p{prompt}-1-npu'
  else:
   golden_folder=matched/'roofline/final-quality';golden_label=f'prompt{prompt}-custom'
  golden_log=golden_folder/(golden_label+'.jsonl' if prompt>512 else golden_label+'.txt')
  if prompt<=512:
   golden_log=matched/'roofline/full-profile/measurements'/('p128-1-reference.jsonl' if prompt==128 else 'p256-2-reference.jsonl')
  golden=next(json.loads(s)['generated_ids'] for s in golden_log.read_text().splitlines() if s.startswith('{') and json.loads(s).get('event')=='run')[:32]
  if not reference:assert list(map(int,(folder/(label+'.teacher.tokens')).read_text().split()))==golden
  predictions=all(r['generated_ids']==golden for r in runs)
  for run,request in zip(runs,requests):
   index=run['run'];assert request['run']==index and request['warmup']==run['warmup']==(index<=0)
   assert run['prefill_tokens']==request['prompt_tokens']==prompt and run['new_tokens']==request['output_tokens']==32 and run['decode_steps']==steps
   assert abs(request['request_ms']-request['prefill_ms']-request['decode_ms']-request['handoff_ms'])<.00001
   assert abs(request['prefill_ms']-run['first_token_ms'])<=.000501
   assert abs(request['decode_ms']-run['decode_ms'])<=.000501
   assert abs(steps*1000/run['decode_ms']-run['decode_tps'])<.000501
   cooldown=next(r for r in rows if r.get('event')=='cooldown' and r['run']==index)
   assert cooldown['after_millidegrees']<=args.cool_c*1000
   for phase in ['prefill','decode']:
    actual=next(r for r in rows if r.get('event')=='device_phase' and r['run']==index and r['phase']==phase)
    n=1 if phase=='prefill' else steps
    if route in ['cpu','gpu']:
     assert actual['npu_ops']==0 and actual['cpu_ops']==197*n*int(route=='cpu')
     event='linear_cpu_profile' if route=='cpu' else 'linear_gpu_profile'
     prof=next(r for r in rows if r.get('event')==event and r['run']==index and r['phase']==phase)
     assert prof['calls']==197*n
     if route=='cpu':assert prof['prefill_attention_calls']==28*int(phase=='prefill')
     else:
      attn=next(r for r in rows if r.get('event')=='gpu_profile' and r['run']==index and r['phase']==phase)
      assert attn['calls']==28*n and attn['kernel_calls']==(112 if phase=='prefill' else 84*steps)
    else:
     padded=(prompt+31)&~31;tile=min(64,4*32768//(padded*2))
     expected_npu=28*(4+12*math.ceil(prompt/tile))+1 if phase=='prefill' else 113*steps
     assert actual['npu_ops']==expected_npu,(label,phase,actual,expected_npu)
     cpu_calls=28*steps*int(route in ['cpu_npu_dec','all_dec']) if phase=='decode' else 0
     assert actual['cpu_ops']==cpu_calls
     if route!='npu':concurrent_audit.phase(rows,env,index,phase,prompt,steps)
  a=np.fromfile(folder/(label+'.f32'),np.float32).astype(np.float64)
  b=np.fromfile(golden_folder/(golden_label+'.f32'),np.float32).astype(np.float64)
  assert a.shape==b.shape==(151936,) and np.isfinite(a).all() and np.isfinite(b).all()
  relative=float(np.linalg.norm(a-b)/np.linalg.norm(b));quality=relative<.001 and predictions
  assert abs(relative-expected['prefill_logit_relative_rmse'])<1e-12
  assert quality==expected['quality_passed']
  measured=[r for r in runs if not r['warmup']];timed=[r for r in requests if not r['warmup']]
  ttft=statistics.median(r['first_token_ms'] for r in measured);decode=statistics.median(r['decode_tps'] for r in measured);wall=statistics.median(r['request_ms'] for r in timed)
  assert all(abs(x-y)<1e-8 for x,y in [(ttft,expected['ttft_ms']),(decode,expected['decode_tps']),(wall,expected['request_ms'])])
  records.append({'directory':directory,'label':label,'prompt':prompt,'route':route,'reference':reference,'quality_passed':quality,
                  'relative_rmse':relative,'all_predictions_match':predictions,'clocks_held':held,'device_counts_checked':True,
                  'prefill_ms':ttft,'prefill_tps':prompt*1000/ttft,'decode_tps':decode,'request_ms':wall,
                  'prefill_range':[min(r['first_token_ms'] for r in measured),max(r['first_token_ms'] for r in measured)],
                  'decode_range':[min(r['decode_tps'] for r in measured),max(r['decode_tps'] for r in measured)],
                  'request_range':[min(r['request_ms'] for r in timed),max(r['request_ms'] for r in timed)],
                  'raw_log_sha256':sha(folder/(label+'.jsonl')),'prompt_sha256':sha(folder/(label+'.tokens'))})
out={'cooldown_target_c':args.cool_c,'model_sha256':sha(model),'binary_sha256':meta['binary_sha256'],'selected_binary_unchanged':True,'records':records,'clock_audits':clock_audits,
     'scope':'Independent audit. Long reference is this checked native NPU path, not an independent HF full-model oracle. Exact FP16 primitive checks and short-prompt parity are separate. Failed rows remain diagnostics at the unchanged0.001 gate.'}
(root/args.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'jobs':len(records),'quality_failed':[r['label'] for r in records if not r['quality_passed']],
                  'clock_failed':[k for k,v in clock_audits.items() if not v['targets_held']]},indent=2))
