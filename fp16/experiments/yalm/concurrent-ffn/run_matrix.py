"""Explicit phase placements, same FP16 weights and teacher-forced decode inputs."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,statistics,subprocess,threading,time
import numpy as np
root=Path(__file__).resolve().parent;roofline=root.parents[2];matched=roofline.parent
parser=argparse.ArgumentParser();parser.add_argument('--output',default='matrix');parser.add_argument('--routes',nargs='*')
parser.add_argument('--prompts',nargs='+',type=int,default=[128,256]);parser.add_argument('--new-tokens',type=int,default=32)
parser.add_argument('--runs',type=int,default=2);parser.add_argument('--profile',action='store_true');parser.add_argument('--cool-c',type=int,default=50);args=parser.parse_args()
settings={f'{route}_{phase}':(cpu,gpu,mode) for route,cpu,gpu in [('cpu_npu',96,0),('gpu_npu',0,96),('all',96,96)] for phase,mode in [('pre',1),('dec',2),('both',3)]}
placements={r:('npu','npu',None,False) for r in ['npu',*settings]}
order=args.routes or ['npu',*settings]
assert all(r in placements for r in order)
out=root/args.output;out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=root/'runq-routes';shutil.copy2(binary,out/'runq-routes')
for name in ['run_matrix.py','source-provenance.json']:shutil.copy2(root/name,out/name)
for stem in ['cpu','gpu']:
 checks=[json.loads(s) for s in (root/f'{stem}-linear-check.jsonl').read_text().splitlines() if s.startswith('{') and 'linear_check' in s]
 assert len(checks)==6 and all(r['passed'] for r in checks)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GOMP_SPINCOUNT='1000',
    NPU_CORES='3',NPU_DOMAIN_ID='1',NPU_FUSED='1',NPU_ATTENTION='1',NPU_SPLIT_DOWN='1',NPU_STREAM_FFN='1',NPU_CLS_TILE='8192',WARMUP_RUNS='2',
    LD_LIBRARY_PATH='/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs',COOL_REQUEST_C=str(args.cool_c))
for k in ['SERIAL_PREFILL','GPU_ATTENTION','GPU_PROFILE','NPU_PROFILE','LINEAR_PROFILE','ROUTE_PROFILE','CPU_CLASSIFIER','TEACHER_IDS','FFN_CPU_CHANNELS','FFN_GPU_CHANNELS','FFN_MODE','FFN_PROFILE']:env.pop(k,None)
if args.profile:env.update(GPU_PROFILE='1',LINEAR_PROFILE='1',NPU_PROFILE='1')
def clocks(script,action):
 subprocess.run(['docker','run','--rm','--platform=linux/amd64','--entrypoint=/qemu','-v',f'{matched}/tools/qemu-x86_64:/qemu:ro',
  '-v','/sys:/hostsys:rw','-v',f'{matched}:/work','python:3.10-slim-bookworm','/usr/local/bin/python3.10','/usr/local/bin/python3.10','/work/'+script,action],check=True)
fields={'cpu4_khz':'/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq','cpu6_khz':'/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq',
 'npu_hz':'/sys/class/devfreq/fdab0000.npu/cur_freq','gpu_hz':'/sys/class/devfreq/fb000000.gpu/cur_freq','ddr_hz':'/sys/class/devfreq/dmc/cur_freq','temperature_millidegrees':'/sys/class/thermal/thermal_zone0/temp'}
stop=threading.Event();phase='setup';thread=None;cpu_locked=False;gpu_locked=False;records=[]
def monitor():
 with (out/'clock-samples.jsonl').open('w') as log:
  while not stop.is_set():
   log.write(json.dumps({'time':time.time(),'phase':phase,**{k:int(Path(p).read_text()) for k,p in fields.items()}})+'\n');log.flush();stop.wait(.25)
try:
 clocks('clocks.py','lock');cpu_locked=True;shutil.copy2(matched/'clock-state.json',out/'clock-state-before.json')
 clocks('roofline/gpu/gpu_clocks.py','lock');gpu_locked=True;shutil.copy2(roofline/'gpu/gpu-clock-state.json',out/'gpu-clock-state-before.json')
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 for prompt in args.prompts:
  route_order=order if prompt==args.prompts[0] else list(reversed(order))
  for block,route in enumerate(route_order,1):
   phase=f'p{prompt}-{block}-{route}';dense_pre,dense_decode,attention,cpu_head=placements[route]
   jobenv=dict(env,DENSE_PREFILL=dense_pre,DENSE_DECODE=dense_decode)
   if attention:jobenv['GPU_ATTENTION']=attention
   if cpu_head:jobenv['CPU_CLASSIFIER']='1'
   if route in settings:
    cpu,gpu,mode=settings[route];jobenv.update(FFN_CPU_CHANNELS=str(cpu),FFN_GPU_CHANNELS=str(gpu),FFN_MODE=str(mode))
    if args.profile:jobenv['FFN_PROFILE']='1'
   phase='cooling-before-'+phase
   wait_start=time.monotonic();next_update=0
   while int(Path(fields['temperature_millidegrees']).read_text())>args.cool_c*1000:
    elapsed=time.monotonic()-wait_start
    if elapsed>=next_update:print(f'Cooling before {route}: {elapsed:.0f}s',flush=True);next_update+=15
    assert elapsed<180, 'Starting-temperature target not reached in 180s; restoring clocks before any new device job.'
    time.sleep(.5)
   phase=f'p{prompt}-{block}-{route}'
   golden_name={1:'hello',24:'chat24',73:'ragged73',128:'prompt128',256:'prompt256'}[prompt]
   tokens=roofline/'final-quality'/f'{golden_name}.tokens';copied=out/f'{phase}.tokens';shutil.copy2(tokens,copied)
   if args.new_tokens>16:golden_path=roofline/'full-profile/measurements'/('p128-1-reference.jsonl' if prompt==128 else 'p256-2-reference.jsonl')
   else:golden_path=roofline/'final-quality'/f'{golden_name}-custom.txt'
   golden=next(json.loads(s)['generated_ids'] for s in golden_path.read_text().splitlines() if s.startswith('{') and json.loads(s).get('event')=='run')[:args.new_tokens]
   teacher=out/f'{phase}.teacher.tokens';teacher.write_text(' '.join(map(str,golden))+'\n');jobenv['TEACHER_IDS']=str(teacher)
   logits=out/f'{phase}.f32';command=['taskset','-c','4-7',str(out/'runq-routes'),str(matched/'Qwen3-0.6B.fp16'),str(copied),str(args.new_tokens),str(args.runs),str(logits)]
   selected_keys=[*fields,'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','GOMP_SPINCOUNT','NPU_CORES','NPU_DOMAIN_ID','NPU_FUSED','NPU_ATTENTION','NPU_SPLIT_DOWN','NPU_STREAM_FFN','NPU_CLS_TILE','WARMUP_RUNS','DENSE_PREFILL','DENSE_DECODE','GPU_ATTENTION','CPU_CLASSIFIER','TEACHER_IDS','LINEAR_PROFILE','GPU_PROFILE','NPU_PROFILE','ROUTE_PROFILE','FFN_CPU_CHANNELS','FFN_GPU_CHANNELS','FFN_MODE','FFN_PROFILE','LD_LIBRARY_PATH','COOL_REQUEST_C']
   (out/f'{phase}.config.json').write_text(json.dumps({'command':command,'environment':{k:jobenv.get(k) for k in selected_keys},'route':route,'binary_sha256':sha(out/'runq-routes')},indent=2)+'\n')
   print('Running',phase,flush=True)
   with (out/f'{phase}.jsonl').open('w') as log:completed=subprocess.run(command,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
   assert completed.returncode==0,(phase,'runtime failure',completed.returncode)
   rows=[json.loads(s) for s in (out/f'{phase}.jsonl').read_text().splitlines() if s.startswith('{')]
   runs=[r for r in rows if r.get('event')=='run'];assert len(runs)==args.runs+2
   a=np.fromfile(logits,np.float32).astype(float);b=np.fromfile(roofline/'final-quality'/f'{golden_name}-custom.f32',np.float32).astype(float)
   assert a.shape==b.shape==(151936,) and np.isfinite(a).all()
   relative=float(np.linalg.norm(a-b)/np.linalg.norm(b));max_error=float(np.max(np.abs(a-b)))
   all_tokens=all(r['generated_ids']==golden for r in runs)
   measured=[r for r in runs if not r['warmup']]
   ttft=statistics.median(r['first_token_ms'] for r in measured);decode=statistics.median(r['decode_tps'] for r in measured)
   record={'label':phase,'prompt':prompt,'route':route,'dense_prefill':dense_pre,'dense_decode':dense_decode,'gpu_attention':attention,'cpu_classifier':cpu_head,
      'teacher_forced_identical_decode_inputs':True,'prefill_logit_relative_rmse':relative,'max_logit_error':max_error,'unchanged_limit':.001,
      'all_predictions_match':all_tokens,'quality_passed':relative<.001 and all_tokens,'ttft_ms':ttft,'effective_prefill_tps':prompt*1000/ttft,
      'decode_tps':decode,'measured_runs':len(measured),'ttft_range':[min(r['first_token_ms'] for r in measured),max(r['first_token_ms'] for r in measured)],
      'decode_range':[min(r['decode_tps'] for r in measured),max(r['decode_tps'] for r in measured)]}
   request_rows=[r for r in rows if r.get('event')=='request_wall' and not r['warmup']]
   assert len(request_rows)==len(measured)
   request_ms=statistics.median(r['request_ms'] for r in request_rows)
   record.update(request_ms=request_ms,output_tps=args.new_tokens*1000/request_ms,
     request_range=[min(r['request_ms'] for r in request_rows),max(r['request_ms'] for r in request_rows)],
     handoff_ms=statistics.median(r['handoff_ms'] for r in request_rows))
   records.append(record);print(json.dumps(record),flush=True)
finally:
 if thread:stop.set();thread.join()
 if gpu_locked:clocks('roofline/gpu/gpu_clocks.py','restore');shutil.copy2(roofline/'gpu/gpu-clock-state-restored.json',out/'gpu-clock-state-restored.json')
 if cpu_locked:clocks('clocks.py','restore');shutil.copy2(matched/'results/clock-state-restored.json',out/'clock-state-restored.json')
samples=[json.loads(s) for s in (out/'clock-samples.jsonl').read_text().splitlines()]
targets={'cpu4_khz':2256000,'cpu6_khz':2256000,'npu_hz':1000000000,'gpu_hz':1000000000,'ddr_hz':2112000000}
held=all(r[k]==v for r in samples for k,v in targets.items())
restored=all((out/f'{stem}-state-before.json').read_bytes()==(out/f'{stem}-state-restored.json').read_bytes() for stem in ['clock','gpu-clock'])
assert held and restored
(out/'summary.json').write_text(json.dumps({'records':records,'clocks_held':held,'clocks_restored':restored,'samples':len(samples),
 'temperature_range_c':[min(r['temperature_millidegrees'] for r in samples)/1000,max(r['temperature_millidegrees'] for r in samples)/1000],
 'binary_sha256':sha(out/'runq-routes'),'interpretation':'Phase-placement experiment. Non-linear host operations stay CPU. Failed quality rows are diagnostic timings, not qualified inference speed claims.'},indent=2)+'\n')
print('Finished; fixed clocks verified and original settings restored',flush=True)
