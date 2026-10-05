"""Serial additional numerical coverage; no timings qualified as performance data."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess
import numpy as np
root=Path(__file__).resolve().parent;roofline=root.parent;matched=roofline.parent
p=argparse.ArgumentParser();p.add_argument('--mode',type=int,choices=[1,2,3],default=3);p.add_argument('--parent',default='.');p.add_argument('--output',required=True);p.add_argument('--routes',nargs='+',default=['npu']);p.add_argument('--prompts',nargs='+',type=int,default=[1,24,73]);args=p.parse_args()
parent=(root/args.parent).resolve();out=parent/args.output;out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
metadata=json.loads((parent/'source-provenance.json').read_text());assert sha(parent/'runq-routes')==metadata['binary_sha256']
shutil.copy2(parent/'runq-routes',out/'runq-routes');shutil.copy2(parent/'source-provenance.json',out/'source-provenance.json')
placements={r:('npu','npu',None,False) for r in ['cpu_npu','gpu_npu','all']}
settings={'cpu_npu':(96,0),'gpu_npu':(0,96),'all':(96,96)}
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GOMP_SPINCOUNT='1000',NPU_CORES='3',NPU_DOMAIN_ID='1',NPU_FUSED='1',NPU_ATTENTION='1',NPU_SPLIT_DOWN='1',NPU_STREAM_FFN='1',NPU_CLS_TILE='8192',WARMUP_RUNS='1',COOL_REQUEST_C='55',LD_LIBRARY_PATH='/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs')
for key in ['SERIAL_PREFILL','GPU_ATTENTION','GPU_PROFILE','NPU_PROFILE','LINEAR_PROFILE','ROUTE_PROFILE','CPU_CLASSIFIER','TEACHER_IDS','CPU_ATTN_BLAS','FFN_PROFILE','FFN_CPU_CHANNELS','FFN_GPU_CHANNELS','FFN_MODE']:env.pop(key,None)
if parent.name=='cpu-blas-attention':env['CPU_ATTN_BLAS']='1'
records=[]
for prompt in args.prompts:
 name={1:'hello',24:'chat24',73:'ragged73',128:'prompt128',256:'prompt256'}[prompt]
 golden_path=roofline/'final-quality'/f'{name}-custom.txt';golden=next(json.loads(s)['generated_ids'] for s in golden_path.read_text().splitlines() if s.startswith('{') and json.loads(s).get('event')=='run')[:16]
 for route in args.routes:
  pre,dec,attention,head=placements[route];label=f'p{prompt}-{route}';jobenv=dict(env,DENSE_PREFILL=pre,DENSE_DECODE=dec)
  if attention:jobenv['GPU_ATTENTION']=attention
  if head:jobenv['CPU_CLASSIFIER']='1'
  cpu,gpu=settings[route];jobenv.update(FFN_CPU_CHANNELS=str(cpu),FFN_GPU_CHANNELS=str(gpu),FFN_MODE=str(args.mode))
  tokens=out/(label+'.tokens');teacher=out/(label+'.teacher.tokens');shutil.copy2(roofline/'final-quality'/f'{name}.tokens',tokens);teacher.write_text(' '.join(map(str,golden))+'\n');jobenv['TEACHER_IDS']=str(teacher)
  logits=out/(label+'.f32');command=['taskset','-c','4-7',str(out/'runq-routes'),str(matched/'Qwen3-0.6B.fp16'),str(tokens),'16','1',str(logits)]
  selected=['DENSE_PREFILL','DENSE_DECODE','GPU_ATTENTION','CPU_CLASSIFIER','CPU_ATTN_BLAS','NPU_CORES','NPU_DOMAIN_ID','NPU_FUSED','NPU_ATTENTION','NPU_SPLIT_DOWN','NPU_STREAM_FFN','NPU_CLS_TILE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','GOMP_SPINCOUNT','WARMUP_RUNS','COOL_REQUEST_C','LD_LIBRARY_PATH','TEACHER_IDS','FFN_CPU_CHANNELS','FFN_GPU_CHANNELS','FFN_MODE','FFN_PROFILE']
  (out/(label+'.config.json')).write_text(json.dumps({'command':command,'environment':{k:jobenv.get(k) for k in selected},'route':route,'purpose':'Additional numerical coverage only; no fixed-clock performance claim.'},indent=2)+'\n')
  print('Checking',parent.name,label,flush=True)
  with (out/(label+'.jsonl')).open('w') as log:r=subprocess.run(command,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(label,r.returncode)
  rows=[json.loads(s) for s in (out/(label+'.jsonl')).read_text().splitlines() if s.startswith('{')]
  runs=[r for r in rows if r.get('event')=='run'];assert [r['run'] for r in runs]==[0,1]
  a=np.fromfile(logits,np.float32).astype(float);b=np.fromfile(roofline/'final-quality'/f'{name}-custom.f32',np.float32).astype(float)
  assert a.shape==b.shape==(151936,) and np.isfinite(a).all()
  relative=float(np.linalg.norm(a-b)/np.linalg.norm(b));matching=all(r['generated_ids']==golden for r in runs)
  routing=next(r for r in rows if r.get('event')=='routing');assert routing['dense_prefill']==pre and routing['dense_decode']==dec
  assert routing['npu_initialized']==('npu' in [pre,dec]) and routing['gpu_projections_initialized']==('gpu' in [pre,dec])
  record={'prompt':prompt,'route':route,'logit_relative_rmse':relative,'unchanged_limit':.001,'all_predictions_match':matching,'quality_passed':relative<.001 and matching,'golden_ids':golden,'raw_log_sha256':sha(out/(label+'.jsonl'))}
  records.append(record);print(json.dumps(record),flush=True)
(out/'quality-summary.json').write_text(json.dumps({'binary_sha256':metadata['binary_sha256'],'records':records,'purpose':'Numerical coverage only. Timings are not qualified performance measurements; clock settings untouched. One complete warmup, one full request with all 16 predictions compared.'},indent=2)+'\n')
print('Additional quality coverage completed; clocks untouched',flush=True)
