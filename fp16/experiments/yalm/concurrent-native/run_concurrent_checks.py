from pathlib import Path
import os,json,subprocess,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument('--parent',default='e2e/concurrent-ffn');args=p.parse_args()
root=Path(__file__).resolve().parent;p=root/args.parent;out=p/'primitive';out.mkdir(exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GOMP_SPINCOUNT='1000',NPU_CORES='3',NPU_DOMAIN_ID='1',NPU_FUSED='1',NPU_ATTENTION='1',NPU_SPLIT_DOWN='1',NPU_STREAM_FFN='1',NPU_CLS_TILE='8192',DENSE_PREFILL='npu',DENSE_DECODE='npu',LD_LIBRARY_PATH='/home/orangepi/.local/lib/python3.10/site-packages/numpy.libs',FFN_MODE='3')
for k in ['FFN_PROFILE','CPU_CLASSIFIER','GPU_ATTENTION']:env.pop(k,None)
records=[]
for label,cpu,gpu in [('cpu_npu',96,0),('gpu_npu',0,96),('all',96,96)]:
 job=dict(env,FFN_CPU_CHANNELS=str(cpu),FFN_GPU_CHANNELS=str(gpu));cmd=['taskset','-c','4-7',str(p/'ffn-check'),str(root.parents[1]/'Qwen3-0.6B.fp16')]
 (out/f'{label}.config.json').write_text(json.dumps({'command':cmd,'environment':job,'cpu_channels':cpu,'gpu_channels':gpu,'npu_channels':3072-cpu-gpu,'purpose':'Numerical only; original clock settings unchanged.'},indent=2)+'\n')
 print('Primitive',label,flush=True)
 with (out/f'{label}.jsonl').open('w') as log:r=subprocess.run(cmd,env=job,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(label,r.returncode)
 cases=[json.loads(s) for s in (out/f'{label}.jsonl').read_text().splitlines() if s.startswith('{') and json.loads(s)['event']=='ffn_primitive_check']
 assert len(cases)==3 and all(r['passed'] for r in cases)
 native=[json.loads(s) for s in (out/f'{label}.jsonl').read_text().splitlines() if s.startswith('{') and json.loads(s)['event']=='ffn_native_check']
 if p.name=='concurrent-native':assert len(native)==2 and all(r['passed'] for r in native)
 records.append({'native_cases':native,'route':label,'cpu_channels':cpu,'gpu_channels':gpu,'cases':cases});print(json.dumps(records[-1]),flush=True)
(out/'summary.json').write_text(json.dumps({'records':records,'check_binary_sha256':hashlib.sha256((p/'ffn-check').read_bytes()).hexdigest(),'scope':'All tested outputs, identical FP16 operands, double accumulation; no qualified timings.'},indent=2)+'\n')
