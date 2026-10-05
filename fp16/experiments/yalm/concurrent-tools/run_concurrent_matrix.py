"""Serial fixed-clock request sweeps of the two frozen FFN layouts."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
routes=['npu','cpu_npu_pre','gpu_npu_pre','all_pre','cpu_npu_dec','gpu_npu_dec','all_dec','cpu_npu_both','gpu_npu_both','all_both']
jobs=[]
for variant in ['concurrent-ffn','concurrent-native']:
 parent=root/'e2e'/variant
 jobs.append({'variant':variant,'binary_sha256':hashlib.sha256((parent/'runq-routes').read_bytes()).hexdigest(),
  'command':['python3',str(parent/'run_matrix.py'),'--output','request-matrix51','--routes',*routes,'--prompts','128','256','--runs','2','--cool-c','51']})
plan={'jobs':jobs,'protocol':'Same pinned FP16 model/operands, two full warmups, two measured requests per placement/prompt, 32 greedy predictions/31 decode steps, original numerical gate, actual CPU/GPU/NPU counters, fixed/sampled/restored clocks and common <=51C request start. Routes reverse at prompt256. One independent monotonic request span includes movement, waits and interphase work. Model load, tokenization, cooling excluded. Branch event profiling disabled. Small differences remain screening observations, not a statistical reliability claim.',
      'known_prefill_failures':'Both layouts fail 24/73-token first-logit gates on all three splits; prefill split rows are diagnostics despite qualified 128/256 local cases.'}
(root/'concurrent-request-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for job in jobs:
 print('REQUEST SWEEP',job['variant'],flush=True)
 subprocess.run(job['command'],check=True,cwd=root)
print('Both sweeps ended normally; forty complete placement/prompt jobs',flush=True)
