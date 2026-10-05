"""Serial reference and complete-request jobs, retaining failed clock screens."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent;parent=root/'e2e/long-context'
routes=['cpu','gpu','cpu_npu_dec','gpu_npu_dec','all_dec']
jobs=[{'directory':'e2e/long-context/references','reference':True,'routes':['npu'],'prompts':[1024,2048,4096]}]
for index,prompt in enumerate([1024,2048,4096]):
    for route in routes if index%2==0 else list(reversed(routes)):
        jobs.append({'directory':f'e2e/long-context/p{prompt}-{route}-screen51','reference':False,'routes':[route],'prompts':[prompt]})
for job in jobs:
    command=['taskset','-c','0-3','python3',str(parent/'run_matrix.py'),'--output',Path(job['directory']).name,
             '--routes',*job['routes'],'--prompts',*map(str,job['prompts']),'--runs','2','--cool-c','51','--context','4128']
    if job['reference']:command.append('--reference')
    job['command']=command
plan={'binary_sha256':hashlib.sha256((parent/'runq-routes').read_bytes()).hexdigest(),'jobs':jobs,
      'protocol':'Pinned shared FP16 model bytes; runtime capacity4128; synthetic exact1024/2048/4096-token nested prefixes; 32 outputs/31 decode steps; same four threads, fixed CPU2.256/NPU1/GPU1/DDR2.112GHz, common<=51C start, two warmups/two measurements, independently timed resident-model request; teacher IDs from the checked extended NPU path, every actual prediction retained. References supply NPU benchmark rows. Six placements per prompt. Clock failures remain diagnostics, no gates weakened.',
      'scope':'Screening, not counterbalanced statistical evidence. Long NPU references are not independent HF full-model oracles.'}
(root/'long-context-execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for index,job in enumerate(jobs,1):
    print('LONG JOB',index,'/',len(jobs),job['directory'],flush=True)
    completed=subprocess.run(job['command'],cwd=root)
    if completed.returncode:
        summary=root/job['directory']/'summary.json'
        if not summary.exists():raise RuntimeError(f"Runtime/setup failure in {job['directory']}; hardware processes have exited normally before any follow-up.")
        report=json.loads(summary.read_text());assert report['clocks_restored'] and not report['clocks_held']
        print('Retained failed clock screen:',job['directory'],flush=True)
print('All long-context jobs ended; original clocks restored for every controller.',flush=True)
