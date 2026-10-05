"""Serial reference and complete-request jobs, retaining failed clock screens."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
root=Path(__file__).resolve().parent;parent=root/'e2e/long-context'
parser=argparse.ArgumentParser();parser.add_argument('--cool-c',type=int,default=55);args=parser.parse_args()
suffix=f'screen{args.cool_c}-cpu1800-fanheld';references=f'references{args.cool_c}-cpu1800-fanheld'
routes=['cpu','gpu','cpu_npu_dec','gpu_npu_dec','all_dec']
jobs=[{'directory':f'e2e/long-context/{references}','reference':True,'routes':['npu'],'prompts':[1024,2048,4096]}]
for index,prompt in enumerate([1024,2048,4096]):
    for route in routes if index%2==0 else list(reversed(routes)):
        jobs.append({'directory':f'e2e/long-context/p{prompt}-{route}-{suffix}','reference':False,'routes':[route],'prompts':[prompt]})
for job in jobs:
    command=['taskset','-c','0-3','python3',str(parent/'run_matrix.py'),'--output',Path(job['directory']).name,
             '--routes',*job['routes'],'--prompts',*map(str,job['prompts']),'--runs','2','--cool-c',str(args.cool_c),'--context','4128','--golden-directory',references,'--cpu-khz','1800000']
    if job['reference']:command.append('--reference')
    job['command']=command
plan={'cpu_target_khz':1800000,'cooldown_target_c':args.cool_c,'rejected_attempt':'e2e/long-context/references (51C cooldown timeout after first4K warmup; no4K measured row)', 'binary_sha256':hashlib.sha256((parent/'runq-routes').read_bytes()).hexdigest(),'jobs':jobs,
      'fan_policy':'setpoint255 reasserted by common10ms feedback loop; kernel thermal protections and notifier remain active','protocol':f'Pinned shared FP16 model bytes; runtime capacity4128; synthetic exact1024/2048/4096-token nested prefixes; 32 outputs/31 decode steps; same four threads, fixed CPU1.800/NPU1/GPU1/DDR2.112GHz, common<={args.cool_c}C start, fanPWM255 feedback setpoint, two warmups/two measurements, independently timed resident-model request; teacher IDs from the checked extended NPU path, every actual prediction retained. References supply NPU benchmark rows. Six placements per prompt. Clock failures remain diagnostics, no gates weakened.',
      'scope':'Screening, not counterbalanced statistical evidence. Long NPU references are not independent HF full-model oracles.'}
(root/f'long-context-execution-plan-{suffix}.json').write_text(json.dumps(plan,indent=2)+'\n')
matched=root.parents[1]
fan_state=root/f'long-context-fan-{suffix}.json'
assert not fan_state.exists()
command=['docker','run','--cpuset-cpus=0-3','--rm','--platform=linux/amd64','--entrypoint=/qemu','-v',f'{matched}/tools/qemu-x86_64:/qemu:ro','-v','/sys:/hostsys:rw','-v',f'{matched}:/work','python:3.10-slim-bookworm','/usr/local/bin/python3.10','/usr/local/bin/python3.10','/work/roofline/yalm/fan_hold.py',f'/work/roofline/yalm/{fan_state.name}']
fan_process=subprocess.Popen(command)
try:
    deadline=time.monotonic()+30
    while not fan_state.with_suffix('.ready').exists():
        assert fan_process.poll() is None and time.monotonic()<deadline, 'Fan controller did not become ready before hardware work'
        time.sleep(.1)
    for index,job in enumerate(jobs,1):
        print('LONG JOB',index,'/',len(jobs),job['directory'],flush=True)
        completed=subprocess.run(job['command'],cwd=root)
        if completed.returncode:
            summary=root/job['directory']/'summary.json'
            if not summary.exists():raise RuntimeError(f"Runtime/setup failure in {job['directory']}; hardware processes have exited normally before any follow-up.")
            report=json.loads(summary.read_text());assert report['clocks_restored'] and not report['clocks_held']
            print('Retained failed clock screen:',job['directory'],flush=True)
    print('All long-context jobs ended; original clocks restored for every controller.',flush=True)
finally:
    fan_state.with_suffix('.stop').write_text('restore\n')
    assert fan_process.wait(timeout=30)==0, 'Fan restoration failed'
