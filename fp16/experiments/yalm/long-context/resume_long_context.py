"""Continue only unmeasured placements; retain thermal behavior and quality failures."""
from pathlib import Path
import argparse, hashlib, json, subprocess, time

root = Path(__file__).resolve().parent
parent = root / 'e2e/long-context'
p = argparse.ArgumentParser()
p.add_argument('--dry-run', action='store_true')
args = p.parse_args()
audit = json.loads((root / 'long-context-audit.json').read_text())
observed = {(row['prompt'], row['route']) for row in audit['records']}
routes = ['cpu', 'gpu', 'npu', 'cpu_npu_dec', 'gpu_npu_dec', 'all_dec']
missing = {(prompt, route) for prompt in [1024, 2048, 4096] for route in routes} - observed
# Complete the shorter accelerator jobs before the scalar CPU attention cases.
order = [(4096, 'gpu'), (4096, 'cpu_npu_dec'), (4096, 'gpu_npu_dec'), (4096, 'all_dec'), (2048, 'cpu'), (4096, 'cpu')]
assert missing == set(order), 'Inspect a changed missing-placement set before launching'
suffix = 'continue-throttle-cpu1800-fanheld'
references = 'references55-cpu1800-fanheld'
jobs = []
for prompt, route in order:
    directory = f'e2e/long-context/p{prompt}-{route}-{suffix}'
    assert not (root / directory).exists(), 'Result directories are immutable'
    command = ['taskset', '-c', '0-3', 'python3', str(parent / 'run_matrix.py'),
               '--output', Path(directory).name, '--routes', route, '--prompts', str(prompt),
               '--runs', '2', '--cool-c', '0', '--context', '4128', '--golden-directory', references,
               '--cpu-khz', '1800000', '--allow-clock-drops']
    jobs.append({'directory': directory, 'reference': False, 'routes': [route], 'prompts': [prompt], 'command': command})
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan = {'cpu_target_khz': 1800000, 'cooldown_target_c': 0, 'allow_clock_drops': True,
        'cooldown_enforced': False, 'binary_sha256': sha(parent / 'runq-routes'),
        'previous_audit_sha256': sha(root / 'long-context-audit.json'), 'jobs': jobs,
        'fan_suffix': suffix, 'golden_directory': references,
        'protocol': 'Same model, compiled runner, target clocks, four host threads, two warmups/two measurements and 32 outputs. Software cooldown disabled by removing COOL_REQUEST_C. Kernel thermal protection remains active. Every actual clock sample and quality failure is retained; original clocks/fan command restore normally.',
        'scope': 'Continuation of six missing rows. Starting-temperature conditions differ from the original <=55C session. These are observed bare-board timings, not a fixed-clock causal speedup experiment.',
        'authorization': 'even clock drop u just mention it in doc and continue run it is ok ?'}
if args.dry_run:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)
fan_state = root / f'long-context-fan-{suffix}.json'
assert not fan_state.exists()
(root / f'long-context-execution-plan-{suffix}.json').write_text(json.dumps(plan, indent=2) + '\n')
matched = root.parents[1]
command = ['docker', 'run', '--cpuset-cpus=0-3', '--rm', '--platform=linux/amd64', '--entrypoint=/qemu',
           '-v', f'{matched}/tools/qemu-x86_64:/qemu:ro', '-v', '/sys:/hostsys:rw', '-v', f'{matched}:/work',
           'python:3.10-slim-bookworm', '/usr/local/bin/python3.10', '/usr/local/bin/python3.10',
           '/work/roofline/yalm/fan_hold.py', f'/work/roofline/yalm/{fan_state.name}']
fan_process = subprocess.Popen(command)
try:
    deadline = time.monotonic() + 30
    while not fan_state.with_suffix('.ready').exists():
        assert fan_process.poll() is None and time.monotonic() < deadline
        time.sleep(.1)
    for index, job in enumerate(jobs, 1):
        print('CONTINUE JOB', index, '/', len(jobs), job['directory'], flush=True)
        completed = subprocess.run(job['command'], cwd=root)
        assert completed.returncode == 0, 'Runtime/setup/restoration failure; native child has exited normally'
        summary = json.loads((root / job['directory'] / 'summary.json').read_text())
        assert summary['clocks_restored'] and summary['allow_clock_drops'] and not summary['cooldown_enforced']
        print('Completed; quality:', [row['quality_passed'] for row in summary['records']],
              'clock targets held:', summary['clocks_held'], flush=True)
finally:
    fan_state.with_suffix('.stop').write_text('restore\n')
    assert fan_process.wait(timeout=30) == 0, 'Fan restoration failed'
