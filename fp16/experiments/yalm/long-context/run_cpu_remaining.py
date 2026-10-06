"""Run the two remaining CPU rows using observable clocks and writable outputs."""
from pathlib import Path
import json, subprocess

root = Path(__file__).resolve().parent
workspace = Path('/home/orangepi/qwen3-bench/matched/roofline/yalm/e2e/long-context')
output = Path('/tmp/qwen3-long-context-cpu-sample-only-20261006')
assert not output.exists(), 'Preserve prior results'
output.mkdir()
jobs = []
for prompt in [2048, 4096]:
    directory = output / f'p{prompt}-cpu'
    command = ['taskset', '-c', '0-3', 'python3', str(root / 'run_matrix.py'),
               '--workspace', str(workspace), '--output-directory', str(directory),
               '--output', directory.name, '--routes', 'cpu', '--prompts', str(prompt),
               '--runs', '2', '--cool-c', '0', '--context', '4128',
               '--golden-directory', 'references55-cpu1800-fanheld', '--cpu-khz', '1800000',
               '--allow-clock-drops', '--sample-only-clocks']
    jobs.append({'prompt': prompt, 'directory': str(directory), 'command': command})
plan = {'jobs': jobs, 'clock_control': 'sample_only', 'cooldown_enforced': False,
        'scope': 'CPU-only continuation in a sandbox without GPU/NPU nodes. Uses existing live clock limits without changing sysfs, fan or Docker settings. The earlier interrupted continuation restoration remains unverified. Same frozen executable, model, inputs, four threads, two warmups/two measurements and 32 outputs.'}
(root / 'long-context-cpu-sample-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
for job in jobs:
    print('CPU CONTINUE', job['prompt'], flush=True)
    completed = subprocess.run(job['command'], cwd=root)
    assert completed.returncode == 0, 'CPU child exited with a runtime failure; preserve raw evidence'
    summary = json.loads((Path(job['directory']) / 'summary.json').read_text())
    assert summary['clock_control'] == 'sample_only'
    assert not summary['clocks_restored'] and not summary['cooldown_enforced']
    print('CPU DONE', job['prompt'], 'quality', summary['records'][0]['quality_passed'],
          'target clocks held', summary['clocks_held'], 'limits unchanged',summary['clock_limits_unchanged'], flush=True)
