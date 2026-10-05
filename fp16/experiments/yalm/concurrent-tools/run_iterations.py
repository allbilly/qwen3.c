"""Serial, counterbalanced single-change complete-request comparisons.

Each block runs two full warmups and two measured requests at each prompt
length. ABBA/BAAB and prompt-order reversal give four measurements per engine
and prompt. The phase harness owns clock fixation, sampling, and restoration.
"""
from pathlib import Path
import argparse, json, subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--baseline', default='baseline-guarded')
parser.add_argument('--candidate', required=True)
parser.add_argument('--route', default='npu')
parser.add_argument('--baseline-route')
parser.add_argument('--candidate-route')
parser.add_argument('--prompts', nargs='+', type=int, default=[128, 256])
parser.add_argument('--order', choices=['ABBA','BAAB'], default='ABBA')
parser.add_argument('--cool-c',type=int,default=51)
args = parser.parse_args()
plan_path = root / ('iteration-' + args.name + '-plan.json')
assert not plan_path.exists(), 'Use a new immutable experiment name.'
jobs = []
for index, letter in enumerate(args.order, 1):
    engine = args.baseline if letter == 'A' else args.candidate
    parent = root / 'e2e' / engine
    output = 'iteration-' + args.name + '-' + str(index)
    prompts = args.prompts if index <= 2 else list(reversed(args.prompts))
    route = (args.baseline_route if letter == 'A' else args.candidate_route) or args.route
    command = ['python3', str(parent / 'run_matrix.py'), '--output', output,
               '--routes', route, '--prompts', *map(str, prompts),
               '--runs', '2', '--new-tokens', '32', '--cool-c', str(args.cool_c)]
    jobs.append({'block':index, 'letter':letter, 'engine':engine, 'route':route,
                 'directory':str((parent / output).relative_to(root)), 'command':command})
plan = {'name':args.name, 'baseline':args.baseline, 'candidate':args.candidate,
        'baseline_route':args.baseline_route or args.route, 'candidate_route':args.candidate_route or args.route,
        'order':args.order, 'jobs':jobs, 'cool_c':args.cool_c,
        'protocol':f'Same shared FP16 weights and teacher-forced decode inputs; all actual predictions checked. Two complete warmups and two measured requests per prompt per block; four measurements per engine/prompt. Start every request at <={args.cool_c}C; cooldown excluded. Hardware processes serial. Independently measured resident request spans.'}
plan_path.write_text(json.dumps(plan, indent=2) + '\n')
for job in jobs:
    print('Iteration', args.name, 'block', job['block'], job['engine'], flush=True)
    subprocess.run(job['command'], check=True)
print('Completed counterbalanced comparison', args.name, flush=True)
