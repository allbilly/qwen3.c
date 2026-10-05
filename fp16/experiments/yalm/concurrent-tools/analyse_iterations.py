"""Aggregate every measured request after independent clock/output audits."""
from pathlib import Path
import argparse, hashlib, json, statistics

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--plan', required=True)
parser.add_argument('--audit', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
plan = json.loads((root / args.plan).read_text())
audit = json.loads((root / args.audit).read_text())
assert len(plan['jobs']) == 4 and ''.join(r['letter'] for r in plan['jobs']) == plan['order']
assert plan['order'] in ['ABBA','BAAB']
values = {}
for job in plan['jobs']:
    rows = [r for r in audit['records'] if r['directory'] == job['directory']]
    assert rows and all(r['quality_passed'] and r['measured_runs'] == 2 for r in rows)
    for row in rows:
        assert row['route'] == job['route'] and 'request_ms' in row
        directory = root / job['directory']
        config = json.loads((directory / (row['label'] + '.config.json')).read_text())
        assert config['environment']['COOL_REQUEST_C'] == str(plan['cool_c'])
        raw = [json.loads(s) for s in (directory / (row['label'] + '.jsonl')).read_text().splitlines() if s.startswith('{')]
        runs = [r for r in raw if r.get('event') == 'run' and not r['warmup']]
        requests = [r for r in raw if r.get('event') == 'request_wall' and not r['warmup']]
        assert len(runs) == len(requests) == 2
        for run, request in zip(runs, requests):
            assert run['run'] == request['run']
            values.setdefault((job['letter'], row['prompt']), []).append({
                'ttft_ms':run['first_token_ms'], 'decode_tps':run['decode_tps'],
                'request_ms':request['request_ms'], 'handoff_ms':request['handoff_ms'],
                'directory':job['directory'], 'label':row['label'], 'run':run['run']})
results = []
for prompt in sorted({p for letter,p in values}):
    engines = {}
    for letter,engine in [('A',plan['baseline']),('B',plan['candidate'])]:
        requests = values[(letter,prompt)]
        assert len(requests) == 4
        medians = {field:statistics.median(r[field] for r in requests)
                   for field in ['ttft_ms','decode_tps','request_ms','handoff_ms']}
        medians['effective_prefill_tps'] = prompt * 1000 / medians['ttft_ms']
        medians['output_tps'] = 32 * 1000 / medians['request_ms']
        ranges = {field:[min(r[field] for r in requests),max(r[field] for r in requests)]
                  for field in ['ttft_ms','decode_tps','request_ms']}
        engines[letter] = {'engine':engine, 'route':plan['baseline_route'] if letter=='A' else plan['candidate_route'],
                           'measurements':4, 'median':medians, 'ranges':ranges, 'requests':requests}
    a,b = engines['A'],engines['B']
    overlap = {field:max(a['ranges'][field][0],b['ranges'][field][0]) <= min(a['ranges'][field][1],b['ranges'][field][1])
               for field in ['ttft_ms','decode_tps','request_ms']}
    results.append({'prompt':prompt, 'baseline':a, 'candidate':b,
        'prefill_gain_percent':100*(a['median']['ttft_ms']/b['median']['ttft_ms']-1),
        'decode_gain_percent':100*(b['median']['decode_tps']/a['median']['decode_tps']-1),
        'request_gain_percent':100*(a['median']['request_ms']/b['median']['request_ms']-1),
        'observed_ranges_overlap':overlap})
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report = {'plan':args.plan, 'plan_sha256':sha(root/args.plan), 'audit':args.audit, 'cool_c':plan['cool_c'],
          'audit_sha256':sha(root/args.audit), 'results':results,
          'scope':'Four independently measured requests per engine/prompt in ABBA or BAAB blocks. Phase and request medians recomputed from all four raw measurements. Full min/max ranges retained; overlapping ranges do not establish a statistically significant small gain.'}
(root / args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps([{'prompt':r['prompt'], **{key:r[key] for key in ['prefill_gain_percent','decode_gain_percent','request_gain_percent','observed_ranges_overlap']}} for r in results], indent=2))
