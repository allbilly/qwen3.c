"""Render audited, transfer-inclusive phase and request changes as artifacts."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('reports', nargs='+')
parser.add_argument('--stem', default='optimization-iterations')
args = parser.parse_args()
groups = [json.loads((root / file).read_text()) for file in args.reports]
assert len({group['cool_c'] for group in groups})==1
cool=groups[0]['cool_c']
prompts = sorted({r['prompt'] for group in groups for r in group['results']})
fig, axes = plt.subplots(len(prompts), 3, figsize=(16, 5*len(prompts)+1), squeeze=False)
fields = [('effective_prefill_tps','Prefill input tokens/s'), ('decode_tps','Decode tokens/s'), ('request_ms','Resident request ms')]
for row,prompt in enumerate(prompts):
    subset = [next(r for r in group['results'] if r['prompt']==prompt) for group in groups]
    labels = []
    for r in subset:
        for label in ['baseline','candidate']:
            engine = r[label]
            labels.append(engine['engine']+' · '+engine['route'])
    positions = np.array([3*group+engine for group in range(len(subset)) for engine in range(2)])
    for col,(field,title) in enumerate(fields):
        ax = axes[row,col]
        values = [r[engine]['median'][field] for r in subset for engine in ['baseline','candidate']]
        ranges = []
        for r in subset:
            for engine in ['baseline','candidate']:
                ranges.append(([prompt*1000/r[engine]['ranges']['ttft_ms'][1],prompt*1000/r[engine]['ranges']['ttft_ms'][0]]
                               if field=='effective_prefill_tps' else r[engine]['ranges'][field]))
        ax.barh(positions, values, color=['#64748b','#e88621']*len(subset), height=.7)
        ax.errorbar(values, positions, xerr=[[max(0,v-r[0]) for v,r in zip(values,ranges)],
                      [max(0,r[1]-v) for v,r in zip(values,ranges)]], fmt='none', ecolor='#111827', capsize=3, lw=1)
        for y,value in zip(positions,values):
            ax.text(value+max(values)*.02,y,f'{value:,.1f}',va='center',fontsize=9)
        ax.set_yticks(positions, labels if col==0 else ['']*len(labels));ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.22);ax.set_xlabel(title)
        ax.set_title(f'{prompt} input tokens · '+('lower is faster' if field=='request_ms' else 'higher is faster'),loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
fig.suptitle('RK3588 · independent changes · same Qwen3-0.6B FP16 weights',fontsize=15,y=.98)
fig.text(.02,.02,f'CPU host norms, RoPE, SwiGLU, residuals and sampling remain. 32 outputs / 31 decode steps; identical teacher inputs.\nTwo complete warmups per block; ABBA/BAAB; median of four measured requests per engine/prompt. Whiskers show min/max.\nFixed CPU/GPU/NPU/DDR clocks verified and restored. Start ≤{cool}°C; cooldown and initialization excluded.\nPhase and complete-request spans include activation movement, device waits and host math. Overlapping ranges limit small-gain claims.',fontsize=9)
fig.tight_layout(rect=(.01,.10,.99,.95))
files = {}
for extension in ['png','jpg','svg']:
    file = root / (args.stem+'.'+extension);fig.savefig(file,dpi=180,facecolor='white')
    files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
plt.close(fig)
(root / (args.stem+'-provenance.json')).write_text(json.dumps({'reports_sha256':{
    file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in args.reports}, 'files_sha256':files},indent=2)+'\n')
print('Rendered',args.stem)
