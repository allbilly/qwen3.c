"""Render audited native-packing request timings; run after hardware jobs finish."""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['concurrent-ffn','concurrent-native'],required=True);p.add_argument('--audit',required=True);p.add_argument('--stem',required=True);args=p.parse_args()
audit=Path(args.audit);data=json.loads(audit.read_text());rows=[r for r in data['records'] if r['directory'].split('/')[1]==args.variant]
assert len(rows)==20 and len({(r['prompt'],r['route']) for r in rows})==20
order=['npu','cpu_npu_pre','gpu_npu_pre','all_pre','cpu_npu_dec','gpu_npu_dec','all_dec','cpu_npu_both','gpu_npu_both','all_both']
labels=['Selected NPU','CPU+NPU pre *','GPU+NPU pre *','CPU+GPU+NPU pre *','CPU+NPU decode','GPU+NPU decode','CPU+GPU+NPU decode','CPU+NPU both *','GPU+NPU both *','CPU+GPU+NPU both *']
fig,axes=plt.subplots(2,3,figsize=(17,11),sharey=True)
colors=['#176b54' if r=='npu' else '#b06327' if not r.endswith('_dec') else '#286998' for r in order]
for i,prompt in enumerate([128,256]):
 subset=[next(r for r in rows if r['prompt']==prompt and r['route']==name) for name in order]
 for j,(key,title,suffix) in enumerate([('effective_prefill_tps','Prefill','tokens/s'),('decode_tps','Decode','tokens/s'),('request_ms','Complete request','ms')]):
  ax=axes[i,j];values=[r[key] for r in subset];pos=np.arange(10)
  bars=ax.barh(pos,values,color=colors,height=.68)
  if key=='request_ms':ranges=[r['request_range'] for r in subset]
  elif key=='decode_tps':ranges=[r['decode_range'] for r in subset]
  else:ranges=[[prompt*1000/r['ttft_range'][1],prompt*1000/r['ttft_range'][0]] for r in subset]
  ax.errorbar(values,pos,xerr=np.array([[v-a for v,(a,b) in zip(values,ranges)],[b-v for v,(a,b) in zip(values,ranges)]]),fmt='none',ecolor='#1b252e',capsize=3,lw=1)
  for y,v in zip(pos,values):ax.text(v+max(values)*.015,y,f'{v:,.1f}',va='center',fontsize=9)
  ax.set_yticks(pos,labels);ax.set_ylim(9.6,-.6);ax.set_xlim(0,max(values)*1.22);ax.set_title(f'{prompt} input tokens — {title}',fontsize=13);ax.set_xlabel(suffix);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
  for edge in ['top','right']:ax.spines[edge].set_visible(False)
layout='Original split' if args.variant=='concurrent-ffn' else 'Native packing, 64-row blocks'
fig.suptitle(f'RK3588: {layout} — concurrent FFN channels',fontsize=18,y=.98)
fig.text(.5,.026,'Same pinned Qwen3-0.6B FP16 operands · auxiliary slices 96/3072 channels each · 32 outputs · 2 warmups + 2 measurements\nCPU 2.256GHz / NPU 1GHz / Mali 1GHz / DDR 2.112GHz · common ≤51°C start · error bars: individual min/max\n* Concurrent prefill exceeds the unchanged logit gate at 24/73 tokens; shown as diagnostics. Model loading and cooling excluded.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.085,1,.95]);stem=Path(args.stem);stem.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','jpg','svg']:
 fig.savefig(str(stem)+'.'+ext,dpi=170)
 if ext=='svg':
  output=Path(str(stem)+'.'+ext)
  output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines())+'\n')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
Path(str(stem)+'-provenance.json').write_text(json.dumps({'audit':str(audit),'variant':args.variant,'audit_sha256':sha(audit),'plot_source_sha256':sha(Path(__file__)),'output_sha256':{ext:sha(Path(str(stem)+'.'+ext)) for ext in ['png','jpg','svg']}},indent=2)+'\n')
print('Rendered audited native concurrency PNG/JPG/SVG')
