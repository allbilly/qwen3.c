"""Standalone scientific figures from independently audited raw measurements."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--audit',default='matrix-audit.json');p.add_argument('--suite',nargs='+',default=['phase-qualified','cpu256-cooled50']);p.add_argument('--stem',default='phase-matrix');args=p.parse_args()
report=json.loads((root/args.audit).read_text());records=[r for r in report['records'] if r['directory'] in args.suite]
assert len(records)==20 and len({(r['prompt'],r['route']) for r in records})==20
order=['cpu','gpu','npu','cpu_gpu','gpu_cpu','cpu_npu','npu_cpu','gpu_npu','npu_gpu','all']
labels=['CPU → CPU','GPU → GPU','NPU → NPU','CPU → GPU','GPU → CPU','CPU → NPU','NPU → CPU','GPU → NPU','NPU → GPU','GPU → NPU + CPU head*']
colours=['#64748b','#16a34a','#e88621',*['#6d5fc7']*7]
has_e2e=all('request_ms' in r for r in records)
columns=[('effective_prefill_tps','Effective prefill (input tokens/s)','higher is faster'),('decode_tps','Decode (tokens/s)','higher is faster')]
if has_e2e:columns.append(('request_ms','Resident request (ms)','lower is faster'))
fig,axes=plt.subplots(2,len(columns),figsize=(16 if has_e2e else 12,11),squeeze=False)
for row,prompt in enumerate([128,256]):
 for col,(field,title,note) in enumerate(columns):
  ax=axes[row,col];subset=[next(r for r in records if r['prompt']==prompt and r['route']==route) for route in order]
  values=np.array([r[field] for r in subset]);bars=ax.barh(np.arange(10),values,color=colours,height=.67)
  ax.invert_yaxis();ax.set_yticks(np.arange(10),labels if col==0 else ['']*10);ax.set_xlabel(title);ax.set_title(f'{prompt} prompt tokens • {note}',loc='left',fontsize=11)
  ax.set_xlim(0,max(values)*1.2);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
  if field=='decode_tps':ranges=[r['decode_range'] for r in subset]
  elif field=='request_ms':ranges=[r['request_range'] for r in subset]
  else:ranges=[[prompt*1000/r['ttft_range'][1],prompt*1000/r['ttft_range'][0]] for r in subset]
  lower=[max(0,v-r[0]) for v,r in zip(values,ranges)];upper=[max(0,r[1]-v) for v,r in zip(values,ranges)]
  ax.errorbar(values,np.arange(10),xerr=[lower,upper],fmt='none',ecolor='#111827',capsize=2,lw=.8)
  for y,v,r in zip(range(10),values,subset):ax.text(v+max(values)*.015,y,f'{v:,.1f}'+(' ×' if not r['quality_passed'] else ''),va='center',fontsize=9)
  ax.spines[['right','top']].set_visible(False)
fig.suptitle('RK3588 • same Qwen3-0.6B FP16 weights • projection placement: prefill → decode',fontsize=14,y=.98)
if has_e2e:
 limits={json.loads((root/r['directory']/(r['label']+'.config.json')).read_text())['environment']['COOL_REQUEST_C'] for r in records}
 assert len(limits)==1, 'Complete-request comparisons require one common start limit.'
 cool_note=f'All requests start at ≤{next(iter(limits))}°C.'
else:cool_note='Start ≤55°C; CPU/CPU 256 starts ≤50°C after a rejected thermal run.'
fig.text(.03,.025,'CPU host norms, RoPE, SwiGLU, residuals and sampling remain in every route. 32 output tokens / 31 decode steps.\nTwo full warmups; median of two measured requests; whiskers show min/max. Fixed clocks verified; cooldown excluded.\nEffective prefill = prompt tokens / time to first token, including classifier and sampling. * GPU decode attention, CPU prefill attention.\n'+cool_note+'\nCPU/GPU prefill routes fail extra short-prompt logit checks; these are finite-case phase measurements.',fontsize=9)
fig.tight_layout(rect=(.01,.09,.99,.95));files={}
for ext in ['png','jpg','svg']:
 path=root/(args.stem+'.'+ext);fig.savefig(path,dpi=180,facecolor='white');files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
plt.close(fig)
(root/(args.stem+'-provenance.json')).write_text(json.dumps({'audit':args.audit,'audit_sha256':hashlib.sha256((root/args.audit).read_bytes()).hexdigest(),'suite':args.suite,'files_sha256':files},indent=2)+'\n')
print('Rendered',args.stem)
