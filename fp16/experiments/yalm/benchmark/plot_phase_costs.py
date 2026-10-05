"""Transfer-inclusive measured component figures from audited profiles."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',default='phase-costs.json');args=p.parse_args()
report=json.loads((root/args.input).read_text());rows=[r for r in report['rows'] if r['directory']=='profiles/baseline-profile/costs256']
order=['cpu','gpu','npu','all']
components=list(rows[0]['components_ms']);palette=plt.colormaps['tab20'](np.linspace(0,1,len(components)))
fig,axes=plt.subplots(1,2,figsize=(13.5,7.5))
for ax,phase in zip(axes,['prefill','decode']):
 subset=[next(r for r in rows if r['phase']==phase and r['route']==route) for route in order]
 bottom=np.zeros(4)
 for name,colour in zip(components,palette):
  values=np.array([r['components_ms'][name] for r in subset]);ax.bar(np.arange(4),values,bottom=bottom,color=colour,label=name);bottom+=values
 ax.set_xticks(np.arange(4),['CPU','GPU','NPU','3-device*']);ax.set_ylabel('milliseconds / prompt' if phase=='prefill' else 'milliseconds / decode token')
 ax.set_title('256-token prefill through first token' if phase=='prefill' else '31 subsequent decode steps (average per token)',fontsize=11)
 ax.set_ylim(0,max(bottom)*1.18);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 for x,r in enumerate(subset):
  value=r['wall_ms'];rate=r['effective_prefill_tps'] if phase=='prefill' else r['decode_tps'];ax.text(x,value+max(bottom)*.015,f'{value:,.1f} ms\n{rate:.1f} t/s',ha='center',va='bottom',fontsize=9)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,.055))
fig.suptitle('RK3588 phase costs • identical FP16 model and decode inputs',y=.97,fontsize=14)
fig.text(.02,.015,'One measured diagnostic profile per route, after two full warmups; all clocks audited. Device times include memory stalls.\n* GPU prefill projections / NPU decode projections / GPU decode attention / CPU classifier and host math. Children counted once.',fontsize=8)
fig.tight_layout(rect=(0,.31,1,.93));files={}
for ext in ['png','jpg','svg']:
 path=root/('phase-costs.'+ext);fig.savefig(path,dpi=180,facecolor='white');files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
plt.close(fig)
(root/'phase-costs-provenance.json').write_text(json.dumps({'input_sha256':hashlib.sha256((root/args.input).read_bytes()).hexdigest(),'files_sha256':files},indent=2)+'\n')
print('Rendered phase cost PNG/JPG/SVG')
