"""Decompose transfer-inclusive root profiles without double counting children."""
from pathlib import Path
import argparse,json
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--audit',default='phase-audit.json');p.add_argument('--output',default='phase-costs.json');args=p.parse_args()
audit=json.loads((root/args.audit).read_text());rows=[]
for record in audit['records']:
 for profile in record['root_profiles']:
  phase=profile['phase'];run=profile['run'];steps=profile['steps']
  cpu=next(v for v in record['cpu_projection_profiles'] if v['phase']==phase and v['run']==run)
  gpu=next((v for v in record['gpu_projection_profiles'] if v['phase']==phase and v['run']==run),None)
  attn=next((v for v in record['gpu_attention_profiles'] if v['phase']==phase and v['run']==run),None)
  details={}
  for owner in profile['details'].values():
   for name,value in owner.items():
    a=details.setdefault(name,{'ms':0,'driver_ms':0,'bytes':0,'calls':0})
    for k in a:a[k]+=value[k]
  wall=profile['wall_ms'];coverage=sum(v['ms'] for v in profile['root'].values())*steps/wall
  assert .98<coverage<1.001
  components={
   'NPU driver (compute + memory stalls)':details['submit']['driver_ms'],
   'NPU submit host/interrupt residual':details['submit']['ms']-details['submit']['driver_ms'],
   'NPU DMA synchronization':details['sync_to_device']['ms']+details['sync_from_device']['ms'],
   'NPU pack/copy/unpack/register':sum(details[k]['ms'] for k in ['register_update','fp16_native_pack','input_memcpy','output_copy_unpack_reduce','attention_kv_pack']),
   'CPU projections (round/expand/math/layout)':cpu['wall_ms'],
   'GPU projection kernels':gpu['kernel_ms'] if gpu else 0,
   'GPU projection device copies':(gpu['upload_ms']+gpu['download_ms']) if gpu else 0,
   'GPU projection input packing':gpu['pack_ms'] if gpu else 0,
   'GPU projection host/API/wait residual':(gpu['wall_ms']-gpu['kernel_ms']-gpu['upload_ms']-gpu['download_ms']-gpu['pack_ms']) if gpu else 0,
   'GPU attention kernels':sum(attn['kernels'].values()) if attn else 0,
   'GPU attention device copies':(attn['upload_device_ms']+attn['download_device_ms']) if attn else 0,
   'GPU attention host/API/wait residual':(attn['wall_ms']-sum(attn['kernels'].values())-attn['upload_device_ms']-attn['download_device_ms']) if attn else 0,
  }
  assert all(v>=-.001 for v in components.values()),(record['directory'],phase,components)
  components={k:max(0,v) for k,v in components.items()}
  components['CPU remaining host math + uncovered overhead']=wall-sum(components.values())
  assert components['CPU remaining host math + uncovered overhead']>=-.001
  row={'directory':record['directory'],'label':record['label'],'route':record['route'],'prompt':record['prompt'],'phase':phase,'run':run,
   'wall_ms':wall/steps,'root_coverage':coverage,'components_ms':{k:v/steps for k,v in components.items()},
   'root_ms':{k:v['ms'] for k,v in profile['root'].items()},
   'npu_detail_ms':{k:v['ms']/steps for k,v in details.items()},
   'npu_detail_bytes':{k:v['bytes']/steps for k,v in details.items()},
   'npu_driver_ms':details['submit']['driver_ms']/steps,
   'gpu_projection':{k:gpu[k]/steps for k in ['calls','kernels','upload_bytes','download_bytes','pack_ms','kernel_ms','upload_ms','download_ms','wall_ms']} if gpu else None,
   'gpu_attention':{k:attn[k]/steps for k in ['calls','kernel_calls','upload_bytes','download_bytes','wall_ms','upload_device_ms','download_device_ms']} if attn else None}
  row['effective_prefill_tps']=1000*record['prompt']/wall if phase=='prefill' else None
  row['decode_tps']=1000*steps/wall if phase=='decode' else None
  rows.append(row)
(root/args.output).write_text(json.dumps({'rows':rows,'interpretation':'GPU event execution and NPU driver time include memory stalls. Logical copy byte counters are not measured DRAM traffic. Root spans are inclusive; children are subtracted once.'},indent=2)+'\n')
print(json.dumps({'profiles':len(rows),'coverage_range':[min(r['root_coverage'] for r in rows),max(r['root_coverage'] for r in rows)]}))
