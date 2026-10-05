"""Recompute physical overlap from raw marker, kernel, CPU and submit timestamps."""
from pathlib import Path
import argparse,hashlib,json,math
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);args=p.parse_args()
root=Path(__file__).resolve().parent;folder=root/args.directory
log=next(folder.glob('p256-*.jsonl'));rows=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((folder/'source-provenance.json').read_text());assert sha(folder/'runq-routes')==meta['binary_sha256']
for name,digest in meta['source_sha256'].items():assert sha(folder.parent/name)==digest
pending=[];results=[]
for row in rows:
 if row.get('event')=='ffn_overlap_trace':pending.append(row)
 elif row.get('event')=='ffn_split_profile' and row['phase']=='prefill':
  assert len(pending)==row['valid_clock_maps']==row['calls']==112
  assert [r['job'] for r in pending]==list(range(1,113))
  gpu=0;all_three=0;max_bound=0;max_submit=0
  for r in pending:
   assert r['layer']==(r['job']-1)//4 and r['clock_margin_ms']==.01
   assert all(math.isfinite(v) for k,v in r.items() if k.endswith('_ms'))
   assert r['marker_before_host_start_ms']<=r['marker_before_host_end_ms']
   assert r['marker_after_host_start_ms']<=r['marker_after_host_end_ms']
   assert r['marker_before_device_end_ms']<=r['gpu_kernel_start_ms']<=r['gpu_kernel_end_ms']<=r['marker_after_device_end_ms']
   lows=[r[f'marker_{t}_host_start_ms']-r[f'marker_{t}_device_end_ms']-.01 for t in ['before','after']]
   highs=[r[f'marker_{t}_host_end_ms']-r[f'marker_{t}_device_end_ms']+.01 for t in ['before','after']]
   low,high=max(lows),min(highs);assert low<=high
   assert r['npu_submit_start_ms']<=r['npu_submit_end_ms'] and r['cpu_start_ms']<=r['cpu_end_ms']
   contains=r['gpu_kernel_start_ms']+high<=r['npu_submit_start_ms'] and r['gpu_kernel_end_ms']+low>=r['npu_submit_end_ms']
   triple=contains and r['cpu_start_ms']<=r['npu_submit_start_ms'] and r['cpu_end_ms']>=r['npu_submit_end_ms']
   assert contains==r['gpu_contains_submit'] and triple==r['all_three_contains_submit']
   gpu+=contains;all_three+=triple;max_bound=max(max_bound,high-low);max_submit=max(max_submit,r['npu_submit_end_ms']-r['npu_submit_start_ms'])
  assert gpu==row['gpu_calibrated_contains_submit'] and all_three==row['all_three_calibrated_contains_submit']
  assert all_three>0
  results.append({'run':row['run'],'warmup':row['warmup'],'jobs':112,'valid_clock_maps':112,'gpu_contains_submit':gpu,'all_three_contains_submit':all_three,'max_clock_offset_interval_ms':max_bound,'max_submit_ms':max_submit})
  pending=[]
assert not pending and [r['run'] for r in results]==[-1,0,1]
report={'records':results,'raw_log_sha256':sha(log),'binary_sha256':meta['binary_sha256'],
 'scope':'Each marker END is bounded by host timestamps before enqueue and after wait. Intersect before/after offset intervals, widened 10us per endpoint. Conservative GPU interval must contain the complete blocking NPU submit interval, and CPU worker interval must also contain it. GPU timestamps use the OpenCL nanosecond scale; clock-map consistency checked for every job. Diagnostic instrumentation changes scheduling; this is overlap evidence, not unprofiled throughput. NPU submit includes driver work and memory stalls, not isolated MAC time.'}
(root/args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
