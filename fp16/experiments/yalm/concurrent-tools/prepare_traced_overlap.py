from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parent;parent=root/'e2e/concurrent-calibrated-proof';dest=root/'e2e/concurrent-traced-proof'
meta=json.loads((parent/'source-provenance.json').read_text());assert not dest.exists();shutil.copytree(parent/'source',dest/'source')
for name in [*[n for n in meta['source_sha256'] if not n.startswith('source/')],'cpu-linear-check.jsonl','gpu-linear-check.jsonl','run_matrix.py','timing-scope.json']:
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
cmd=[v.replace(str(parent),str(dest)) for v in meta['command']];(dest/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n');(dest/'parent-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
p=dest/'concurrent_ffn.c';s=p.read_text()
s=s.replace('static void ff_clock_bounds(double *low,double *high){','static double ff_marker_begin[2],ff_marker_end[2],ff_marker_device[2];\nstatic void ff_clock_bounds(double *low,double *high,int index){')
s=s.replace('    *low=begin-device_end/1e6-.01;', '    ff_marker_begin[index]=begin;ff_marker_end[index]=end;ff_marker_device[index]=device_end/1e6;\n    *low=begin-device_end/1e6-.01;')
s=s.replace('ff_clock_bounds(&clock_low,&clock_high);','ff_clock_bounds(&clock_low,&clock_high,0);').replace('ff_clock_bounds(&low,&high);','ff_clock_bounds(&low,&high,1);')
old='                ff_profile.all_three_calibrated_contains_submit+=contains&&ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end;'
new=old+'''
                fprintf(stderr,"{\\"event\\":\\"ffn_overlap_trace\\",\\"job\\":%llu,\\"layer\\":%d,\\"marker_before_host_start_ms\\":%.9f,\\"marker_before_host_end_ms\\":%.9f,\\"marker_before_device_end_ms\\":%.9f,\\"marker_after_host_start_ms\\":%.9f,\\"marker_after_host_end_ms\\":%.9f,\\"marker_after_device_end_ms\\":%.9f,\\"gpu_kernel_start_ms\\":%.9f,\\"gpu_kernel_end_ms\\":%.9f,\\"npu_submit_start_ms\\":%.9f,\\"npu_submit_end_ms\\":%.9f,\\"cpu_start_ms\\":%.9f,\\"cpu_end_ms\\":%.9f,\\"clock_margin_ms\\":0.01,\\"gpu_contains_submit\\":%s,\\"all_three_contains_submit\\":%s}\\n",(unsigned long long)ff_profile.calls+1,layer,ff_marker_begin[0],ff_marker_end[0],ff_marker_device[0],ff_marker_begin[1],ff_marker_end[1],ff_marker_device[1],start/1e6,end/1e6,submit_start,submit_end,ff_cpu_start,ff_cpu_end,contains?"true":"false",contains&&ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end?"true":"false");'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
