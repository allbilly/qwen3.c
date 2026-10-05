"""Calibrate GPU event timestamps against bounded host marker spans."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parent;parent=root/'e2e/concurrent-overlap-proof';dest=root/'e2e/concurrent-calibrated-proof'
meta=json.loads((parent/'source-provenance.json').read_text());assert not dest.exists()
shutil.copytree(parent/'source',dest/'source')
for name in [*[n for n in meta['source_sha256'] if not n.startswith('source/')],'cpu-linear-check.jsonl','gpu-linear-check.jsonl','run_matrix.py','timing-scope.json']:
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
cmd=[v.replace(str(parent),str(dest)) for v in meta['command']];(dest/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n');(dest/'parent-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
p=dest/'concurrent_ffn.c';s=p.read_text()
s=s.replace('all_three_spans_submit;', 'all_three_spans_submit,gpu_calibrated_contains_submit,all_three_calibrated_contains_submit,valid_clock_maps;')
pos=s.index('static void *ff_cpu_worker')
s=s[:pos]+'''// Marker END lies between host timestamps captured before enqueue and after wait.
// Both marker bounds must admit the same clock offset; widen by 10us for rounding.
static void ff_clock_bounds(double *low,double *high){
    cl_event marker;double begin=ff_now();
    ff_check(clEnqueueMarkerWithWaitList(ff_queue,0,NULL,&marker),"clock marker");
    ff_check(clWaitForEvents(1,&marker),"clock marker wait");double end=ff_now();
    cl_ulong device_end;ff_check(clGetEventProfilingInfo(marker,CL_PROFILING_COMMAND_END,sizeof(device_end),&device_end,NULL),"clock marker end");clReleaseEvent(marker);
    *low=begin-device_end/1e6-.01;*high=end-device_end/1e6+.01;
}
'''+s[pos:]
s=s.replace('    double rounded_end=ff_now();', '    double rounded_end=ff_now();\n    double clock_low=0,clock_high=0;int calibrated=ff_gpu&&ff_events&&ff_native_mode;\n    if(calibrated)ff_clock_bounds(&clock_low,&clock_high);')
old='if(ff_events){ff_profile.gpu_upload_ms+=ff_event_ms(upload);'
new='''if(ff_events){
        if(calibrated){
            double low,high;ff_clock_bounds(&low,&high);if(low>clock_low)clock_low=low;if(high<clock_high)clock_high=high;
            if(clock_low<=clock_high){
                ff_profile.valid_clock_maps++;
                cl_ulong start,end;ff_check(clGetEventProfilingInfo(kernels[0],CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL),"calibrated start");ff_check(clGetEventProfilingInfo(kernels[0],CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL),"calibrated end");
                int contains=start/1e6+clock_high<=submit_start&&end/1e6+clock_low>=submit_end;
                ff_profile.gpu_calibrated_contains_submit+=contains;
                ff_profile.all_three_calibrated_contains_submit+=contains&&ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end;
            }
        }
        ff_profile.gpu_upload_ms+=ff_event_ms(upload);'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('\\"event_profiling\\":%s}\\n",phase', '\\"valid_clock_maps\\":%llu,\\"gpu_calibrated_contains_submit\\":%llu,\\"all_three_calibrated_contains_submit\\":%llu,\\"event_profiling\\":%s}\\n",phase')
s=s.replace('(unsigned long long)ff_profile.all_three_spans_submit,ff_events?', '(unsigned long long)ff_profile.all_three_spans_submit,(unsigned long long)ff_profile.valid_clock_maps,(unsigned long long)ff_profile.gpu_calibrated_contains_submit,(unsigned long long)ff_profile.all_three_calibrated_contains_submit,ff_events?')
p.write_text(s)
print('Prepared marker-calibrated diagnostic')
