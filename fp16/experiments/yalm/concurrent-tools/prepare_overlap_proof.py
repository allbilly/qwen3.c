"""Freeze a diagnostic-only observer; native submit arguments are unchanged."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parent;parent=root/'e2e/concurrent-native';dest=root/'e2e/concurrent-overlap-proof'
meta=json.loads((parent/'source-provenance.json').read_text());assert not dest.exists()
shutil.copytree(parent/'source',dest/'source')
for name in [*[n for n in meta['source_sha256'] if not n.startswith('source/')],'cpu-linear-check.jsonl','gpu-linear-check.jsonl','run_matrix.py','timing-scope.json']:
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
cmd=[v.replace(str(parent),str(dest)) for v in meta['command']];(dest/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n');(dest/'parent-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
p=dest/'concurrent_ffn.c';s=p.read_text();s=s.replace('logical_cpu_ops;', 'logical_cpu_ops,gpu_running_across_submit,cpu_spans_submit,all_three_spans_submit;')
s=s.replace('    double npu_start=ff_now();cl_int before=CL_COMPLETE,after=CL_COMPLETE;', '    double submit_start=0,submit_end=0;int gpu_across=0;\n    double npu_start=ff_now();cl_int before=CL_COMPLETE,after=CL_COMPLETE;')
old='        if(submit_bundle(p[0])<0){perror("Concurrent native gate/up");exit(1);}'
new='''        cl_int running_before=CL_COMPLETE,running_after=CL_COMPLETE;
        if(ff_gpu&&ff_events)ff_check(clGetEventInfo(kernels[0],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(running_before),&running_before,NULL),"kernel before submit");
        submit_start=ff_now();
        if(submit_bundle(p[0])<0){perror("Concurrent native gate/up");exit(1);}
        submit_end=ff_now();
        if(ff_gpu&&ff_events)ff_check(clGetEventInfo(kernels[0],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(running_after),&running_after,NULL),"kernel after submit");
        gpu_across=running_before==CL_RUNNING&&running_after==CL_RUNNING;
'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('    double joined_end=ff_now();', '''    if(submit_start){
        int cpu_across=ff_cpu&&ff_cpu_start<=submit_start&&ff_cpu_end>=submit_end;
        ff_profile.gpu_running_across_submit+=gpu_across;
        ff_profile.cpu_spans_submit+=cpu_across;
        ff_profile.all_three_spans_submit+=gpu_across&&cpu_across;
    }
    double joined_end=ff_now();''')
s=s.replace('\\"event_profiling\\":%s}\\n",phase', '\\"gpu_running_across_submit\\":%llu,\\"cpu_spans_submit\\":%llu,\\"all_three_spans_submit\\":%llu,\\"event_profiling\\":%s}\\n",phase')
s=s.replace('(unsigned long long)ff_profile.logical_cpu_ops,ff_events?', '(unsigned long long)ff_profile.logical_cpu_ops,(unsigned long long)ff_profile.gpu_running_across_submit,(unsigned long long)ff_profile.cpu_spans_submit,(unsigned long long)ff_profile.all_three_spans_submit,ff_events?')
p.write_text(s)
print('Prepared diagnostic proof; build and run only after current device jobs end')
