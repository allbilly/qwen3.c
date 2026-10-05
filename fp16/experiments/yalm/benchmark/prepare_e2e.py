"""Freeze request-timed routes separately from the phase-profile experiments."""
from pathlib import Path
import json,shutil
from cooldown_guard import bounded
root=Path(__file__).resolve().parent

def replace(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)

def prepare(parent,name):
    dest=root/'e2e'/name
    dest.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/'source',dest/'source')
    metadata=json.loads((parent/'source-provenance.json').read_text())
    for path in metadata['source_sha256']:
        if not path.startswith('source/'):
            target=dest/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(parent/path,target)
    for file in ['cpu-linear-check.jsonl','gpu-linear-check.jsonl']:
        shutil.copy2(parent/file,dest/file)
    p=dest/'source/fp16/run.c';s=p.read_text()
    s=replace(s,'        double decode_ms = now_ms() - decode_start;',
      '''        double request_end=now_ms();
        double decode_ms = request_end - decode_start;
        double request_ms = request_end - begin;
        double handoff_ms = decode_start - (begin + first_ms);''')
    s=replace(s,'        gpu_profile_dump("decode",run);route_dump("decode",run);',
      '''        fprintf(stderr,"{\\"event\\":\\"request_wall\\",\\"run\\":%d,\\"warmup\\":%s,\\"prompt_tokens\\":%d,\\"output_tokens\\":%d,\\"request_ms\\":%.6f,\\"prefill_ms\\":%.6f,\\"decode_ms\\":%.6f,\\"handoff_ms\\":%.6f,\\"output_tps\\":%.6f,\\"weights_resident\\":true}\\n",run,run<=0?"true":"false",count,new_tokens,request_ms,first_ms,decode_ms,handoff_ms,new_tokens*1000.0/request_ms);
        gpu_profile_dump("decode",run);route_dump("decode",run);''')
    p.write_text(bounded(s))
    s=(root/'run_matrix.py').read_text()
    s=replace(s,'root=Path(__file__).resolve().parent;roofline=root.parent;matched=roofline.parent',
                  'root=Path(__file__).resolve().parent;roofline=root.parents[2];matched=roofline.parent')
    s=replace(s,"   records.append(record);print(json.dumps(record),flush=True)",
      '''   request_rows=[r for r in rows if r.get('event')=='request_wall' and not r['warmup']]
   assert len(request_rows)==len(measured)
   request_ms=statistics.median(r['request_ms'] for r in request_rows)
   record.update(request_ms=request_ms,output_tps=args.new_tokens*1000/request_ms,
     request_range=[min(r['request_ms'] for r in request_rows),max(r['request_ms'] for r in request_rows)],
     handoff_ms=statistics.median(r['handoff_ms'] for r in request_rows))
   records.append(record);print(json.dumps(record),flush=True)''')
    if parent.name=='cpu-blas-attention':
        s=replace(s,"if args.profile:env.update(GPU_PROFILE='1',LINEAR_PROFILE='1',NPU_PROFILE='1')",
                    "env['CPU_ATTN_BLAS']='1'\nif args.profile:env.update(GPU_PROFILE='1',LINEAR_PROFILE='1',NPU_PROFILE='1')")
        s=replace(s,"'ROUTE_PROFILE','LD_LIBRARY_PATH'","'ROUTE_PROFILE','CPU_ATTN_BLAS','LD_LIBRARY_PATH'")
    (dest/'run_matrix.py').write_text(s)
    command=[v.replace(str(parent),str(dest)) if v.startswith(str(parent)) else v for v in metadata['command']]
    (dest/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
    (dest/'parent-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (dest/'timing-scope.json').write_text(json.dumps({
      'phase':'Host monotonic spans include activation packing, device copies, submits, blocking waits, output unpacking and host math within that phase.',
      'request':'One independent monotonic span from prompt prefill start through the last of 32 greedy output tokens; includes phase transition, counter reset and interphase diagnostic writes.',
      'excluded':'Model initialization, pre-request thermal cooldown, input-ID file reads/tokenization and post-request output/report writes.',
      'handoff':'Residual request time between first-token completion and decode start. Cache preparation that executes in the first decode forward is included in decode, not this residual.',
      'warmup':'Two full requests before measurement; first warmup writes logits to disk and is excluded.'},indent=2)+'\n')
    print('Prepared',name)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',default='.');p.add_argument('--name',default='baseline');args=p.parse_args()
    prepare((root/args.parent).resolve(),args.name)
