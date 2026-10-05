"""Freeze a request-timed, disjoint FFN channel experiment, keeping the selected runner intact."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parent
parent=root/'e2e/baseline-guarded';dest=root/'e2e/concurrent-ffn'
assert not dest.exists()
metadata=json.loads((parent/'source-provenance.json').read_text())
shutil.copytree(parent/'source',dest/'source')
for name in metadata['source_sha256']:
 if not name.startswith('source/'):
  target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
for name in ['cpu-linear-check.jsonl','gpu-linear-check.jsonl','run_matrix.py','timing-scope.json']:
 shutil.copy2(parent/name,dest/name)
command=[s.replace(str(parent),str(dest)) for s in metadata['command']]
command[command.index(str(dest/'npu_backend.c'))]=str(dest/'concurrent_ffn.c')
command.insert(command.index('-lm'),'-pthread')
metadata['source_sha256']['concurrent_ffn.c']='pending'
metadata['source_sha256']['concurrent_ffn.h']='pending'
(dest/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
(dest/'parent-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
def edit(name,old,new):
 p=dest/name;s=p.read_text();assert s.count(old)==1,(name,old,s.count(old));p.write_text(s.replace(old,new))
edit('router.c','#include "router.h"','#include "router.h"\n#include "concurrent_ffn.h"')
edit('router.c','current_device=position?decode_device:pre_device;}','current_device=position?decode_device:pre_device;ffn_phase(position);}')
edit('router.c','    if(has_gpu&&!linear_gpu_init(c,matrices))return 0;','    if(!ffn_init(ctx,c,matrices))return 0;\n    if(has_gpu&&!linear_gpu_init(c,matrices))return 0;')
edit('router.c','return hw_matmul_fused_to(ctx,combo,layer,input,a,b,c,rows);','if(combo==1&&ffn_active())return ffn_gate_up(ctx,layer,input,a,b,rows);return hw_matmul_fused_to(ctx,combo,layer,input,a,b,c,rows);')
edit('router.c','return hw_stream_ffn_prefill(ctx,layer,input,output,rows);','if(ffn_active())return ffn_prefill(ctx,layer,input,output,rows);return hw_stream_ffn_prefill(ctx,layer,input,output,rows);')
edit('router.c','linear_gpu_clear();}','linear_gpu_clear();ffn_clear();}')
edit('router.c','    linear_gpu_dump(phase,run);','    linear_gpu_dump(phase,run);ffn_dump(phase,run);')
edit('router.c','if(has_npu)hw_shutdown(ctx);','ffn_close();if(has_npu)hw_shutdown(ctx);')
p=dest/'run_matrix.py';s=p.read_text();s=s.replace("'ROUTE_PROFILE','LD_LIBRARY_PATH'","'ROUTE_PROFILE','FFN_CPU_CHANNELS','FFN_GPU_CHANNELS','FFN_MODE','FFN_PROFILE','LD_LIBRARY_PATH'")
p.write_text(s)
print(dest)
