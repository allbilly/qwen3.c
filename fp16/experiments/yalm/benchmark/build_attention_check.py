"""Compile the same full-output primitive check for original and row128 NPU."""
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path(__file__).resolve().parent
candidate=root/'variants/attention-row128'
for parent,rows in [(root,64),(candidate,128)]:
    if parent==root:shutil.copy2(candidate/'npu_attention_check.c',root/'npu_attention_check.c')
    helpers=parent/'source/fp16/model_helpers.h'
    source=(parent/'source/fp16/run.c').read_text()
    assert source.count('int main(int argc, char **argv) {')==1
    helpers.write_text(source.split('int main(int argc, char **argv) {')[0])
    metadata=json.loads((parent/'source-provenance.json').read_text())
    command=metadata['command']
    command=[str(parent/'npu_attention_check.c') if v==str(parent/'source/fp16/run.c')
             else str(parent/'npu-attention-check') if v==str(parent/'runq-routes') else v for v in command]
    command[1:1]=['-I'+str(parent/'source/fp16'),'-DNPU_CHECK_ATTN_ROWS='+str(rows)]
    with (parent/'npu-attention-build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (parent/'npu-attention-check-provenance.json').write_text(json.dumps({
        'command':command,'tile_rows':rows,'source_sha256':sha(parent/'npu_attention_check.c'),
        'model_helpers_sha256':sha(helpers),
        'model_entrypoint_source_sha256':sha(parent/'source/fp16/run.c'),
        'backend_source_provenance':metadata,'binary_sha256':sha(parent/'npu-attention-check')},indent=2)+'\n')
    print('Built NPU full-output primitive checker, rows',rows,flush=True)
