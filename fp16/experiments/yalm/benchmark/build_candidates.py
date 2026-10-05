"""Compile prepared candidates with source hashes; hardware execution is separate."""
from pathlib import Path
import hashlib,json,shutil,subprocess,argparse
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+');args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for dest in (root/'variants').iterdir():
    if args.only and dest.name not in args.only:continue
    if not (dest/'build-command.json').exists():continue
    command=json.loads((dest/'build-command.json').read_text())
    with (dest/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    original=json.loads((root/'matrix/source-provenance.json').read_text())
    source_names=list(original['source_sha256'])
    shape_metadata={'attention_prefill_max_rows':128} if dest.name=='attention-row128' else {}
    (dest/'source-provenance.json').write_text(json.dumps({'command':command,'variant':dest.name,
        **shape_metadata,
        'binary_sha256':sha(dest/'runq-routes'),'selected_binary_sha256':original['selected_binary_sha256'],
        'source_sha256':{name:sha(dest/name) for name in source_names}},indent=2)+'\n')
    if dest.name in ['parallel-softmax','fused-attention']:
        shutil.copy2(root.parent/'hybrid/attention_check.c',dest/'attention_check.c')
        command=['gcc','-O3','-march=native',str(dest/'attention_check.c'),str(dest/'source/fp16/gpu_attention.c'),
                 '-I'+str(dest/'source/fp16'),'-lm','-l:libOpenCL.so.1','-o',str(dest/'attention-check')]
        with (dest/'attention-build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    if dest.name=='gpu-tile16':
        command=json.loads((dest/'build-command.json').read_text())
        command=[str(dest/'linear_check.c') if v==str(dest/'source/fp16/run.c') else (str(dest/'linear-check') if v==str(dest/'runq-routes') else v) for v in command]
        with (dest/'linear-build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    if dest.name=='cpu-blas-attention':
        command=json.loads((dest/'build-command.json').read_text())
        command=[str(dest/'attention_check.c') if v==str(dest/'source/fp16/run.c') else (str(dest/'attention-check') if v==str(dest/'runq-routes') else v) for v in command]
        with (dest/'attention-build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Built',dest.name,sha(dest/'runq-routes'),flush=True)
