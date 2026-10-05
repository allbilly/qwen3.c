"""Rebuild the frozen runner; no hardware jobs are started by this script."""
from pathlib import Path
import argparse,hashlib,json,subprocess

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--npu-include');args=p.parse_args()
meta=json.loads((root/'source-provenance.json').read_text())
for name,digest in meta['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
old=str(Path(meta['command'][-1]).parent)
cmd=[value.replace(old,str(root)) for value in meta['command']]
if args.npu_include:
    index=next(i for i,v in enumerate(cmd) if v.startswith('-I'));cmd[index]='-I'+str(Path(args.npu_include).resolve())
cmd[-1]=str(Path(args.output).resolve())
subprocess.run(cmd,check=True)
digest=hashlib.sha256(Path(cmd[-1]).read_bytes()).hexdigest()
print(json.dumps({'binary_sha256':digest,'matches_tested_binary':digest==meta['binary_sha256'],'command':cmd},indent=2))
if not args.npu_include:assert digest==meta['binary_sha256']
