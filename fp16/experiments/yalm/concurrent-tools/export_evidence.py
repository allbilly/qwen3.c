"""Export small immutable audit/log evidence; keep model/logit binaries external."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('name');args=p.parse_args()
out=root/args.directory;target=Path('/home/orangepi/qwen3.c/fp16/experiments/yalm/evidence')/args.name;target.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
for source in sorted(out.iterdir()):
 if not source.is_file():continue
 if source.name=='clock-samples.jsonl':
  dest=target/(source.name+'.gz');dest.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
 elif source.suffix in ['.json','.jsonl','.tokens']:
  dest=target/source.name;shutil.copy2(source,dest)
 else:continue
 manifest[source.name]={'original_sha256':sha(source),'export':dest.name,'export_sha256':sha(dest)}
(target/'export-manifest.json').write_text(json.dumps({'workspace_directory':str(out),'files':manifest,
 'external_artifacts':'Shared FP16 model, full per-vocabulary logits and compiled binary remain in the comparison workspace; source/binary/input hashes are retained here.'},indent=2)+'\n')
print('Exported',len(manifest),'evidence files to',target)
