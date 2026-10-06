"""Export immutable logs, first logits and harness; keep model/executable external."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('name');args=p.parse_args()
out=root/args.directory;target=Path('/home/orangepi/qwen3.c/fp16/experiments/yalm/evidence')/args.name;target.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
for source in sorted(out.iterdir()):
 if not source.is_file():continue
 if source.name=='clock-samples.jsonl' or source.suffix=='.f32':
  dest=target/(source.name+'.gz');dest.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
 elif source.suffix in ['.json','.jsonl','.tokens','.py']:
  dest=target/source.name;shutil.copy2(source,dest)
 else:continue
 manifest[source.name]={'original_sha256':sha(source),'export':dest.name,'export_sha256':sha(dest)}
(target/'export-manifest.json').write_text(json.dumps({'workspace_directory':str(out),'files':manifest,
 'external_artifacts':'Compiled binary remains in workspace_directory; the shared FP16 model location is recorded in run configs. Full per-vocabulary first logits are exported losslessly as .f32.gz. Source/binary/input hashes and any captured Python harness are retained here.'},indent=2)+'\n')
print('Exported',len(manifest),'evidence files to',target)
