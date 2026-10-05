"""Apply each exported patch in isolation and compare every compiled source hash."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile
root=Path(__file__).resolve().parent;repo=Path('/home/orangepi/qwen3.c/fp16/experiments/yalm')
manifest=json.loads((repo/'patches/manifest.json').read_text());base=json.loads((root/'matrix-cooled/source-provenance.json').read_text());checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,row in manifest.items():
 target_meta=json.loads(Path(row['source_provenance']).read_text());patch=repo/'patches'/(name+'.patch');assert sha(patch)==row['patch_sha256']
 with tempfile.TemporaryDirectory(prefix='qwen3-yalm-patch-') as temp:
  dest=Path(temp)
  for path in [*base['source_sha256'],'attention.cl','embed.py']:
   file=dest/path;file.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/path,file)
  r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(patch)],cwd=dest,capture_output=True,text=True);assert r.returncode==0,(name,r.stdout,r.stderr)
  for raw,header,symbol in [('linear.cl','linear_source.h','linear_source'),('attention.cl','source/fp16/gpu_source.h','gpu_source')]:
   subprocess.run(['python3',str(dest/'embed.py'),str(dest/raw),str(dest/header),symbol],check=True)
  for path,digest in target_meta['source_sha256'].items():assert sha(dest/path)==digest,(name,path)
  checks.append({'name':name,'every_compiled_source_matches':True,'binary_sha256':target_meta['binary_sha256']})
(root/'patch-audit.json').write_text(json.dumps({'checks':checks},indent=2)+'\n');print('All six patches reproduce every externally compiled source hash')
