from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parent;parent=root/'e2e/concurrent-ffn';dest=root/'e2e/concurrent-native'
meta=json.loads((parent/'source-provenance.json').read_text());assert not dest.exists()
shutil.copytree(parent/'source',dest/'source')
for name in [*[n for n in meta['source_sha256'] if not n.startswith('source/')],'ffn_check.c','cpu-linear-check.jsonl','gpu-linear-check.jsonl','run_matrix.py','timing-scope.json']:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,p)
cmd=[v.replace(str(parent),str(dest)) for v in meta['command']];(dest/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
meta['concurrent_prefill_block_rows']=64
(dest/'parent-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(dest)
