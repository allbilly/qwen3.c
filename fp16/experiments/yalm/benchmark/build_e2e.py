"""Compile frozen complete-request timer builds with source provenance."""
from pathlib import Path
import argparse,json,hashlib,subprocess
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+');args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for dest in sorted((root/'e2e').iterdir()):
    if args.only and dest.name not in args.only:continue
    if not (dest/'build-command.json').exists():continue
    command=json.loads((dest/'build-command.json').read_text())
    with (dest/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata=json.loads((dest/'parent-provenance.json').read_text())
    shape_metadata={k:metadata[k] for k in ['attention_prefill_max_rows'] if k in metadata}
    (dest/'source-provenance.json').write_text(json.dumps({'command':command,'variant':dest.name,
      **shape_metadata,
      'binary_sha256':sha(dest/'runq-routes'),'selected_binary_sha256':metadata['selected_binary_sha256'],
      'source_sha256':{name:sha(dest/name) for name in metadata['source_sha256']}},indent=2)+'\n')
    print('Built request timer',dest.name,sha(dest/'runq-routes'),flush=True)
