"""Tokenize a prompt and run the direct-register FP16 engine on RK3588."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('--tokenizer', type=Path)
parser.add_argument('--prompt', default='Hello')
parser.add_argument('--new-tokens', type=int, default=32)
parser.add_argument('--warmups', type=int, default=2)
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()
model = args.model.resolve()
tokenizer_path = args.tokenizer or model.parent/'fp16-hf/tokenizer.json'
tokenizer = Tokenizer.from_file(str(tokenizer_path))
ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
if not ids or args.new_tokens<2 or len(ids)+args.new_tokens>512:
    parser.error('Use a nonempty prompt and at least two new tokens; total context is limited to 512 tokens')
env = dict(os.environ)
for key,value in {'OMP_NUM_THREADS':'4','GOMP_SPINCOUNT':'1000','NPU_CORES':'3','NPU_FUSED':'1',
                  'NPU_DOMAIN_ID':'1','NPU_ATTENTION':'1','NPU_SPLIT_DOWN':'1','NPU_STREAM_FFN':'1',
                  'NPU_CLS_TILE':'8192','WARMUP_RUNS':str(args.warmups)}.items():
    env.setdefault(key,value)
binary = Path(__file__).resolve().parents[1]/'runq-fp16'
with tempfile.TemporaryDirectory(prefix='qwen3-fp16-') as directory:
    token_path = Path(directory)/'input.tokens'
    token_path.write_text(' '.join(map(str,ids))+'\n')
    result = subprocess.run(['taskset','-c','4-7',str(binary),str(model),str(token_path),
                             str(args.new_tokens),str(args.runs)],env=env,text=True,capture_output=True)
if result.returncode:
    print(result.stdout,end='')
    print(result.stderr,end='')
    raise SystemExit(result.returncode)
records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"event":')]
runs = [r for r in records if r['event']=='run' and not r['warmup']]
print(tokenizer.decode(runs[-1]['generated_ids'], skip_special_tokens=False))
for run in runs:
    print(f"Warm run {run['run']}: {run['first_token_ms']:.1f} ms to first token; {run['decode_tps']:.2f} tokens/s")
