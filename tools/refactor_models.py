#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_fp16(snapshot, output, q8):
    from ane.export_checkpoint import Weights, floats
    source = Weights(snapshot)
    with q8.open('rb') as stream:
        header = list(struct.unpack('<12i',stream.read(48)))
    header[1] = 2
    layers = header[4]
    norms = [f'model.layers.{i}.input_layernorm.weight' for i in range(layers)]
    norms += [f'model.layers.{i}.post_attention_layernorm.weight' for i in range(layers)] + ['model.norm.weight']
    norms += [f'model.layers.{i}.self_attn.q_norm.weight' for i in range(layers)]
    norms += [f'model.layers.{i}.self_attn.k_norm.weight' for i in range(layers)]
    matrices = ['model.embed_tokens.weight']
    for suffix in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                   'mlp.gate_proj','mlp.down_proj','mlp.up_proj']:
        matrices += [f'model.layers.{i}.{suffix}.weight' for i in range(layers)]
    if not header[10]:
        matrices.append('lm_head.weight')
    with output.open('xb') as destination:
        prefix = struct.pack('<12i',*header)
        destination.write(prefix+bytes(256-len(prefix)))
        for name in norms+matrices:
            data,kind = source.tensor(name)
            data = data.reshape(-1)
            for offset in range(0,len(data),1048576):
                half = floats(data[offset:offset+1048576],kind).astype('<f2')
                destination.write(half.astype('<f4').tobytes() if name in norms else half.tobytes())


def main():
    cli = argparse.ArgumentParser(description='Recreate both original refactor-baseline checkpoint formats')
    cli.add_argument('--snapshot',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    args = cli.parse_args()
    source = json.loads((ROOT/'fp16/source-manifest.json').read_text())
    args.snapshot.mkdir(parents=True,exist_ok=True)
    for name,record in source['source_files'].items():
        path = args.snapshot/name
        if not path.exists():
            url = f'https://huggingface.co/{source["model_id"]}/resolve/{source["revision"]}/{name}'
            temporary = path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as output:
                shutil.copyfileobj(response,output,4*1024*1024)
            if sha(temporary)!=record['sha256'] or temporary.stat().st_size!=record['bytes']:
                raise ValueError('Downloaded source differs: '+name)
            temporary.replace(path)
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Checkpoint source differs: '+name)
    args.out.mkdir(parents=True,exist_ok=True)
    q8 = args.out/'Qwen3-0.6B.bin'
    if not q8.exists():
        subprocess.run([sys.executable,ROOT/'ane/export_checkpoint.py',args.snapshot,q8,'--context','512'],check=True)
    expected_q8 = json.loads((ROOT/'ane/benchmarks/m1-qwen3-0.6b.json').read_text())['model']['sha256']
    if sha(q8)!=expected_q8:
        raise ValueError('Q8 version-1 model differs from the original ANE baseline')
    fp16 = args.out/'Qwen3-0.6B.fp16'
    if not fp16.exists():
        prepare_fp16(args.snapshot,fp16,q8)
    expected_fp16 = json.loads((ROOT/'fp16/fp16-manifest.json').read_text())['c_checkpoint_sha256']
    if sha(fp16)!=expected_fp16:
        raise ValueError('FP16 version-2 model differs from the original RKNPU baseline')
    report = {'status':'BOTH_ORIGINAL_CHECKPOINT_FORMATS_REPRODUCED',
              'model_id':source['model_id'],'source_revision':source['revision'],
              'q8_v1':{'sha256':expected_q8,'bytes':q8.stat().st_size},
              'fp16_v2':{'sha256':expected_fp16,'bytes':fp16.stat().st_size},
              'source_manifest_sha256':sha(ROOT/'fp16/source-manifest.json')}
    (args.out/'models.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
