#!/usr/bin/env python3
"""Export a local dense Qwen3 safetensors snapshot to the existing Q8 format.

Requires NumPy and Jinja2, without loading a PyTorch model into memory.
"""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np
from jinja2 import Template


class Weights:
    def __init__(self, root):
        self.entries = {}
        for path in sorted(root.glob('*.safetensors')):
            with path.open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(size))
            for name, entry in header.items():
                if name == '__metadata__':
                    continue
                self.entries[name] = (path, size + 8, entry)

    def tensor(self, name):
        path, base, entry = self.entries[name]
        kind = entry['dtype']
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}[kind]
        data = np.memmap(path, dtype=dtype, mode='r',
                         offset=base + entry['data_offsets'][0], shape=tuple(entry['shape']))
        return data, kind


def floats(data, kind):
    if kind == 'BF16':
        return (data.astype('<u4') << 16).view('<f4')
    return data.astype('<f4')


def quantized(f, source, name, group_size):
    data, kind = source.tensor(name)
    data = data.reshape(-1)
    if data.size % group_size:
        raise ValueError(f'{name} is not divisible by group size')
    start = f.tell()
    scale_start = start + data.size
    for offset in range(0, data.size, 1048576):
        values = floats(data[offset:offset + 1048576], kind).reshape(-1, group_size)
        scale = np.max(np.abs(values), axis=1) / np.float32(127)
        safe_scale = np.where(scale == 0, np.float32(1), scale)
        q = np.rint(values / safe_scale[:, None]).astype('i1')
        f.seek(start + offset)
        f.write(q.tobytes())
        f.seek(scale_start + offset // group_size * 4)
        f.write(scale.astype('<f4').tobytes())
    f.seek(scale_start + data.size // group_size * 4)
    print(f'Exported {name}', flush=True)


def tokenizer(root, output, config):
    spec = json.loads((root / 'tokenizer.json').read_text())
    tokenizer_config = json.loads((root / 'tokenizer_config.json').read_text())
    vocab = dict(spec['model']['vocab'])
    vocab.update({t['content']: t['id'] for t in spec['added_tokens']})
    by_id = {value: key for key, value in vocab.items()}
    visible = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    inverse = {chr(b): b for b in visible}
    inverse.update({chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in visible)})
    ranks = {''.join(merge if isinstance(merge, list) else merge.split()): i
             for i, merge in enumerate(spec['model']['merges'])}
    bos = vocab.get(tokenizer_config.get('bos_token'), 151643)
    eos = vocab[tokenizer_config['eos_token']]
    with Path(str(output) + '.tokenizer').open('wb') as f:
        max_len = max(len(t.encode('utf8')) for t in vocab)
        f.write(struct.pack('<3I', max_len, bos, eos))
        for i in range(config['vocab_size']):
            token = by_id.get(i, '')
            token_bytes = b''.join(bytes([inverse[c]]) if c in inverse else c.encode('utf8') for c in token)
            score = -math.log(ranks[token] + 1) if token in ranks else -1e6
            f.write(struct.pack('<fI', score, len(token_bytes)))
            f.write(token_bytes)
    template = Template(tokenizer_config['chat_template'])
    for system, thinking, suffix in [(False,False,''), (False,True,'.with-thinking'),
                                     (True,False,'.with-system'), (True,True,'.with-system-and-thinking')]:
        messages = ([{'role': 'system', 'content': '%s'}] if system else []) + [{'role': 'user', 'content': '%s'}]
        rendered = template.render(messages=messages, add_generation_prompt=True, enable_thinking=thinking)
        Path(str(output) + '.template' + suffix).write_text(rendered)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('snapshot', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--context', type=int, default=512)
    args = cli.parse_args()
    c = json.loads((args.snapshot / 'config.json').read_text())
    if c['model_type'] != 'qwen3' or c.get('attention_bias'):
        cli.error('requires a dense Qwen3 model without attention biases')
    if not 1 <= args.context <= c['max_position_embeddings']:
        cli.error('invalid context length')
    source = Weights(args.snapshot)
    if not source.entries:
        cli.error('snapshot contains no safetensors')
    dim = c['hidden_size']
    hidden = c['intermediate_size']
    layers = c['num_hidden_layers']
    head_dim = c.get('head_dim', dim // c['num_attention_heads'])
    gs = 64
    while dim % gs or hidden % gs or (c['num_attention_heads'] * head_dim) % gs:
        gs //= 2
    tied = bool(c.get('tie_word_embeddings'))
    header = struct.pack('<12i', 0x616a6331, 1, dim, hidden, layers,
                         c['num_attention_heads'], c['num_key_value_heads'], c['vocab_size'],
                         args.context, head_dim, int(tied), gs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        f.write(header + bytes(256 - len(header)))
        norms = [f'model.layers.{i}.input_layernorm.weight' for i in range(layers)]
        norms += [f'model.layers.{i}.post_attention_layernorm.weight' for i in range(layers)]
        norms += ['model.norm.weight']
        norms += [f'model.layers.{i}.self_attn.q_norm.weight' for i in range(layers)]
        norms += [f'model.layers.{i}.self_attn.k_norm.weight' for i in range(layers)]
        for name in norms:
            data, kind = source.tensor(name)
            f.write(floats(data, kind).tobytes())
        quantized(f, source, 'model.embed_tokens.weight', gs)
        for part in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                     'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']:
            for i in range(layers):
                quantized(f, source, f'model.layers.{i}.{part}.weight', gs)
        if not tied:
            quantized(f, source, 'lm_head.weight', gs)
    tokenizer(args.snapshot, args.output, c)


if __name__ == '__main__':
    main()
