"""Q8 checkpoint layout and benchmark inputs, without Apple dependencies."""
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIELDS = ('magic', 'version', 'dim', 'hidden_dim', 'layers', 'heads',
                 'kv_heads', 'vocab', 'context', 'head_dim', 'tied', 'group_size')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = [p for p in (ROOT/'macos').iterdir() if p.suffix in ('.py', '.swift')]
    paths.append(ROOT/'tools/benchmark_queue.py')
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Checkpoint:
    """Read exactly the version-1 layout in runq.c, retaining Q8 provenance."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        with self.path.open('rb') as stream:
            header = stream.read(256)
        if len(header) != 256:
            raise ValueError('truncated checkpoint header')
        self.config = dict(zip(CONFIG_FIELDS, struct.unpack('<12i', header[:48])))
        c = self.config
        if c['magic'] != 0x616a6331 or c['version'] != 1:
            raise ValueError('requires a qwen3.c version-1 Q8 checkpoint')
        if any(c[k] <= 0 for k in CONFIG_FIELDS[2:] if k != 'tied'):
            raise ValueError('invalid checkpoint dimensions')
        if c['heads'] % c['kv_heads'] or c['head_dim'] % 2 or c['tied'] not in (0, 1):
            raise ValueError('invalid GQA, RoPE or classifier configuration')
        self.offset = 256
        self.norms = {}
        for name, shape in [('attention', (c['layers'], c['dim'])),
                            ('ffn', (c['layers'], c['dim'])),
                            ('final', (c['dim'],)),
                            ('q', (c['layers'], c['head_dim'])),
                            ('k', (c['layers'], c['head_dim']))]:
            count = int(np.prod(shape))
            self.norms[name] = np.memmap(self.path, mode='r', dtype='<f4',
                                        offset=self.offset, shape=shape)
            self.offset += count * 4
        self.tensors = {}
        self._add('embedding', (c['vocab'], c['dim']))
        shapes = {'q': (c['heads'] * c['head_dim'], c['dim']),
                  'k': (c['kv_heads'] * c['head_dim'], c['dim']),
                  'v': (c['kv_heads'] * c['head_dim'], c['dim']),
                  'o': (c['dim'], c['heads'] * c['head_dim']),
                  'gate': (c['hidden_dim'], c['dim']),
                  'down': (c['dim'], c['hidden_dim']),
                  'up': (c['hidden_dim'], c['dim'])}
        for name, shape in shapes.items():
            for layer in range(c['layers']):
                self._add(f'{layer}.{name}', shape)
        if c['tied']:
            self.tensors['head'] = self.tensors['embedding']
        else:
            self._add('head', (c['vocab'], c['dim']))
        if self.offset != self.path.stat().st_size:
            raise ValueError('checkpoint payload size does not match its header')

    def _add(self, name, shape):
        size = int(np.prod(shape))
        gs = self.config['group_size']
        if size % gs or shape[-1] % gs:
            raise ValueError('tensor dimensions must be divisible by quantization group size')
        self.tensors[name] = (self.offset, shape)
        self.offset += size + size // gs * 4

    def matrix(self, name):
        offset, shape = self.tensors[name]
        size, gs = int(np.prod(shape)), self.config['group_size']
        values = np.memmap(self.path, mode='r', dtype='i1', offset=offset, shape=(size // gs, gs))
        scales = np.memmap(self.path, mode='r', dtype='<f4', offset=offset + size, shape=(size // gs, 1))
        result = (values.astype('f4') * scales).reshape(shape).astype('f2')
        if not np.isfinite(result).all():
            raise ValueError(f'nonfinite FP16 matrix: {name}')
        return result


def load_prompts(path, vocab, context, new_tokens):
    prompts = json.loads(Path(path).read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('prompts must be a nonempty JSON list')
    names = set()
    for p in prompts:
        if not isinstance(p.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', p['name']):
            raise ValueError('prompt names must use letters, digits, underscores or hyphens')
        if p['name'] in names:
            raise ValueError('duplicate prompt name')
        names.add(p['name'])
        ids = p['ids']
        if not ids or any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError('invalid prompt token IDs')
        if len(ids) + new_tokens - 1 > context:
            raise ValueError('prompt plus decode exceeds logical context capacity')
    return prompts


def quality(reference, actual):
    reference, actual = np.asarray(reference, 'f8'), np.asarray(actual, 'f8')
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('logits must be finite with identical shapes')
    denominator = np.linalg.norm(reference)
    nrmse = float(np.linalg.norm(actual - reference) / max(denominator, 1e-30))
    a = reference - reference.max(axis=-1, keepdims=True)
    b = actual - actual.max(axis=-1, keepdims=True)
    logp = a - np.log(np.exp(a).sum(axis=-1, keepdims=True))
    logq = b - np.log(np.exp(b).sum(axis=-1, keepdims=True))
    kl = float(np.max((np.exp(logp) * (logp - logq)).sum(axis=-1)))
    return {'nrmse': nrmse, 'max_abs_error': float(np.max(np.abs(actual - reference))),
            'max_kl': max(0., kl), 'top1_mismatches': int(np.sum(actual.argmax(-1) != reference.argmax(-1)))}
