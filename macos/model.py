"""Qwen3 decoder blocks with explicit, cached attention for Core ML export."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rms(x, weight):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight


class Layer(nn.Module):
    def __init__(self, checkpoint, index):
        super().__init__()
        c = checkpoint.config
        self.heads, self.kv_heads, self.head_dim = c['heads'], c['kv_heads'], c['head_dim']
        self.repeat = self.heads // self.kv_heads
        for name in ('q', 'k', 'v', 'o', 'gate', 'down', 'up'):
            matrix = checkpoint.matrix(f'{index}.{name}').astype('f4')
            # Exact powers of two move small coefficients into a normal FP16 range.
            # The FP32 output division restores the original decoded coefficients.
            peak = np.max(np.abs(matrix), axis=1)
            exponent = np.clip(np.floor(np.log2(.5 / np.maximum(peak, np.float32(1e-30)))), 0, 5)
            scale = np.exp2(exponent).astype('f4')
            self.register_buffer(name, torch.from_numpy(matrix * scale[:, None]))
            self.register_buffer(name + '_scale', torch.from_numpy(scale))
        for name, key in [('attention_norm', 'attention'), ('ffn_norm', 'ffn'),
                          ('q_norm', 'q'), ('k_norm', 'k')]:
            self.register_buffer(name, torch.tensor(np.array(checkpoint.norms[key][index])))

    def forward(self, x, keys, values, indices, mask, cosine, sine):
        norm = rms(x, self.attention_norm)
        length = x.shape[1]
        q = (F.linear(norm, self.q) / self.q_scale).reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = (F.linear(norm, self.k) / self.k_scale).reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = (F.linear(norm, self.v) / self.v_scale).reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = rms(q, self.q_norm), rms(k, self.k_norm)
        half = self.head_dim // 2
        q = torch.cat((q[..., :half] * cosine - q[..., half:] * sine,
                       q[..., :half] * sine + q[..., half:] * cosine), dim=-1)
        k = torch.cat((k[..., :half] * cosine - k[..., half:] * sine,
                       k[..., :half] * sine + k[..., half:] * cosine), dim=-1)
        slots = indices.to(torch.int64).reshape(1, 1, length, 1).expand(1, self.kv_heads, length, self.head_dim)
        keys = keys.scatter(2, slots, k)
        values = values.scatter(2, slots, v)
        shape = (1, self.kv_heads, self.repeat, keys.shape[2], self.head_dim)
        all_k = keys.unsqueeze(2).expand(shape).reshape(1, self.heads, keys.shape[2], self.head_dim)
        all_v = values.unsqueeze(2).expand(shape).reshape(1, self.heads, keys.shape[2], self.head_dim)
        scores = torch.matmul(q, all_k.transpose(-1, -2)) * (self.head_dim ** -.5) + mask
        probs = torch.softmax(scores, dim=-1)
        attention = torch.matmul(probs, all_v).transpose(1, 2).reshape(1, length, -1)
        x = x + F.linear(attention, self.o) / self.o_scale
        norm = rms(x, self.ffn_norm)
        gate = F.linear(norm, self.gate) / self.gate_scale
        up = F.linear(norm, self.up) / self.up_scale
        x = x + F.linear(F.silu(gate) * up, self.down) / self.down_scale
        return x, keys, values


class Block(nn.Module):
    def __init__(self, checkpoint, start, stop):
        super().__init__()
        self.layers = nn.ModuleList([Layer(checkpoint, i) for i in range(start, stop)])

    def forward(self, hidden, keys, values, indices, mask, cosine, sine):
        key_outputs, value_outputs = [], []
        for i, layer in enumerate(self.layers):
            hidden, key, value = layer(hidden, keys[i], values[i], indices, mask, cosine, sine)
            key_outputs.append(key)
            value_outputs.append(value)
        return hidden, torch.stack(key_outputs), torch.stack(value_outputs)


def inputs(config, count, start, capacity):
    positions = np.arange(start, start + count, dtype='int32')
    mask = np.where(np.arange(capacity)[None, :] <= positions[:, None], 0., -10000.).astype('f4')[None, None]
    angles = positions[:, None] * np.power(1e6, -np.arange(config['head_dim'] // 2) / (config['head_dim'] // 2))
    return {'indices': positions, 'mask': mask,
            'cosine': np.cos(angles).astype('f4')[None, None],
            'sine': np.sin(angles).astype('f4')[None, None]}
