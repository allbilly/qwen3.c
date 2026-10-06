"""Numerical and checkpoint contract checks independent of Core ML hardware."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from macos.common import Checkpoint, load_prompts, quality


def tiny_checkpoint(path):
    header = (0x616a6331, 1, 32, 64, 2, 4, 2, 128, 64, 16, 1, 8)
    rng = np.random.default_rng(123)
    matrices = {}
    with Path(path).open('wb') as stream:
        stream.write(struct.pack('<12i', *header) + bytes(208))
        for count in (64, 64, 32, 32, 32):
            stream.write(np.ones(count, '<f4').tobytes())
        shapes = [('embedding', (128, 32))]
        for name, shape in [('q', (64,32)), ('k', (32,32)), ('v', (32,32)),
                            ('o', (32,64)), ('gate', (64,32)), ('down', (32,64)),
                            ('up', (64,32))]:
            shapes += [(f'{layer}.{name}', shape) for layer in range(2)]
        for name, shape in shapes:
            count = int(np.prod(shape))
            values = rng.integers(-127, 128, count, dtype='i1')
            scales = rng.uniform(.0001, .0005, count//8).astype('<f4')
            stream.write(values.tobytes())
            stream.write(scales.tobytes())
            matrices[name] = (values.reshape(-1,8).astype('f4') * scales[:,None]).reshape(shape).astype('f2')
    return matrices


class Contracts(unittest.TestCase):
    def test_layout_recovers_every_projection_and_tied_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tiny.bin'
            expected = tiny_checkpoint(path)
            checkpoint = Checkpoint(path)
            for name, matrix in expected.items():
                np.testing.assert_array_equal(checkpoint.matrix(name), matrix)
            np.testing.assert_array_equal(checkpoint.matrix('head'), expected['embedding'])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                Checkpoint(path)

    def test_quality_gate_sees_logit_drift_even_if_argmax_matches(self):
        ref = np.array([[3., 1., -2.]])
        exact = quality(ref, ref)
        self.assertEqual(exact['nrmse'], 0)
        shifted = quality(ref, ref + 1.)
        self.assertEqual(shifted['top1_mismatches'], 0)
        self.assertGreater(shifted['nrmse'], .005)
        with self.assertRaises(ValueError):
            quality(ref, np.array([[np.nan, 1., 0.]]))
        with self.assertRaises(ValueError):
            quality(ref, ref[:, :2])

    def test_prompt_context_counts_only_subsequent_decode_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prompts.json'
            path.write_text('[{"name":"boundary","ids":[1,2,3]}]')
            self.assertEqual(len(load_prompts(path, 4, 5, 3)), 1)
            with self.assertRaises(ValueError):
                load_prompts(path, 4, 4, 3)
            with self.assertRaises(ValueError):
                load_prompts(path, 3, 5, 3)

    def test_prefill_padding_and_cache_overwrite_equal_single_token_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is required for decoder checks')
        from macos.model import Block, inputs
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tiny.bin'
            tiny_checkpoint(path)
            checkpoint = Checkpoint(path)
            block = Block(checkpoint, 0, 2).eval()
            for index, layer in enumerate(block.layers):
                for name in ('q','k','v','o','gate','down','up'):
                    restored = getattr(layer,name) / getattr(layer,name+'_scale')[:,None]
                    np.testing.assert_array_equal(restored.numpy(),checkpoint.matrix(f'{index}.{name}').astype('f4'))
            config = checkpoint.config
            embedding = checkpoint.matrix('embedding').astype('f4')
            capacity = 12
            shape = (2, 1, 2, capacity, 16)
            keys, values = np.zeros(shape,'f4'), np.zeros(shape,'f4')

            def step(tokens, start, width, keys, values):
                hidden = np.zeros((1,width,32),'f4')
                hidden[0,:len(tokens)] = embedding[tokens]
                kwargs = dict(hidden=hidden, keys=keys, values=values,
                              **inputs(config,width,start,capacity))
                with torch.no_grad():
                    result = block(*(torch.from_numpy(v) for v in kwargs.values()))
                return tuple(v.numpy() for v in result)

            batched, bk, bv = step([1,2,3],0,4,keys.copy(),values.copy())
            for position, token in enumerate([1,2,3]):
                serial, keys, values = step([token],position,1,keys,values)
                np.testing.assert_allclose(batched[0,position],serial[0,0],rtol=1e-4,atol=1e-5)
            next_batch, _, _ = step([4],3,1,bk,bv)
            next_serial, _, _ = step([4],3,1,keys,values)
            np.testing.assert_allclose(next_batch,next_serial,rtol=1e-4,atol=1e-5)


if __name__ == '__main__':
    unittest.main()
