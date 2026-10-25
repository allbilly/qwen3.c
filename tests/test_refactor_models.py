import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.refactor_models import prepare_fp16


class Fp16BaselineExportTests(unittest.TestCase):
    def test_version2_preserves_half_weights_without_q8_conversion(self):
        norms = ['model.layers.0.input_layernorm.weight','model.layers.0.post_attention_layernorm.weight',
                 'model.norm.weight','model.layers.0.self_attn.q_norm.weight','model.layers.0.self_attn.k_norm.weight']
        matrices = ['model.embed_tokens.weight'] + [f'model.layers.0.{name}.weight' for name in
            ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
             'mlp.gate_proj','mlp.down_proj','mlp.up_proj']]
        entries, payload, expected_norms, expected_matrices = {}, bytearray(), bytearray(), bytearray()
        for index,name in enumerate(norms+matrices):
            values = [0.123456+index/32,-0.654321-index/32] * (1 if name in norms else 2)
            values = list(struct.unpack('<'+str(len(values))+'f',struct.pack('<'+str(len(values))+'f',*values)))
            start = len(payload)
            payload.extend(struct.pack('<'+str(len(values))+'f',*values))
            entries[name] = {'dtype':'F32','shape':[2] if name in norms else [2,2],
                             'data_offsets':[start,len(payload)]}
            halves = b''.join(struct.pack('<e',value) for value in values)
            if name in norms:
                expected_norms.extend(b''.join(struct.pack('<f',value) for value in struct.unpack('<2e',halves)))
            else:
                expected_matrices.extend(halves)
        encoded = json.dumps(entries).encode()
        header = [0x616a6331,1,2,2,1,1,1,2,8,2,1,1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            q8 = root/'q8.bin'
            q8.write_bytes(struct.pack('<12i',*header)+bytes(208))
            output = root/'fp16.bin'
            prepare_fp16(root,output,q8)
            header[1] = 2
            self.assertEqual(output.read_bytes(),struct.pack('<12i',*header)+bytes(208)+expected_norms+expected_matrices)
            previous = output.read_bytes()
            with self.assertRaises(FileExistsError):prepare_fp16(root,output,q8)
            self.assertEqual(output.read_bytes(),previous)


if __name__=='__main__':unittest.main()
