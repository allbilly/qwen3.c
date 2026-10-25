import copy
import struct
import unittest

from tools.refactor_baseline import validate


class BaselineValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'run':i,'warmup':i==0,'ids':[1,0],'prompt_tokens':2,'new_tokens':2,
                      'prefill_ms':1.0,'decode_ms':2.0,'decode_tps':500.0,'peak_rss_bytes':1024,
                      'npu_ops':0,'cpu_ops':8} for i in range(3)]
        self.logits = struct.pack('<4f',0,1,1,0)

    def check(self, rows=None, data=None, backend='cpu'):
        return validate(self.rows if rows is None else rows,self.logits if data is None else data,
                        [1,0],2,2,2,backend)

    def test_valid_complete_baseline(self):
        self.assertEqual(self.check(),[1,0])

    def test_missing_trials_and_wrong_warmup_rejected(self):
        with self.assertRaises(ValueError):self.check(rows=self.rows[:-1])
        rows=copy.deepcopy(self.rows);rows[0]['warmup']=False
        with self.assertRaises(ValueError):self.check(rows=rows)

    def test_nondeterministic_tokens_and_missing_logits_rejected(self):
        rows=copy.deepcopy(self.rows);rows[-1]['ids']=[0,1]
        with self.assertRaises(ValueError):self.check(rows=rows)
        with self.assertRaises(ValueError):self.check(data=self.logits[:-4])

    def test_nonfinite_timing_and_logits_rejected(self):
        rows=copy.deepcopy(self.rows);rows[-1]['decode_tps']=float('nan')
        with self.assertRaises(ValueError):self.check(rows=rows)
        with self.assertRaises(ValueError):self.check(data=struct.pack('<4f',0,1,float('inf'),0))

    def test_wrong_argmax_and_unexpected_accelerator_rejected(self):
        with self.assertRaises(ValueError):self.check(data=struct.pack('<4f',1,0,0,1))
        rows=copy.deepcopy(self.rows);rows[-1]['npu_ops']=1
        with self.assertRaises(ValueError):self.check(rows=rows)


if __name__=='__main__':unittest.main()
