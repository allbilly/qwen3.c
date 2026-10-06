"""Reject plausible tampering of qualification, provenance and timing receipts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from macos.audit import audit
from macos.common import quality, sha256, write_json


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root/'reference').mkdir()
        logits = np.array([[3.,1.,0.], [0.,3.,1.]], 'f4')
        np.save(self.root/'reference/p.npy', logits)
        manifest = dict(config=dict(vocab=3), batch=4, context=16,
                        artifacts=[dict(package='block.mlpackage', function='width_4', width=4)])
        write_json(self.root/'manifest.json', manifest)
        identity = dict(source_sha256={'source.py':'source'}, model_manifest_sha256=sha256(self.root/'manifest.json'))
        reference = dict(mode='reference', status='PASS', **identity, prompts=[dict(
            name='p', ids=[1,2], teacher_ids=[0,1], logits_file='p.npy',
            logits_sha256=sha256(self.root/'reference/p.npy'))])
        write_json(self.root/'reference/reference.json', reference)
        jobs = [dict(mode='reference', returncode=0, output='reference/reference.json',
                     output_sha256=sha256(self.root/'reference/reference.json'))]
        measurement = dict(prefill_ms=2., prefill_tps=1000., decode_ms=4., decode_tps=250.,
                           completion_ms=6., decode_step_ms=[4.], ids=[0,1],
                           timed_predictions_match_reference=True)
        summaries = {}
        for mode, target in [('gpu','MLGPUComputeDevice'), ('ane','MLNeuralEngineComputeDevice')]:
            np.save(self.root/f'{mode}.npy', logits)
            worker = dict(mode=mode, status='PASS', **identity, compute_plans=[dict(
                package='block.mlpackage', function='width_4', preferred_operation_counts={target:1},
                operations=[dict(operator='linear', preferred=target)])], prompts=[dict(
                name='p', prompt_ids=[1,2], teacher_ids=[0,1], qualified=True,
                validation_logits=f'{mode}.npy', validation_logits_sha256=sha256(self.root/f'{mode}.npy'),
                accuracy=[quality(v,v) for v in logits], warmups=[copy.deepcopy(measurement)],
                measurement=copy.deepcopy(measurement))])
            write_json(self.root/f'{mode}.json', worker)
            jobs.append(dict(mode=mode, returncode=0, output=f'{mode}.json',
                             output_sha256=sha256(self.root/f'{mode}.json')))
            summaries[mode] = [dict(name='p',prompt_tokens=2,qualified=True,
                median={k:measurement[k] for k in ('prefill_ms','prefill_tps','decode_ms','decode_tps','completion_ms')},
                samples=[copy.deepcopy(measurement)])]
        self.report = dict(status='PASS', modes=['gpu','ane'], trials_per_mode=1,
            warmups_per_worker_prompt=1, new_tokens=2, decode_steps=1,
            physical_prefill_width=4, physical_decode_width=4, **identity,
            model_manifest=manifest, prompts=[dict(name='p',ids=[1,2])],
            numerical_gate=dict(max_nrmse=.005,max_kl=.01,top1_mismatches=0),
            jobs=jobs, summaries=summaries)
        self.save()

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        write_json(self.root/'results.json', self.report)

    def change_worker(self, change, mode='gpu'):
        job = next(j for j in self.report['jobs'] if j['mode']==mode)
        path = self.root/job['output']
        worker = json.loads(path.read_text())
        change(worker)
        write_json(path,worker)
        job['output_sha256'] = sha256(path)
        self.save()

    def rejected(self):
        with self.assertRaises(ValueError):
            audit(self.root)

    def test_valid_selected_devices_are_audited(self):
        result = audit(self.root)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['checked_logit_vectors'],4)
        self.assertEqual(result['workers'],2)

    def test_changed_logits_with_matching_argmax_fail_the_gate(self):
        values = np.load(self.root/'gpu.npy') + 1.
        np.save(self.root/'gpu.npy',values)
        original = np.load(self.root/'reference/p.npy')
        def change(worker):
            row = worker['prompts'][0]
            row['validation_logits_sha256'] = sha256(self.root/'gpu.npy')
            row['accuracy'] = [quality(r,a) for r,a in zip(original,values)]
        self.change_worker(change)
        self.rejected()

    def test_refreshed_hash_does_not_hide_reference_source_swap(self):
        self.change_worker(lambda w:w['source_sha256'].update({'source.py':'changed'}),'reference')
        self.rejected()

    def test_changed_embedded_manifest_with_old_hash_cannot_pass(self):
        self.report['model_manifest']['context'] = 4096
        self.save()
        self.rejected()

    def test_claimed_prefill_width_must_match_the_hashed_manifest(self):
        self.report['physical_prefill_width'] = 1
        self.save()
        self.rejected()

    def test_missing_summary_cannot_pass(self):
        self.report['summaries']['gpu'] = []
        self.save()
        self.rejected()

    def test_miscounted_decode_total_cannot_pass(self):
        self.change_worker(lambda w:w['prompts'][0]['measurement'].update(decode_ms=3.))
        self.rejected()

    def test_wrong_prefill_throughput_cannot_pass(self):
        self.change_worker(lambda w:w['prompts'][0]['measurement'].update(prefill_tps=2000.))
        self.rejected()

    def test_wrong_warmup_tokens_cannot_pass(self):
        self.change_worker(lambda w:w['prompts'][0]['warmups'][0].update(ids=[2,2]))
        self.rejected()

    def test_missing_accelerator_placement_cannot_pass(self):
        def change(worker):
            worker['compute_plans'][0]['operations'][0]['preferred']='MLCPUComputeDevice'
            worker['compute_plans'][0]['preferred_operation_counts']={'MLCPUComputeDevice':1}
        self.change_worker(change)
        self.rejected()

    def test_duplicate_receipt_cannot_pass(self):
        self.report['jobs'][-1]['output']=self.report['jobs'][1]['output']
        self.save()
        self.rejected()


if __name__ == '__main__':
    unittest.main()
