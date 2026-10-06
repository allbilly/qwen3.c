"""Keep benchmark evidence and release locks when native work fails."""
import argparse
from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from macos import benchmark
from macos.common import sha256, write_json


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        model = self.root/'model'
        model.mkdir()
        checkpoint = self.root/'checkpoint.bin'
        checkpoint.write_bytes(b'fixture')
        manifest = dict(protocol='qwen3-coreml-v1', config=dict(vocab=3), context=16,
                        batch=4, checkpoint_sha256=sha256(checkpoint), file_sha256={})
        write_json(model/'manifest.json', manifest)
        prompts = self.root/'prompts.json'
        write_json(prompts, [dict(name='first', ids=[1]), dict(name='second', ids=[2])])
        self.args = argparse.Namespace(model=model, checkpoint=checkpoint, prompts=prompts,
                                       output=self.root/'result', modes=['gpu','ane'], trials=1,
                                       new_tokens=2, warmups=2, max_nrmse=.005, max_kl=.01)
        self.enterContext(patch('macos.benchmark.host_snapshot', return_value={'fixture': True}))
        self.enterContext(redirect_stdout(io.StringIO()))


class WorkerFailures(Fixture):
    def setUp(self):
        super().setUp()
        self.args.output = self.root/'worker.json'
        self.args.worker_mode = 'gpu'
        self.args.reference = self.root/'reference'
        self.args.reference.mkdir()
        self.logits = np.array([[3.,1.,0.], [0.,3.,1.]], 'f4')
        rows = []
        for prompt in json.loads(self.args.prompts.read_text()):
            name = prompt['name']+'.npy'
            np.save(self.args.reference/name, self.logits)
            rows.append(dict(**prompt, teacher_ids=[0,1], logits_file=name,
                             logits_sha256=sha256(self.args.reference/name)))
        write_json(self.args.reference/'reference.json', dict(prompts=rows))
        self.enterContext(patch('macos.benchmark.importlib.metadata.version', return_value='fixture'))
        self.checkpoints = []

    def engine(self, fail_call=None, drift=0., init_error=False, mismatch_call=None):
        case = self

        class FakeEngine:
            def __init__(self, *_):
                self.plans = [{'fixture': True}]
                if init_error:
                    raise RuntimeError('initialization failed')
                self.config = {'vocab': 3}
                self.manifest = {'context': 16}
                self.calls = 0

            def request(self, prompt, *_args, **_kwargs):
                case.checkpoints.append(json.loads(case.args.output.read_text()))
                self.calls += 1
                if self.calls == fail_call:
                    raise RuntimeError('prediction failed on second prompt')
                measurement = dict(prefill_ms=2., prefill_tps=len(prompt)*500.,
                                   decode_ms=4., decode_tps=250., completion_ms=6.,
                                   decode_step_ms=[4.], ids=[0,1])
                if self.calls == mismatch_call:
                    measurement['ids'] = [2,2]
                return measurement, case.logits.copy() + np.float32(drift)

        return FakeEngine

    def receipt(self):
        return json.loads(self.args.output.read_text())

    def test_prediction_failure_retains_completed_prompt_and_logits(self):
        with patch('macos.benchmark.Engine', self.engine(fail_call=5)):
            with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                benchmark.worker(self.args)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertIn('prediction failed', receipt['error'])
        self.assertIn('finished', receipt)
        self.assertEqual([p['name'] for p in receipt['prompts']], ['first'])
        self.assertEqual(receipt['prompts'], self.checkpoints[-1]['prompts'])
        row = receipt['prompts'][0]
        self.assertTrue(row['qualified'])
        self.assertEqual(len(row['warmups']), 2)
        self.assertEqual(row['measurement']['decode_step_ms'], [4.])
        self.assertEqual(sha256(self.root/row['validation_logits']), row['validation_logits_sha256'])
        np.testing.assert_array_equal(np.load(self.root/row['validation_logits']), self.logits)

    def test_reference_failure_retains_completed_prompt(self):
        self.args.worker_mode = 'reference'
        with patch('macos.benchmark.Engine', self.engine(fail_call=2)):
            with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                benchmark.worker(self.args)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertEqual([p['name'] for p in receipt['prompts']], ['first'])
        row = receipt['prompts'][0]
        self.assertEqual(sha256(self.root/row['logits_file']), row['logits_sha256'])

    def test_initialization_failure_records_partial_plans_and_error(self):
        with patch('macos.benchmark.Engine', self.engine(init_error=True)):
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                benchmark.worker(self.args)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertEqual(receipt['prompts'], [])
        self.assertEqual(receipt['compute_plans'], [{'fixture': True}])
        self.assertIn('finished', receipt)

    def test_worker_is_running_until_every_prompt_completes(self):
        with patch('macos.benchmark.Engine', self.engine()):
            benchmark.worker(self.args)
        self.assertTrue(all(r['status'] == 'RUNNING' for r in self.checkpoints))
        self.assertTrue(any(len(r['prompts']) == 1 for r in self.checkpoints))
        self.assertEqual(self.receipt()['status'], 'PASS')
        self.assertEqual(len(self.receipt()['prompts']), 2)

    def test_numeric_failure_remains_unqualified_at_completion(self):
        with patch('macos.benchmark.Engine', self.engine(drift=1.)):
            benchmark.worker(self.args)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'FAIL')
        self.assertEqual(len(receipt['prompts']), 2)
        self.assertTrue(all(not row['qualified'] for row in receipt['prompts']))

    def test_either_warmup_mismatch_fails_gpu_and_ane_qualification(self):
        for mode in ('gpu', 'ane'):
            self.args.worker_mode = mode
            for warmup_call in (2, 3):
                with self.subTest(mode=mode, warmup_call=warmup_call):
                    with patch('macos.benchmark.Engine', self.engine(mismatch_call=warmup_call)):
                        benchmark.worker(self.args)
                    receipt = self.receipt()
                    self.assertEqual(receipt['status'], 'FAIL')
                    self.assertEqual(len(receipt['prompts']), 2)
                    first, second = receipt['prompts']
                    self.assertFalse(first['qualified'])
                    self.assertTrue(second['qualified'])
                    self.assertTrue(all(check['top1_mismatches'] == 0 for check in first['accuracy']))
                    self.assertTrue(first['measurement']['timed_predictions_match_reference'])
                    self.assertEqual(first['warmups'][warmup_call-2]['ids'], [2,2])
                    self.assertEqual(first['warmups'][3-warmup_call]['ids'], [0,1])
                    self.assertIn('finished', receipt)


class QueueFailures(Fixture):
    def setUp(self):
        super().setUp()
        self.enterContext(patch('macos.benchmark.sys.platform', 'darwin'))
        self.enterContext(patch('macos.benchmark.Path.home', return_value=self.root))
        self.enterContext(patch('macos.benchmark.subprocess.run', side_effect=AssertionError('unexpected worker launch')))
        self.opened_locks = []

    def intercept_locks(self, denied=None):
        original_open = Path.open

        def open_checked(path, *args, **kwargs):
            is_lock = path.parent == self.root and path.name in ('ane.lock', 'gpu.lock')
            if is_lock and path.name == denied:
                raise PermissionError('lock open denied')
            stream = original_open(path, *args, **kwargs)
            if is_lock:
                self.opened_locks.append(stream)
            return stream

        return patch('macos.benchmark.Path.open', new=open_checked)

    def assert_error(self, message):
        report = json.loads((self.args.output/'results.json').read_text())
        self.assertEqual(report['status'], 'ERROR')
        self.assertIn(message, report['error'])
        self.assertIn('finished', report)
        self.assertEqual(report['jobs'], [])

    def assert_locks_released(self):
        self.assertTrue(all(stream.closed for stream in self.opened_locks))
        for name in ('ane.lock', 'gpu.lock'):
            path = self.root/name
            if path.exists():
                with path.open('rb') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def test_first_lock_open_failure_finalizes_error(self):
        with self.intercept_locks(denied='ane.lock'):
            with self.assertRaisesRegex(PermissionError, 'lock open denied'):
                benchmark.run(self.args)
        self.assert_error('lock open denied')
        self.assert_locks_released()

    def test_second_lock_open_failure_closes_first_lock(self):
        with self.intercept_locks(denied='gpu.lock'):
            with self.assertRaisesRegex(PermissionError, 'lock open denied'):
                benchmark.run(self.args)
        self.assertEqual(len(self.opened_locks), 1)
        self.assert_error('lock open denied')
        self.assert_locks_released()

    def test_flock_failure_closes_both_open_files(self):
        original_flock = fcntl.flock

        def acquire(stream, operation):
            if Path(stream.name).name == 'gpu.lock':
                raise PermissionError('flock denied')
            return original_flock(stream, operation)

        with self.intercept_locks(), patch('macos.benchmark.fcntl.flock', side_effect=acquire):
            with self.assertRaisesRegex(PermissionError, 'flock denied'):
                benchmark.run(self.args)
        self.assertEqual(len(self.opened_locks), 2)
        self.assert_error('flock denied')
        self.assert_locks_released()

    def test_process_survey_failure_finalizes_error_and_releases_locks(self):
        with self.intercept_locks(), patch('tools.benchmark_queue.competing_jobs', side_effect=PermissionError('ps denied')):
            with self.assertRaisesRegex(PermissionError, 'ps denied'):
                benchmark.run(self.args)
        self.assert_error('ps denied')
        self.assert_locks_released()

    def test_receipt_write_failure_still_releases_locks(self):
        def persist(path, report):
            if report['status'] == 'ERROR':
                raise OSError('receipt write failed')
            write_json(path, report)

        with self.intercept_locks(), patch('tools.benchmark_queue.competing_jobs', side_effect=PermissionError('ps denied')), \
             patch('macos.benchmark.write_json', side_effect=persist):
            with self.assertRaisesRegex(OSError, 'receipt write failed'):
                benchmark.run(self.args)
        self.assert_locks_released()

    def test_successful_coordinator_still_collects_all_workers(self):
        launched = []

        def launch(command, **_kwargs):
            mode = command[command.index('--worker-mode')+1]
            output = Path(command[command.index('--output')+1])
            launched.append(mode)
            identity = dict(source_sha256=benchmark.source_hashes(),
                            model_manifest_sha256=sha256(self.args.model/'manifest.json'))
            prompts = [dict(qualified=True, measurement=dict(prefill_ms=2., prefill_tps=500.,
                           decode_ms=4., decode_tps=250., completion_ms=6.)) for _ in range(2)]
            write_json(output, dict(mode=mode, status='PASS', **identity, prompts=prompts))
            return SimpleNamespace(returncode=0)

        with self.intercept_locks(), patch('tools.benchmark_queue.competing_jobs', return_value=[]), \
             patch('macos.benchmark.subprocess.run', side_effect=launch):
            self.assertTrue(benchmark.run(self.args))
        report = json.loads((self.args.output/'results.json').read_text())
        self.assertEqual(launched, ['reference','gpu','ane'])
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(len(report['jobs']), 3)
        self.assertTrue(all(row['qualified'] for rows in report['summaries'].values() for row in rows))
        self.assert_locks_released()

    @unittest.skipUnless(sys.platform == 'darwin', 'native queue coordination requires macOS')
    def test_two_coordinators_waiting_on_shared_locks_complete_without_overlap(self):
        # Invoke the real coordinator and native scanner in each child. Only
        # model workers are replaced; these checks never exercise hardware.
        workspace = Path(benchmark.__file__).resolve().parents[1]
        package = self.root/'macos'
        package.mkdir()
        (package/'__init__.py').write_text('__path__.append('+repr(str(workspace/'macos'))+')\n')
        (package/'benchmark.py').write_text('''import argparse, importlib.util, json, os, pathlib, sys, time
from unittest.mock import patch
workspace=pathlib.Path(sys.argv[1]); directory=pathlib.Path(sys.argv[2]); role=sys.argv[3]
sys.path.append(str(workspace))
spec=importlib.util.spec_from_file_location('macos.test_actual',workspace/'macos/benchmark.py')
actual=importlib.util.module_from_spec(spec); spec.loader.exec_module(actual)
from tools import benchmark_queue
survey=benchmark_queue.competing_jobs
original_run=actual.subprocess.run
args=argparse.Namespace(model=directory/'model',checkpoint=directory/'checkpoint.bin',prompts=directory/'prompts.json',output=directory/role,modes=['gpu'],trials=1,new_tokens=2,warmups=1,max_nrmse=.005,max_kl=.01)
def mark(name):
    temporary=directory/(name+'.tmp')
    temporary.write_text(str(os.getpid()))
    temporary.replace(directory/name)
if role=='B': mark('B-ready')
def inspect():
    if role=='A':
        mark('A-locked')
        deadline=time.monotonic()+10
        while not (directory/'B-ready').exists():
            if time.monotonic()>deadline: raise RuntimeError('second coordinator did not start')
            time.sleep(.01)
    peer=int((directory/('B-ready' if role=='A' else 'A-locked')).read_text())
    # Other sessions do not participate in this private-lock regression.
    return [job for job in survey() if job[0]==peer]
interval=[]
def guarded_run(command,*positional,**kwargs):
    if '--worker-mode' not in command: return original_run(command,*positional,**kwargs)
    if not interval: interval.append(time.monotonic())
    time.sleep(.05)
    output=pathlib.Path(command[command.index('--output')+1])
    mode=command[command.index('--worker-mode')+1]
    prompts=json.loads(args.prompts.read_text())
    rows=[dict(qualified=True,measurement=dict(prefill_ms=2.,prefill_tps=len(p['ids'])*500.,decode_ms=4.,decode_tps=250.,completion_ms=6.)) for p in prompts]
    actual.write_json(output,dict(mode=mode,status='PASS',source_sha256=actual.source_hashes(),model_manifest_sha256=actual.sha256(args.model/'manifest.json'),prompts=rows))
    (directory/(role+'-interval.json')).write_text(json.dumps([interval[0],time.monotonic()]))
    return actual.subprocess.CompletedProcess(command,0)
with patch('pathlib.Path.home',return_value=directory),patch.object(actual,'host_snapshot',return_value={'fixture':True}),patch.object(benchmark_queue,'competing_jobs',side_effect=inspect),patch.object(actual.subprocess,'run',side_effect=guarded_run):
    assert actual.run(args)
''')
        processes = []
        try:
            first = subprocess.Popen([sys.executable, '-u', '-m', 'macos.benchmark', str(workspace), str(self.root), 'A'],
                                     cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(first)
            deadline = time.monotonic()+10
            while not (self.root/'A-locked').exists():
                self.assertIsNone(first.poll(), 'first coordinator exited before acquiring locks')
                self.assertLess(time.monotonic(), deadline, 'first coordinator did not acquire locks')
                time.sleep(.01)
            second = subprocess.Popen([sys.executable, '-u', '-m', 'macos.benchmark', str(workspace), str(self.root), 'B'],
                                      cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(second)
            for process in processes:
                _, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr)
            intervals = [json.loads((self.root/(role+'-interval.json')).read_text()) for role in ('A','B')]
            self.assertGreaterEqual(intervals[1][0], intervals[0][1])
            for role in ('A','B'):
                self.assertEqual(json.loads((self.root/role/'results.json').read_text())['status'], 'PASS')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
