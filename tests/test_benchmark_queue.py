"""Exercise the real queue with isolated locks and short-lived processes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'tools/benchmark_queue.py'
spec = importlib.util.spec_from_file_location('queue_module', QUEUE)
queue_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_module)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)
        self.directory.cleanup()

    def launch(self, command, cwd=None):
        if len(command) > 1 and command[1] == str(QUEUE):
            # Use the real CLI/flock/process survey while keeping test locks
            # independent of long hardware runs in other sessions. HOME stays
            # unchanged; only this queue subprocess's Path.home is redirected.
            launcher = (
                'import pathlib, runpy, sys\n'
                'lock_root = pathlib.Path(sys.argv[1])\n'
                'pathlib.Path.home = classmethod(lambda cls: lock_root)\n'
                'queue_path = sys.argv[2]\n'
                'sys.argv = sys.argv[2:]\n'
                'runpy.run_path(queue_path, run_name="__main__")\n')
            command = [command[0], '-c', launcher, str(self.root), *command[1:]]
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def probe(self, name, duration):
        path = self.root / name
        path.write_text(
            'import json, pathlib, sys, time\n'
            'output=pathlib.Path(sys.argv[1])\n'
            'start=time.monotonic()\n'
            'output.with_suffix(".started").write_text(str(start))\n'
            f'time.sleep({duration})\n'
            'output.write_text(json.dumps([start,time.monotonic()]))\n')
        return path

    def wait_started(self, output):
        deadline = time.monotonic() + 60
        while not output.with_suffix('.started').exists():
            if time.monotonic() > deadline:
                self.fail('queued process did not start within 60 seconds')
            time.sleep(0.02)

    def test_two_queued_jobs_do_not_overlap_or_deadlock(self):
        script = self.probe('queue_probe.py', 0.5)
        outputs = [self.root / f'{i}.json' for i in range(2)]
        first = self.launch([sys.executable, str(QUEUE), '--', sys.executable, str(script), str(outputs[0])])
        self.wait_started(outputs[0])
        second = self.launch([sys.executable, str(QUEUE), '--', sys.executable, str(script), str(outputs[1])])
        for process in (first, second):
            _, stderr = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, stderr)
        intervals = [json.loads(path.read_text()) for path in outputs]
        self.assertGreaterEqual(intervals[1][0], intervals[0][1])

    def test_python_unbuffered_live_job_is_detected_and_waited_for(self):
        script = self.probe('bench_queue_probe.py', 0.5)
        external_output = self.root / 'external.json'
        external = self.launch([sys.executable, '-u', str(script), str(external_output)])
        self.wait_started(external_output)
        self.assertIn(external.pid, [pid for pid, _ in queue_module.competing_jobs()])
        output = self.root / 'queued.json'
        queued = self.launch([sys.executable, str(QUEUE), '--', sys.executable,
                              str(self.probe('queue_probe.py', 0)), str(output)])
        _, stderr = queued.communicate(timeout=60)
        external.communicate(timeout=10)
        self.assertEqual(queued.returncode, 0, stderr)
        self.assertIn('Queued behind live jobs', stderr)
        self.assertGreaterEqual(json.loads(output.read_text())[0], json.loads(external_output.read_text())[1])

    def test_child_exit_code_is_preserved(self):
        process = self.launch([sys.executable, str(QUEUE), '--', sys.executable, '-c', 'raise SystemExit(7)'])
        process.communicate(timeout=60)
        self.assertEqual(process.returncode, 7)

    def test_native_argv_preserves_empty_arguments_and_excludes_environment(self):
        argv = ['python', '-u', '/folder with spaces/bench_probe.py', "it's", '', 'a"b', '漢字']
        data = (struct.pack('=i', len(argv)) + b'/real executable\0\0\0' +
                b'\0'.join(arg.encode() for arg in argv) + b'\0PRIVATE_ENV=value\0')
        self.assertEqual(queue_module.parse_macos_argv(data), argv)

    def test_truncated_native_argv_is_rejected(self):
        data = struct.pack('=i', 2) + b'/python\0\0python\0unterminated'
        with self.assertRaises(ValueError):
            queue_module.parse_macos_argv(data)

    @unittest.skipUnless(sys.platform == 'darwin', 'native process arguments require macOS')
    def test_quotes_and_spaced_script_paths_are_detected_and_waited_for(self):
        (self.root/'folder with spaces').mkdir()
        script = self.probe('folder with spaces/bench_queue_probe.py', 1.)
        external_output = self.root/'external-quoted.json'
        extra = ["it's", 'a"b', '', '漢字']
        external = self.launch([sys.executable, '-u', str(script), str(external_output), *extra])
        self.wait_started(external_output)
        self.assertEqual(queue_module.macos_process_argv(external.pid)[2:],
                         [str(script), str(external_output), *extra])
        self.assertIn(external.pid, [pid for pid, _ in queue_module.competing_jobs()])
        output = self.root/'queued-quoted.json'
        queued = self.launch([sys.executable, str(QUEUE), '--', sys.executable,
                              str(self.probe('queue_probe.py', 0)), str(output)])
        _, stderr = queued.communicate(timeout=60)
        external.communicate(timeout=10)
        self.assertEqual(queued.returncode, 0, stderr)
        self.assertIn('Queued behind live jobs', stderr)
        self.assertGreaterEqual(json.loads(output.read_text())[0], json.loads(external_output.read_text())[1])

    @unittest.skipUnless(sys.platform == 'darwin', 'Core ML coordinators run on macOS')
    def test_macos_coordinators_are_ignored_but_workers_are_detected(self):
        (self.root/'macos').mkdir()
        (self.root/'macos/__init__.py').write_text('')
        self.probe('macos/benchmark.py', .5)
        for index, extra in enumerate(([], ['--worker-mode', 'gpu'], ['--worker-mode=ane'])):
            with self.subTest(arguments=extra):
                output = self.root/f'module-{index}.json'
                process = self.launch([sys.executable, '-u', '-m', 'macos.benchmark', str(output), *extra], cwd=self.root)
                self.wait_started(output)
                self.assertEqual(process.pid in [pid for pid, _ in queue_module.competing_jobs()], bool(extra))
                process.communicate(timeout=10)

    @unittest.skipUnless(sys.platform == 'darwin', 'Core ML benchmark modules run on macOS')
    def test_unbuffered_benchmark_module_is_detected_on_macos(self):
        self.probe('benchmark.py', 0.5)
        external_output = self.root/'external-module.json'
        external = self.launch([sys.executable, '-u', '-m', 'benchmark', str(external_output)], cwd=self.root)
        self.wait_started(external_output)
        self.assertIn(external.pid, [pid for pid, _ in queue_module.competing_jobs()])
        external.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
