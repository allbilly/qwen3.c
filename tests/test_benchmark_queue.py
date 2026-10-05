"""Exercise the real shared queue with harmless, short-lived processes."""
import importlib.util
import json
from pathlib import Path
import subprocess
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

    def launch(self, command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


if __name__ == '__main__':
    unittest.main()
