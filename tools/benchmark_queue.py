#!/usr/bin/env python3
"""Queue benchmarks on the shared M1; also wait for uncooperative live jobs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def competing_jobs():
    jobs = []
    own = {os.getpid(), os.getppid()}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            if not argv or not argv[0]:
                continue
            name = Path(os.fsdecode(argv[0])).name
            # Inspect executables and arguments, never shell command bodies.
            names = [name]
            if name.startswith('python') and len(argv) > 1:
                names.append(Path(os.fsdecode(argv[1])).name)
            if 'benchmark_queue.py' in names:
                continue
            if any(n.startswith(('runq', 'bench_', 'benchmark_')) or
                   n in ('paired_benchmark.py', 'llama-bench') for n in names):
                jobs.append((int(entry.name), ' '.join(names)))
                continue
            for fd in (entry / 'fd').iterdir():
                try:
                    if os.readlink(fd).startswith('/dev/accel/'):
                        jobs.append((int(entry.name), name))
                        break
                except OSError:
                    pass
        except (OSError, PermissionError):
            continue
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('provide a benchmark command after --')
    # Keep the same lock files/inodes for other Codex sessions to share.
    locks = []
    for name in ('ane.lock', 'gpu.lock'):
        lock = (Path.home() / name).open('a')
        print(f'Waiting for {lock.name}', file=sys.stderr, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        locks.append(lock)
    while jobs := competing_jobs():
        print(f'Queued behind live jobs: {jobs}', file=sys.stderr, flush=True)
        time.sleep(2)
    print('Benchmark locks acquired; no competing benchmark or ANE process.',
          file=sys.stderr, flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
