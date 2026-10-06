#!/usr/bin/env python3
"""Queue benchmarks on the shared M1; also wait for uncooperative live jobs."""
import argparse
import ctypes
import errno
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


def competing_jobs():
    if sys.platform == 'darwin':
        return macos_competing_jobs()
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
            if name.startswith('python'):
                i = 1
                while i < len(argv) and argv[i]:
                    arg = os.fsdecode(argv[i])
                    if arg == '-c':
                        break  # Never interpret inline source as a file name.
                    if arg in ('-W', '-X'):
                        i += 2
                        continue
                    if arg == '-m':
                        if i + 1 < len(argv):
                            names.append(os.fsdecode(argv[i + 1]))
                        break
                    if not arg.startswith('-'):
                        names.append(Path(arg).name)
                        break
                    i += 1
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


def parse_macos_argv(data):
    """Decode KERN_PROCARGS2's argc, executable path, padding and NUL argv."""
    argc = struct.unpack_from('=i', data)[0]
    if argc < 0:
        raise ValueError('negative process argument count')
    position = data.index(b'\0', 4) + 1
    while position < len(data) and data[position] == 0:
        position += 1
    argv = []
    for _ in range(argc):
        end = data.index(b'\0', position)
        argv.append(os.fsdecode(data[position:end]))
        position = end + 1
    return argv


@lru_cache(maxsize=1)
def _macos_sysctl():
    call = ctypes.CDLL(None, use_errno=True).sysctl
    call.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    return call


def macos_process_argv(pid):
    """Read actual argument boundaries, without interpreting ps display text."""
    # Darwin's sys/sysctl.h: CTL_KERN = 1, KERN_PROCARGS2 = 49.
    mib = (ctypes.c_int * 3)(1, 49, pid)
    length = ctypes.c_size_t()
    call = _macos_sysctl()
    if call(mib, len(mib), None, ctypes.byref(length), None, 0):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    buffer = ctypes.create_string_buffer(length.value)
    if call(mib, len(mib), buffer, ctypes.byref(length), None, 0):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return parse_macos_argv(buffer.raw[:length.value])


def macos_competing_jobs():
    """Inspect native argv; ps supplies only process ancestry and executable."""
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,comm='], text=True)
    rows = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rows[int(parts[0])] = (int(parts[1]), parts[2])
        except ValueError:
            continue
    ancestors = {os.getpid()}
    pid = os.getpid()
    while pid in rows and rows[pid][0] not in ancestors:
        pid = rows[pid][0]
        ancestors.add(pid)
    jobs = []
    for pid, (_, executable) in rows.items():
        if pid in ancestors:
            continue
        try:
            argv = macos_process_argv(pid)
        except OSError as error:
            if error.errno in (errno.ESRCH, errno.EINVAL):
                continue  # Exited or no longer has an argument area.
            if error.errno not in (errno.EPERM, errno.EACCES):
                raise
            argv = [executable]  # Another user's native executable is still identifiable.
        if not argv:
            argv = [executable]
        name = Path(argv[0]).name
        names = [name]
        if name.lower().startswith('python'):
            i = 1
            while i < len(argv):
                arg = argv[i]
                if arg == '-c':
                    break
                if arg in ('-W', '-X'):
                    i += 2
                    continue
                if arg == '-m':
                    if i + 1 < len(argv):
                        module = argv[i + 1]
                        if module == 'macos.benchmark' and not any(
                                a == '--worker-mode' or a.startswith('--worker-mode=') for a in argv[i + 2:]):
                            break  # Coordinators acquire these same locks; counting waiters deadlocks.
                        names.append(module.split('.')[-1])
                    break
                if not arg.startswith('-'):
                    names.append(Path(arg).name)
                    break
                i += 1
        if 'benchmark_queue.py' in names:
            continue
        if any(n.startswith(('runq', 'bench_', 'benchmark_')) or
               n in ('benchmark', 'benchmark.py', 'paired_benchmark.py',
                     'paired_ane.py', 'llama-bench', 'run_pending.py') for n in names):
            jobs.append((pid, ' '.join(names)))
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
