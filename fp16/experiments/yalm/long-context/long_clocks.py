"""Temporarily fix the benchmark clocks; restore the saved values after measurements."""
from pathlib import Path
import json
import sys
import time

base = Path('/hostsys')
root = Path(__file__).resolve().parents[2]
state_path = root / 'clock-state.json'
groups = {
    'cpu4': (base / 'devices/system/cpu/cpufreq/policy4', 'scaling_governor', 'scaling_min_freq', 'scaling_max_freq', '1800000'),
    'cpu6': (base / 'devices/system/cpu/cpufreq/policy6', 'scaling_governor', 'scaling_min_freq', 'scaling_max_freq', '1800000'),
    'npu': (base / 'class/devfreq/fdab0000.npu', 'governor', 'min_freq', 'max_freq', '1000000000'),
    'ddr': (base / 'class/devfreq/dmc', 'governor', 'min_freq', 'max_freq', '2112000000'),
}

def restore(state):
    for name, (path, governor, minimum, maximum, _) in groups.items():
        saved = state[name]
        (path / minimum).write_text(saved[minimum])
        (path / maximum).write_text(saved[maximum])
        (path / governor).write_text(saved[governor])

mode = sys.argv[1]
if mode == 'lock':
    if state_path.exists():
        raise RuntimeError('Clock backup already exists; restore it before locking again')
    state = {name: {key: (path / key).read_text().strip() for key in [governor, minimum, maximum]}
             for name, (path, governor, minimum, maximum, _) in groups.items()}
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    try:
        for name, (path, governor, minimum, maximum, target) in groups.items():
            (path / maximum).write_text(target)
            (path / minimum).write_text(target)
            (path / governor).write_text('performance')
        time.sleep(0.3)
        print('Benchmark clocks fixed; saved previous settings to', state_path)
    except Exception:
        restore(state)
        state_path.unlink()
        raise
elif mode == 'restore':
    restore(json.loads(state_path.read_text()))
    (root / 'results/clock-state-restored.json').write_text(state_path.read_text())
    state_path.unlink()
    print('Previous clock settings restored')
else:
    raise ValueError(mode)
