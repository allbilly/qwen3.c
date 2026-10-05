"""Audit completed-controller restoration, fan feedback and live clock limits."""
from pathlib import Path
import argparse, hashlib, json, math

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--plan', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--partial', action='store_true', help='Audit started controllers and explicitly list jobs that never started')
args = p.parse_args()
root = args.workspace
plan = json.loads((root / args.plan).read_text())
assert plan['cpu_target_khz'] == 1800000 and plan['cooldown_target_c'] >= 0
suffix = plan.get('fan_suffix', 'screen55-cpu1800-fanheld')
stem = 'long-context-fan-' + suffix
before = root / (stem + '.json')
restored = root / (stem + '-restored.json')
assert before.read_bytes() == restored.read_bytes()
control_summary = json.loads((root / (stem + '-summary.json')).read_text())
assert control_summary['original_pwm_restored'] and control_summary['setpoint'] == 255
assert control_summary['period_ms'] == 10 and control_summary['iterations'] > 0
assert math.isfinite(control_summary['max_poll_gap_ms'])
control_log = root / (stem + '-control.jsonl')
events = [json.loads(s) for s in control_log.read_text().splitlines()]
assert len(events) == control_summary['reassertions']
assert all(r['commanded_pwm'] == 255 and 0 <= r['observed_pwm'] < 255
           and math.isfinite(r['poll_gap_ms']) and r['poll_gap_ms'] >= 0 for r in events)
assert all(a['monotonic'] < b['monotonic'] for a, b in zip(events, events[1:]))
paths = {
    'cpu4': Path('/sys/devices/system/cpu/cpufreq/policy4'),
    'cpu6': Path('/sys/devices/system/cpu/cpufreq/policy6'),
    'npu': Path('/sys/class/devfreq/fdab0000.npu'),
    'ddr': Path('/sys/class/devfreq/dmc'),
}
snapshots = []
not_started = []
for job in plan['jobs']:
    directory = root / job['directory']
    if not directory.exists():
        not_started.append(job['directory'])
        continue
    for kind in ['clock', 'gpu-clock']:
        assert (directory / (kind + '-state-before.json')).read_bytes() == (directory / (kind + '-state-restored.json')).read_bytes()
    saved = json.loads((directory / 'clock-state-restored.json').read_text())
    assert {key: {field: (paths[key] / field).read_text().strip() for field in value}
            for key, value in saved.items()} == saved
    gpu = json.loads((directory / 'gpu-clock-state-restored.json').read_text())
    assert {key: (Path('/sys/class/devfreq/fb000000.gpu') / key).read_text().strip()
            for key in gpu} == gpu
    snapshots.append({'directory': job['directory'], 'clock_and_gpu_snapshots_restored': True,
                      'live_governors_and_limits_match': True,
                      'summary_present': (directory / 'summary.json').is_file()})
assert snapshots, 'No started controllers to audit'
assert args.partial or not not_started, 'Incomplete sweep; use --partial to report its scope explicitly'
model = Path('/sys/firmware/devicetree/base/model').read_bytes().rstrip(b'\0').decode()
out = {'board_model': model, 'user_reported_no_heatsink': True,
       'fan_before': json.loads(before.read_text()), 'fan_command_restored': True,
       'fan_control_summary': control_summary,
       'fan_control_log_sha256': hashlib.sha256(control_log.read_bytes()).hexdigest(),
       'clock_restoration': snapshots,
       'partial_sweep': bool(not_started), 'not_started': not_started,
       'scope': 'PWM feedback commands and immediate restoration are audited. They do not establish physical fan presence or cooling effectiveness. The automatic kernel notifier remains active and can change PWM after restoration. All live clock governors/limits match original snapshots after the serial sweep.'}
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(f'{len(snapshots)} started controllers restored; {len(not_started)} never started; live limits and fan command restoration verified')
