"""Read small progress records without hashing the model or changing hardware."""
from pathlib import Path
import json, time

base = Path('/tmp/qwen3-long-context-cpu-sample-only-20261006')
for prompt in [2048, 4096]:
    directory = base / f'p{prompt}-cpu'
    if not directory.exists():
        continue
    raw = directory / f'p{prompt}-1-cpu.jsonl'
    rows = []
    for line in raw.read_text().splitlines():
        if line.startswith('{'):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A live writer may not have completed the last line yet.
    runs = [row for row in rows if row.get('event') == 'run']
    with (directory / 'clock-samples.jsonl').open('rb') as stream:
        first = json.loads(stream.readline())
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2000))
        last = json.loads(stream.read().decode().splitlines()[-1])
    out = {'prompt': prompt, 'complete': (directory / 'summary.json').exists(),
           'warmups_complete': sum(row['warmup'] for row in runs),
           'measurements_complete': sum(not row['warmup'] for row in runs),
           'job_elapsed_s': round(last['time'] - first['time']),
           'log_age_s': round(time.time() - last['time'], 1),
           'cpu_mhz': [last['cpu4_khz'] / 1000, last['cpu6_khz'] / 1000],
           'temperature_c': last['temperature_millidegrees'] / 1000}
    print(json.dumps(out))
