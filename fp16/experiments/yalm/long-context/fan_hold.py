"""Hold the board fan PWM setpoint while leaving kernel thermal protection active."""
from pathlib import Path
import argparse,json,time
p=argparse.ArgumentParser();p.add_argument('state');args=p.parse_args();state=Path(args.state)
fans=[p.parent for p in Path('/hostsys/class/hwmon').glob('hwmon*/name') if p.read_text().strip()=='pwmfan'];assert len(fans)==1
pwm=fans[0]/'pwm1';before={'path':str(pwm),'pwm1':pwm.read_text().strip()};assert not state.exists();state.write_text(json.dumps(before,indent=2)+'\n')
ready=state.with_suffix('.ready');stop=state.with_suffix('.stop');last=time.monotonic();iterations=0;deviations=0;max_gap=0.
try:
 with state.with_name(state.stem+'-control.jsonl').open('w') as log:
  while not stop.exists():
   now=time.monotonic();gap=now-last;max_gap=max(max_gap,gap);last=now;previous=int(pwm.read_text());iterations+=1
   if previous!=255:
    pwm.write_text('255');assert pwm.read_text().strip()=='255';deviations+=1
    log.write(json.dumps({'monotonic':now,'observed_pwm':previous,'commanded_pwm':255,'poll_gap_ms':gap*1000})+'\n');log.flush()
   if not ready.exists():ready.write_text('255\n');print('Fan full-speed setpoint held with10ms feedback polling',flush=True)
   time.sleep(.01)
finally:
 pwm.write_text(before['pwm1']);assert pwm.read_text().strip()==before['pwm1']
 state.with_name(state.stem+'-restored.json').write_text(json.dumps({'path':str(pwm),'pwm1':pwm.read_text().strip()},indent=2)+'\n')
 state.with_name(state.stem+'-summary.json').write_text(json.dumps({'iterations':iterations,'reassertions':deviations,'max_poll_gap_ms':max_gap*1000,'setpoint':255,'period_ms':10,'original_pwm_restored':True,'kernel_thermal_notifier':'left active; every observed override promptly reasserted'},indent=2)+'\n');print('Original fan PWM restored; kernel automatic fan control remains active',flush=True)
