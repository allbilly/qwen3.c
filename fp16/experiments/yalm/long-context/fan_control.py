"""Snapshot, set and restore the existing board fan PWM for a serial sweep."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','restore']);p.add_argument('state');args=p.parse_args()
root=Path('/hostsys/class/hwmon');state=Path(args.state)
if args.action=='lock':
 fans=[p.parent for p in root.glob('hwmon*/name') if p.read_text().strip()=='pwmfan'];assert len(fans)==1
 pwm=fans[0]/'pwm1';before={'path':str(pwm),'pwm1':pwm.read_text().strip()};state.write_text(json.dumps(before,indent=2)+'\n')
 pwm.write_text('255');assert pwm.read_text().strip()=='255';print('Board fan PWM set to255',flush=True)
else:
 before=json.loads(state.read_text());pwm=Path(before['path']);pwm.write_text(before['pwm1']);assert pwm.read_text().strip()==before['pwm1']
 state.with_name(state.stem+'-restored.json').write_text(json.dumps({'path':str(pwm),'pwm1':pwm.read_text().strip()},indent=2)+'\n');print('Original fan PWM restored',flush=True)
