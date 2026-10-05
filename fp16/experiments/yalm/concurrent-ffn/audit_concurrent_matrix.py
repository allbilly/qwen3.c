"""Recompute quality/timing and prove device placement from raw per-phase counters."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
import numpy as np
from attention_rows import checked_rows
import concurrent_audit
root=Path(__file__).resolve().parent;roofline=root.parent
parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+');parser.add_argument('--output',default='matrix-audit.json');args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
model=roofline.parent/'Qwen3-0.6B.fp16';model_sha=sha(model)
assert model_sha=='c92807935bfbb81f6628e9c2723267da367577a3f2b1bb2530b87bf5e5bba879'
assert sha(Path('/home/orangepi/qwen3.c/runq-fp16'))=='100716de3aca7c4efc7a5b0a5a87178c41adc3bc4049a5f1c009a83fe051ff58'
records=[];suites={};raw_hashes={}
for directory in args.directories:
    out=(root/directory).resolve();summary=json.loads((out/'summary.json').read_text())
    assert summary['clocks_held'] and summary['clocks_restored']
    metadata=json.loads((out/'source-provenance.json').read_text())
    assert sha(out/'runq-routes')==metadata['binary_sha256']==summary['binary_sha256']
    # Original suites contain frozen source copies; candidate suites retain the
    # exact source tree in their immutable parent build directory.
    source_base=out if (out/'source').exists() else out.parent
    for name,digest in metadata['source_sha256'].items():assert sha(source_base/name)==digest,(directory,name)
    attention_rows=checked_rows(metadata,source_base)
    for stem in ['clock','gpu-clock']:
        assert (out/f'{stem}-state-before.json').read_bytes()==(out/f'{stem}-state-restored.json').read_bytes()
    clocks=[json.loads(s) for s in (out/'clock-samples.jsonl').read_text().splitlines()]
    targets={'cpu4_khz':2256000,'cpu6_khz':2256000,'npu_hz':1000000000,'gpu_hz':1000000000,'ddr_hz':2112000000}
    assert clocks and all(r[k]==v for r in clocks for k,v in targets.items())
    checks=[]
    for config_path in sorted(out.glob('*.config.json')):
        config=json.loads(config_path.read_text());env=config['environment'];cmd=config['command'];label=config_path.name.removesuffix('.config.json')
        assert cmd[:3]==['taskset','-c','4-7'] and sha(Path(cmd[3]))==metadata['binary_sha256'] and Path(cmd[4])==model
        assert env['OMP_NUM_THREADS']=='4' and env['WARMUP_RUNS']=='2' and env['NPU_DOMAIN_ID']=='1' and env['NPU_CORES']=='3'
        assert env['COOL_REQUEST_C'] in ['45','50','51','55']
        prompt=len(Path(cmd[5]).read_text().split());steps=int(cmd[6])-1;run_count=int(cmd[7])
        assert int(cmd[6]) in [16,32]
        tokens=list(map(int,Path(env['TEACHER_IDS']).read_text().split()));assert len(tokens)==steps+1
        name={1:'hello',24:'chat24',73:'ragged73',128:'prompt128',256:'prompt256'}[prompt]
        if steps+1<=16:golden_path=roofline/'final-quality'/f'{name}-custom.txt'
        else:golden_path=roofline/'full-profile/measurements'/('p128-1-reference.jsonl' if prompt==128 else 'p256-2-reference.jsonl')
        golden_ids=next(json.loads(s)['generated_ids'] for s in golden_path.read_text().splitlines() if s.startswith('{') and json.loads(s).get('event')=='run')[:steps+1]
        assert tokens==golden_ids
        log=out/f'{label}.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
        runs=[r for r in rows if r.get('event')=='run'];assert [r['run'] for r in runs]==[-1,0,*range(1,run_count+1)]
        route=next(r for r in rows if r.get('event')=='routing')
        pre=env['DENSE_PREFILL'];decode=env['DENSE_DECODE'];head=env['CPU_CLASSIFIER'] is not None
        attention=env['GPU_ATTENTION'];gpu_pre=attention in ['both','prefill'] and prompt>1;gpu_dec=attention in ['both','decode']
        assert route['dense_prefill']==pre and route['dense_decode']==decode and route['cpu_classifier']==head
        assert route['npu_initialized']==('npu' in [pre,decode]) and route['gpu_projections_initialized']==('gpu' in [pre,decode])
        initial=[r for r in rows if r.get('event')=='linear_gpu_init']
        assert len(initial)==int('gpu' in [pre,decode])
        if initial:assert 'Mali' in initial[0]['device'] and initial[0]['weight_bytes']==1191968768 and initial[0]['projection_count']==197
        if env.get('FFN_CPU_CHANNELS') is not None:concurrent_audit.init(rows,env)
        cooling=[r for r in rows if r.get('event')=='cooldown']
        assert len(cooling)==len(runs) and all(r['after_millidegrees']<=int(env['COOL_REQUEST_C'])*1000 for r in cooling)
        profiles=[r for r in rows if r.get('event')=='phase_profile']
        assert len(profiles) in [0,2*len(runs)]
        for r in profiles:
            total=sum(v['ms'] for v in r['root'].values())
            assert .98<total/r['wall_ms']<1.001,(directory,label,'root coverage',total/r['wall_ms'])
            assert all(v['ms']>=0 for v in r['root'].values())
            for details in r['details'].values():
                assert all(v['ms']>=0 and v['bytes']>=0 for v in details.values())
                submit=details['submit']
                assert submit['invalid_driver_times']==0
                assert 0<=submit['driver_ms']<=submit['ms']+.01*submit['calls']
            assert sum(v['ms'] for d in r['details'].values() for v in d.values())<=r['wall_ms']*1.001
        for run in runs:
            index=run['run'];assert run['decode_steps']==steps and run['new_tokens']==steps+1 and run['prefill_tokens']==prompt
            assert abs(run['decode_tps']-1000*steps/run['decode_ms'])<.0006
            for phase,device in [('prefill',pre),('decode',decode)]:
                n=1 if phase=='prefill' else steps
                d=next(r for r in rows if r.get('event')=='device_phase' and r['run']==index and r['phase']==phase)
                cpu=next(r for r in rows if r.get('event')=='linear_cpu_profile' and r['run']==index and r['phase']==phase)
                expected_cpu=n*(197 if device=='cpu' else int(head))
                split=concurrent_audit.phase(rows,env,index,phase,prompt,steps) if env.get('FFN_CPU_CHANNELS') is not None else None
                assert cpu['calls']==expected_cpu
                assert d['cpu_ops']==expected_cpu+(split['cpu_calls'] if split else 0)
                expected_npu=0
                if device=='npu':
                    expected_npu=112*n+(0 if head else n)
                    if phase=='prefill' and prompt>1 and not gpu_pre:expected_npu+=28*12*math.ceil(prompt/attention_rows)
                assert d['npu_ops']==expected_npu,(label,phase,d['npu_ops'],expected_npu)
                g=[r for r in rows if r.get('event')=='linear_gpu_profile' and r['run']==index and r['phase']==phase]
                assert len(g)==int('gpu' in [pre,decode])
                if g:
                    g=g[0];expected_gpu=n*(197-int(head)) if device=='gpu' else 0
                    assert g['calls']==expected_gpu
                    expected_kernels=0
                    if device=='gpu':expected_kernels=(196+(0 if head else 2)) if phase=='prefill' and prompt>1 else 2*expected_gpu
                    assert g['kernels']==expected_kernels
                    expected_weights=n*(1191968768-(311164928 if head else 0)) if device=='gpu' else 0
                    assert g['weight_bytes']==expected_weights
                    assert g['wall_ms']>=0 and g['kernel_ms']>=0
                a=[r for r in rows if r.get('event')=='gpu_profile' and r['run']==index and r['phase']==phase]
                assert len(a)==int(attention is not None)
                if a:
                    a=a[0];enabled=gpu_pre if phase=='prefill' else gpu_dec
                    expected_calls=28*n if enabled else 0
                    assert a['calls']==expected_calls
                    fused='decode_softmax_values' in a['kernels']
                    kernel_count=4 if phase=='prefill' else (2 if fused else 3)
                    assert a['kernel_calls']==expected_calls*kernel_count
            assert run['cpu_matmul_ops']==sum(r['cpu_ops'] for r in rows if r.get('event')=='device_phase' and r['run']==index)
            assert run['npu_ops']==sum(r['npu_ops'] for r in rows if r.get('event')=='device_phase' and r['run']==index)
        request_rows=[r for r in rows if r.get('event')=='request_wall']
        if request_rows:
            assert len(request_rows)==len(runs)
            for q in request_rows:
                r=next(v for v in runs if v['run']==q['run'])
                assert q['warmup']==r['warmup'] and q['prompt_tokens']==prompt and q['output_tokens']==steps+1 and q['weights_resident']
                assert q['handoff_ms']>=0
                assert abs(q['request_ms']-q['prefill_ms']-q['decode_ms']-q['handoff_ms'])<.000004
                assert abs(q['prefill_ms']-r['first_token_ms'])<.00051 and abs(q['decode_ms']-r['decode_ms'])<.00051
                assert abs(q['output_tps']-(steps+1)*1000/q['request_ms'])<.000002
        a=np.fromfile(out/f'{label}.f32',np.float32).astype(float);b=np.fromfile(roofline/'final-quality'/f'{name}-custom.f32',np.float32).astype(float)
        assert a.shape==b.shape==(151936,) and np.isfinite(a).all()
        relative=float(np.linalg.norm(a-b)/np.linalg.norm(b));predictions=all(r['generated_ids']==golden_ids for r in runs)
        measured=[r for r in runs if not r['warmup']];ttft=statistics.median(r['first_token_ms'] for r in measured)
        observed={'directory':directory,'label':label,'route':config['route'],'prompt':prompt,'dense_prefill':pre,'dense_decode':decode,
                  'gpu_attention':attention,'cpu_classifier':head,'quality_passed':relative<.001 and predictions,
                  'relative_rmse':relative,'all_predictions_match':predictions,'ttft_ms':ttft,'effective_prefill_tps':1000*prompt/ttft,
                  'decode_tps':statistics.median(r['decode_tps'] for r in measured),'measured_runs':len(measured),
                  'ttft_range':[min(r['first_token_ms'] for r in measured),max(r['first_token_ms'] for r in measured)],
                  'decode_range':[min(r['decode_tps'] for r in measured),max(r['decode_tps'] for r in measured)],
                  'root_profiles':[{**r,'root':{k:{**v,'ms':v['ms']/(1 if r['phase']=='prefill' else steps)} for k,v in r['root'].items()}} for r in profiles if not r['warmup']],
                  'gpu_attention_profiles':[r for r in rows if r.get('event')=='gpu_profile' and not r['warmup']],
                  'gpu_projection_profiles':[r for r in rows if r.get('event')=='linear_gpu_profile' and not r['warmup']],
                  'cpu_projection_profiles':[r for r in rows if r.get('event')=='linear_cpu_profile' and not r['warmup']],
                  'ffn_split_profiles':[r for r in rows if r.get('event')=='ffn_split_profile' and not r['warmup']],
                  'cpu_projection_details':[r for r in rows if r.get('event')=='cpu_projection_details' and not r['warmup']]}
        saved=next(r for r in summary['records'] if r['label']==label)
        assert saved['quality_passed']==observed['quality_passed'] and saved['all_predictions_match']==predictions
        assert abs(saved['ttft_ms']-ttft)<1e-9 and abs(saved['decode_tps']-observed['decode_tps'])<1e-9 and saved['unchanged_limit']==.001
        if request_rows:
            measured_requests=[r for r in request_rows if not r['warmup']]
            request_ms=statistics.median(r['request_ms'] for r in measured_requests)
            observed.update(request_ms=request_ms,output_tps=(steps+1)*1000/request_ms,
              handoff_ms=statistics.median(r['handoff_ms'] for r in measured_requests),
              request_range=[min(r['request_ms'] for r in measured_requests),max(r['request_ms'] for r in measured_requests)])
            assert abs(saved['request_ms']-request_ms)<1e-9
            assert abs(saved['output_tps']-observed['output_tps'])<1e-9
        records.append(observed);checks.append({'label':label,'device_counts_checked_all_runs':True,'same_teacher_ids':True,'independent_request_span':bool(request_rows)})
        raw_hashes[str(log.relative_to(root))]=sha(log)
    suites[directory]={'clock_samples':len(clocks),'clocks_held_and_restored':True,'checked_jobs':checks}
# The last completed suite's restored settings must also match live sysfs.
out=root/args.directories[-1];state=json.loads((out/'clock-state-before.json').read_text())
for name,path in [('cpu4',Path('/sys/devices/system/cpu/cpufreq/policy4')),('cpu6',Path('/sys/devices/system/cpu/cpufreq/policy6')),('npu',Path('/sys/class/devfreq/fdab0000.npu')),('ddr',Path('/sys/class/devfreq/dmc'))]:
    assert {k:(path/k).read_text().strip() for k in state[name]}==state[name]
state=json.loads((out/'gpu-clock-state-before.json').read_text())
assert {k:(Path('/sys/class/devfreq/fb000000.gpu')/k).read_text().strip() for k in state}==state
report={'shared_fp16_model_sha256':model_sha,'selected_binary_unchanged':True,'records':records,'audits':suites,'raw_log_sha256':raw_hashes,
        'scope':'All nine dense-projection prefill/decode backend pairs plus an explicit three-device placement. CPU host non-linear operations remain in every route. Teacher-forced timing uses identical decode inputs; matching every prediction proves the same greedy sequence on the finite tested cases.'}
(root/args.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'jobs':len(records),'all_clock_and_device_audits_passed':True,'quality_failed_jobs':[(r['directory'],r['label'],r['relative_rmse']) for r in records if not r['quality_passed']]},indent=2))
