"""Strict execution accounting for the new disjoint FFN, separate from old audits."""
import math

def init(rows,env):
    cpu=int(env['FFN_CPU_CHANNELS']);gpu=int(env['FFN_GPU_CHANNELS']);mode=int(env['FFN_MODE'])
    assert cpu>=0 and gpu>=0 and (cpu or gpu) and cpu%96==gpu%96==0
    npu=3072-cpu-gpu;assert npu>0 and npu%96==0 and mode in [1,2,3]
    r=[r for r in rows if r.get('event')=='ffn_split_init'];assert len(r)==1;r=r[0]
    assert (r['cpu_channels'],r['gpu_channels'],r['npu_channels'],r['mode'])==(cpu,gpu,npu,mode)
    assert r['cpu_threads']==1 and r['npu_tasks']==3 and r['fp16_operands']
    assert r['cpu_resident_weight_bytes']==28*2*cpu*1024*4
    assert r['extra_npu_weight_bytes']==28*2*npu*1024*2
    g=[r for r in rows if r.get('event')=='ffn_gpu_init'];assert len(g)==int(gpu>0)
    if g:
        assert 'Mali' in g[0]['device'] and g[0]['projection_count']==28
        assert g[0]['weight_bytes']==28*2*gpu*1024*2
        assert g[0]['event_profiling']==(env.get('FFN_PROFILE') is not None)
    return cpu,gpu,mode

def phase(rows,env,index,name,prompt,steps):
    cpu,gpu,mode=init(rows,env)
    active=bool(mode&(1 if name=='prefill' else 2))
    n=1 if name=='prefill' else steps
    native=next(r for r in rows if r.get('event')=='ffn_split_init').get('prefill_block_rows')
    multiplier=math.ceil(prompt/native) if native and name=='prefill' else 1
    calls=28*n*multiplier*int(active)
    matches=[r for r in rows if r.get('event')=='ffn_split_profile' and r['run']==index and r['phase']==name]
    assert len(matches)==1;r=matches[0]
    assert r['warmup']==(index<=0)
    assert r['calls']==r['npu_calls']==calls
    if native:
        assert native==64 and r['logical_cpu_ops']==28*n*int(active)*int(cpu>0)
        assert r['activation_pack_ms']>=0 and r['down_wall_ms']>=0
        if name=='decode' or not active:assert r['activation_pack_ms']==r['down_wall_ms']==0
    assert r['cpu_calls']==calls*int(cpu>0) and r['gpu_calls']==calls*int(gpu>0)
    assert r['rows']==(28*prompt if name=='prefill' else 28*steps)*int(active)
    assert 0<=r['gpu_completed_during_npu']<=r['gpu_calls']
    keys=['wall_ms','round_ms','dispatch_ms','npu_branch_ms','cpu_branch_ms',
          'cpu_npu_overlap_ms','join_wait_ms','merge_ms','gpu_upload_ms','gpu_kernel_ms','gpu_download_ms']
    assert all(math.isfinite(r[k]) and r[k]>=0 for k in keys)
    exclusive=sum(r[k] for k in ['round_ms','dispatch_ms','npu_branch_ms','join_wait_ms','merge_ms'])
    assert abs(exclusive-r['wall_ms'])<.00001
    assert r['cpu_npu_overlap_ms']<=min(r['cpu_branch_ms'],r['npu_branch_ms'])+.00001
    assert r['event_profiling']==(env.get('FFN_PROFILE') is not None)
    if not r['event_profiling']:assert r['gpu_upload_ms']==r['gpu_kernel_ms']==r['gpu_download_ms']==0
    if not cpu:assert r['cpu_branch_ms']==r['cpu_npu_overlap_ms']==0
    if not gpu:assert r['gpu_completed_during_npu']==r['gpu_upload_ms']==r['gpu_kernel_ms']==r['gpu_download_ms']==0
    if not active:assert all(r[k]==0 for k in keys)
    return r
