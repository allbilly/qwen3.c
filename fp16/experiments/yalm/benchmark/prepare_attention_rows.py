"""Prepare an attention-only batch/register experiment inside four data banks."""
from pathlib import Path
import json, shutil

root=Path(__file__).resolve().parent
dest=root/'variants/attention-row128'
dest.mkdir(parents=True,exist_ok=False)
shutil.copytree(root/'source',dest/'source')
metadata=json.loads((root/'source-provenance.json').read_text())
for name in metadata['source_sha256']:
    if name.startswith('source/'):continue
    shutil.copy2(root/name,dest/name)
for name in ['cpu-linear-check.jsonl','gpu-linear-check.jsonl']:
    shutil.copy2(root/name,dest/name)
path=dest/'source/fp16/fp16_backend.c'
source=path.read_text()
old='    p->max_rows = K > 2048 ? 16 : (K > 1024 ? 32 : 64);'
assert source.count(old)==1
source=source.replace(old,'''    // Experiment: attention K<=512 can use 128 rows within four data banks.
    // Dense projection row limits and submission metadata remain unchanged.
    p->max_rows = K <= 512 ? 128 : (K > 2048 ? 16 : (K > 1024 ? 32 : 64));''')
path.write_text(source)
path=dest/'source/fp16/attention_backend.h';source=path.read_text()
for old,new in [('attention_prob[i]=malloc(64*512*4);','attention_prob[i]=malloc(128*512*4);'),
                ('for(int start=0;start<rows;start+=64) {','for(int start=0;start<rows;start+=128) {'),
                ('int m=rows-start<64?rows-start:64;','int m=rows-start<128?rows-start:128;')]:
    assert source.count(old)==1;source=source.replace(old,new)
path.write_text(source)
command=[v.replace(str(root),str(dest)) if v.startswith(str(root)) else v for v in metadata['command']]
(dest/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
source=(root/'run_matrix.py').read_text().replace('root=Path(__file__).resolve().parent;roofline=root.parent;matched=roofline.parent',
       'root=Path(__file__).resolve().parent;roofline=root.parents[2];matched=roofline.parent')
(dest/'run_matrix.py').write_text(source)
reference=(root.parent/'hybrid/attention_check.c').read_text().split('int main(void) {')[0]
checker=reference+'''
#include "source/fp16/model_helpers.h"
#ifndef NPU_CHECK_ATTN_ROWS
#define NPU_CHECK_ATTN_ROWS 128
#endif
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    setenv("DENSE_PREFILL","npu",1);setenv("DENSE_DECODE","npu",1);
    unsetenv("GPU_ATTENTION");unsetenv("CPU_CLASSIFIER");
    setenv("NPU_CORES","3",1);setenv("NPU_DOMAIN_ID","1",1);
    setenv("NPU_FUSED","1",1);setenv("NPU_ATTENTION","1",1);
    setenv("NPU_SPLIT_DOWN","1",1);setenv("NPU_STREAM_FFN","1",1);
    setenv("NPU_CLS_TILE","8192",1);
    Transformer t;const __fp16 **matrices;
    build_fp16(&t,argv[1],&matrices);
    if(!npu_init_fp16(&g_npu,&t.config,matrices)){free(matrices);free_transformer(&t);return 1;}
    free(matrices);
    float *q=malloc(512*2048*4),*k=malloc(512*1024*4),*v=malloc(512*1024*4);
    float *out=malloc(512*2048*4),*ref=malloc(512*2048*4);int status=0;
    if(!q||!k||!v||!out||!ref){status=1;goto cleanup;}
    for(int i=0;i<512*2048;i++)q[i]=((i*13%127)-63)/97.0f;
    for(int i=0;i<512*1024;i++){k[i]=((i*17%131)-65)/101.0f;v[i]=((i*19%137)-68)/103.0f;}
    int lengths[]={1,24,73,128,256};
    for(int index=0;index<5;index++) {
        int rows=lengths[index];npu_matmul_reset_stats(&g_npu);route_phase(0);
        npu_attention_prefill(&g_npu,q,k,v,out,rows);
        reference(q,k,v,ref,rows,0,1);
        double error=0,energy=0,maximum=0;
        for(int i=0;i<rows*2048;i++) {
            uint32_t bits;memcpy(&bits,out+i,4);
            if((bits&0x7f800000u)==0x7f800000u){status=2;goto cleanup;}
            double delta=out[i]-ref[i];error+=delta*delta;energy+=(double)ref[i]*ref[i];
            if(fabs(delta)>maximum)maximum=fabs(delta);
        }
        double relative=sqrt(error/energy);
        int expected=12*((rows+NPU_CHECK_ATTN_ROWS-1)/NPU_CHECK_ATTN_ROWS);
        int passed=relative<1e-4&&g_npu.npu_ops==(unsigned long long)expected;
        printf("{\\"event\\":\\"attention_check\\",\\"phase\\":\\"prefill\\",\\"context\\":%d,\\"checked\\":%d,\\"relative_rmse\\":%.12g,\\"max_absolute_error\\":%.12g,\\"limit\\":0.0001,\\"tile_rows\\":%d,\\"npu_ops\\":%llu,\\"expected_npu_ops\\":%d,\\"passed\\":%s}\\n",rows,rows*2048,relative,maximum,NPU_CHECK_ATTN_ROWS,g_npu.npu_ops,expected,passed?"true":"false");
        fflush(stdout);if(!passed){status=3;goto cleanup;}
    }
cleanup:
    npu_matmul_shutdown(&g_npu);free_transformer(&t);
    free(q);free(k);free(v);free(out);free(ref);return status;
}
'''
(dest/'npu_attention_check.c').write_text(checker)
print('Prepared attention-only row128 candidate; K<=512, <=4 data banks, original dense limits and submits')
