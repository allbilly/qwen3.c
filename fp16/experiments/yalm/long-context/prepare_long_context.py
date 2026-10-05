"""Extend only an isolated, frozen benchmark runner; model bytes stay unchanged."""
from pathlib import Path
import hashlib,json,shutil

root=Path(__file__).resolve().parent
parent=root/'e2e/concurrent-ffn';dest=root/'e2e/long-context'
assert not dest.exists()
meta=json.loads((parent/'source-provenance.json').read_text())
for name,digest in meta['source_sha256'].items():
    assert hashlib.sha256((parent/name).read_bytes()).hexdigest()==digest
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,p)
for name in ['cpu-linear-check.jsonl','gpu-linear-check.jsonl','timing-scope.json']:
    shutil.copy2(parent/name,dest/name)
command=[s.replace(str(parent),str(dest)) for s in meta['command']]
(dest/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
(dest/'parent-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
def edit(name,old,new):
    p=dest/name;s=p.read_text();assert s.count(old)==1,(name,old,s.count(old));p.write_text(s.replace(old,new))

edit('source/fp16/run.c','static void build_fp16(','''static int benchmark_capacity(void) {
    const char *value=getenv("BENCH_CONTEXT");
    if(!value)return 512;
    char *end;long capacity=strtol(value,&end,10);
    if(!*value||*end||capacity<512||capacity>4128){fprintf(stderr,"Invalid BENCH_CONTEXT\\n");exit(2);}
    return (int)capacity;
}

static void build_fp16(''')
edit('source/fp16/run.c','    GS = c->group_size;', '    c->seq_len=benchmark_capacity(); // Runtime KV/scratch capacity; checkpoint bytes and tensor offsets are unchanged.\n    GS = c->group_size;')
edit('source/fp16/run.c','    int ids[512], count = 0, generated[512], new_tokens = atoi(argv[3]), runs = atoi(argv[4]);',
     '    int capacity=benchmark_capacity(),count=0,new_tokens=atoi(argv[3]),runs=atoi(argv[4]);\n    int *ids=malloc((size_t)capacity*sizeof(int)),*generated=malloc((size_t)capacity*sizeof(int));\n    if(!ids||!generated)return 2;')
edit('source/fp16/run.c','while (count < 512 &&','while (count < capacity &&')
edit('source/fp16/run.c','count + new_tokens > 512','count > 4096 || new_tokens > capacity-count')
edit('source/fp16/run.c','    int teacher[512],teacher_count=0;','    int *teacher=malloc((size_t)capacity*sizeof(int)),teacher_count=0;if(!teacher)return 2;')
edit('source/fp16/run.c','while(teacher_count<512&&','while(teacher_count<capacity&&')
edit('source/fp16/run.c','    return request_status;','    free(ids);free(generated);free(teacher);\n    return request_status;')
edit('source/fp16/run.c','    malloc_run_state(&t->state, c);','    malloc_run_state(&t->state, c);\n    fprintf(stderr,"{\\"event\\":\\"runtime_context\\",\\"checkpoint_context\\":512,\\"capacity\\":%d,\\"max_prompt\\":4096}\\n",c->seq_len);')
edit('source/fp16/prefill.h','float scores[512];','float scores[p->seq_len];')

# The register emitter/submit metadata are unchanged. Extra BO space permits
# dynamic attention shapes while each actual submission obeys the old guard.
edit('source/fp16/fp16_backend.c','static Plan *prepare(const __fp16 *source, int K, int N, int offset) {',
     'static Plan *prepare_capacity(const __fp16 *source, int K, int N, int offset,int row_capacity) {')
edit('source/fp16/fp16_backend.c','    p->max_rows = K > 2048 ? 16 : (K > 1024 ? 32 : 64);','    p->max_rows = row_capacity;')
edit('source/fp16/fp16_backend.c','// Tasks keep absolute command addresses',
     'static Plan *prepare(const __fp16 *source,int K,int N,int offset) {\n    return prepare_capacity(source,K,N,offset,K>2048?16:(K>1024?32:64));\n}\n\n// Tasks keep absolute command addresses')
edit('source/fp16/fp16_backend.c','!attention_prepare()','!attention_prepare(c->seq_len)')
edit('source/fp16/fp16_backend.c','rows>512','rows>4096')
edit('source/fp16/attention_backend.h','static int attention_enabled;','static int attention_enabled,attention_capacity;')
edit('source/fp16/attention_backend.h','static int attention_prepare(void) {','static int attention_prepare(int context) {\n    attention_capacity=(context+31)&~31;')
for old,new in [('calloc(512*128,2)','calloc((size_t)attention_capacity*128,2)'),
                ('prepare(zero,128,512,0)','prepare_capacity(zero,128,attention_capacity,0,64)'),
                ('prepare(zero,512,128,0)','prepare_capacity(zero,attention_capacity,128,0,64)'),
                ('malloc(64*512*4)','malloc((size_t)64*attention_capacity*4)'),
                ('rows>512','rows>4096'),
                ('int padded_rows=(rows+31)&~31;','int padded_rows=(rows+31)&~31;\n    int tile_rows=4*NPU_CBUF_BANK_SIZE/(padded_rows*2);if(tile_rows>64)tile_rows=64;\n    if(tile_rows<1||padded_rows>attention_capacity)exit(1);'),
                ('start+=64','start+=tile_rows'),
                ('rows-start<64?rows-start:64','rows-start<tile_rows?rows-start:tile_rows'),
                ('attention_prob[i],512,m','attention_prob[i],attention_capacity,m'),
                ('row*512','row*attention_capacity'),
                ('(512-length)*4','(attention_capacity-length)*4')]:
    p=dest/'source/fp16/attention_backend.h';s=p.read_text();assert old in s,old;p.write_text(s.replace(old,new))

edit('router.c','static int layers;','static int layers,capacity;')
edit('router.c','    layers=c->n_layers;','    capacity=c->seq_len;layers=c->n_layers;')
for name in ['router.c','linear_gpu.c','concurrent_ffn.c']:
    p=dest/name;s=p.read_text()
    if name=='router.c':s=s.replace('(size_t)512*','(size_t)capacity*').replace('float prob[512]','float prob[capacity]')
    elif name=='linear_gpu.c':
        s=s.replace('static __fp16 *half_input;static int count,event_profile;', 'static __fp16 *half_input;static int count,event_profile,capacity;')
        s=s.replace('int linear_gpu_init(const Config *c,const __fp16 **matrices){','int linear_gpu_init(const Config *c,const __fp16 **matrices){\n    capacity=c->seq_len;')
        s=s.replace('(size_t)512*','(size_t)capacity*').replace('rows>512','rows>capacity')
    else:
        assert s.count('static int ff_cpu,ff_gpu,')==1
        s=s.replace('static int ff_cpu,ff_gpu,', 'static int ff_capacity;\nstatic int ff_cpu,ff_gpu,',1)
        start=s.index('int ffn_init(');body=s.index('{',start)+1;s=s[:body]+'\n    ff_capacity=c->seq_len;'+s[body:]
        s=s.replace('(size_t)512*','(size_t)ff_capacity*').replace('rows>512','rows>ff_capacity')
    p.write_text(s)
edit('source/fp16/gpu_attention.c','if(layers!=28||capacity!=512)return 0;','if(layers!=28||capacity<512||capacity>4128)return 0;')

# A self-contained build wrapper resolves all exported source paths.
shutil.copy2(Path('/home/orangepi/qwen3.c/fp16/experiments/yalm/concurrent-ffn/build.py'),dest/'build.py')
print(dest)
