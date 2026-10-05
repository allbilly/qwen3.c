// Direct M1 ANE register programming. ABI and GEMV stream from ~/ane.
#define _GNU_SOURCE
#include "ane_matmul.h"
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linear_template.h"

enum { BATCH = 32, TD_SIZE = 504, CMD_SIZE = 2304 };
// MIT-compatible UAPI: ~/ane/kmod/uapi/drm/ane_accel.h (Eileen Yoon).
struct bo_init { uint32_t handle, pad; uint64_t size, offset; };
struct bo_free { uint32_t handle, pad; };
struct submit {
    uint64_t tsk_size;
    uint32_t td_count, td_size, handles[32], btsp_handle, pad;
};
#define BO_INIT _IOWR('d', 0x41, struct bo_init)
#define BO_FREE _IOWR('d', 0x42, struct bo_free)
#define SUBMIT _IOWR('d', 0x43, struct submit)
_Static_assert(sizeof(struct bo_init) == 24, "ANE BO ABI");
_Static_assert(sizeof(struct submit) == 152, "ANE submission ABI");

typedef struct { uint32_t handle; size_t size; void *map; } Buffer;
struct AneDevice {
    int fd;
    Buffer input, output, bootstrap, scratch, bias, constants;
    unsigned long long submissions;
};
struct AnePlan {
    AneDevice *device;
    int inputs, outputs, k, n;
    Buffer command, weights;
};

static void buffer_free(AneDevice *d, Buffer *b) {
    if (b->map) munmap(b->map, b->size);
    if (b->handle) {
        struct bo_free request = {.handle = b->handle};
        ioctl(d->fd, BO_FREE, &request);
    }
    memset(b, 0, sizeof(*b));
}

static int buffer_alloc(AneDevice *d, Buffer *b, size_t bytes) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    b->size = (bytes + page - 1) & ~(page - 1);
    struct bo_init request = {.size = b->size};
    if (ioctl(d->fd, BO_INIT, &request) || !request.handle) {
        memset(b,0,sizeof(*b));
        return 0;
    }
    b->handle = request.handle;
    void *map = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     d->fd, request.offset);
    if (map == MAP_FAILED) { buffer_free(d, b); return 0; }
    b->map = map;
    memset(map,0,b->size);
    return 1;
}

static int m1_host(void) {
    char data[4096];
    int fd = open("/proc/device-tree/compatible", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    ssize_t bytes = read(fd, data, sizeof(data));
    close(fd);
    static const char compatible[] = "apple,t8103";
    return bytes > 0 && memmem(data, bytes, compatible, sizeof(compatible));
}

static int open_ane(void) {
    glob_t paths = {0};
    int fd = -1;
    const char *requested = getenv("ANE_DEVICE");
    if (glob(requested ? requested : "/dev/accel/accel*", 0, NULL, &paths))
        return -1;
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        const char *name = strrchr(paths.gl_pathv[i], '/');
        char driver[PATH_MAX], resolved[PATH_MAX];
        snprintf(driver, sizeof(driver), "/sys/class/accel/%s/device/driver", name + 1);
        if (!realpath(driver, resolved)) continue;
        name = strrchr(resolved, '/');
        if (!name || strcmp(name + 1, "ane")) continue;
        fd = open(paths.gl_pathv[i], O_RDWR | O_CLOEXEC);
        if (fd >= 0) break;
    }
    globfree(&paths);
    return fd;
}

AneDevice *ane_device_open(void) {
    if (!m1_host()) {
        fprintf(stderr, "ANE: this register stream requires base M1 (T8103) Asahi Linux.\n");
        return NULL;
    }
    AneDevice *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->fd = open_ane();
    if (d->fd < 0) {
        fprintf(stderr, "ANE: no accessible ane driver device; see ~/ane/kmod/README.md.\n");
        free(d); return NULL;
    }
    if (!buffer_alloc(d, &d->input, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->output, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->bootstrap, TD_SIZE) ||
        !buffer_alloc(d, &d->scratch, 768 * BATCH * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->bias, 768 * sizeof(__fp16)) ||
        !buffer_alloc(d, &d->constants, 16384)) {
        ane_device_close(d); return NULL;
    }
    return d;
}

void ane_device_close(AneDevice *d) {
    if (!d) return;
    buffer_free(d, &d->constants);
    buffer_free(d, &d->bias);
    buffer_free(d, &d->scratch);
    buffer_free(d, &d->bootstrap);
    buffer_free(d, &d->output);
    buffer_free(d, &d->input);
    close(d->fd);
    free(d);
}

void ane_plan_free(AnePlan *p) {
    if (!p) return;
    buffer_free(p->device, &p->weights);
    buffer_free(p->device, &p->command);
    free(p);
}

static int ensure_buffer(AneDevice *d, Buffer *b, size_t bytes) {
    if (b->size >= bytes) return 1;
    buffer_free(d, b);
    return buffer_alloc(d, b, bytes);
}

AnePlan *ane_plan_create(AneDevice *d, const QuantizedTensor *w,
                         int inputs, int outputs, int gs) {
    if (!d || !w || !w->q || !w->s || inputs <= 0 || outputs <= 0 ||
        gs <= 0 || inputs % gs || inputs > 32736 || outputs > 32736 ||
        (int64_t)inputs * outputs > INT_MAX) return NULL;
    AnePlan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->device = d; p->inputs = inputs; p->outputs = outputs;
    p->k = (inputs + 31) & ~31; p->n = (outputs + 31) & ~31;
    if (!ensure_buffer(d, &d->input, (size_t)p->k * BATCH * 2) ||
        !ensure_buffer(d, &d->scratch, (size_t)p->k * BATCH * 2) ||
        !ensure_buffer(d, &d->output, (size_t)p->n * BATCH * 2) ||
        !ensure_buffer(d, &d->bias, (size_t)p->n * 2) ||
        !buffer_alloc(d, &p->command, CMD_SIZE + 1) ||
        !buffer_alloc(d, &p->weights, (size_t)p->k * p->n * 2)) {
        ane_plan_free(p); return NULL;
    }
    uint32_t *program = p->command.map;
    memcpy(program, ane_linear_template, CMD_SIZE);
    for (size_t i=0;i<sizeof(ane_k_patches)/sizeof(ane_k_patches[0]);i++) {
        AneDimensionPatch a=ane_k_patches[i];
        program[a.offset/4] += (p->k - 768) * a.scale;
    }
    for (size_t i=0;i<sizeof(ane_n_patches)/sizeof(ane_n_patches[0]);i++) {
        AneDimensionPatch a=ane_n_patches[i];
        program[a.offset/4] += (p->n - 768) * a.scale;
    }
    // The runtime matrix occupies K*N FP16 values in the TileDMA source.
    program[0x37c/4] = program[0x384/4] = program[0x388/4] = p->k*p->n*2;
    __fp16 *packed = p->weights.map;
    // The dynamic graph transposes the input activations into the convolution
    // coefficient stream. The model matrix stays resident in [K,N] order.
    for (int k=0;k<inputs;k++) {
        for (int n=0;n<outputs;n++) {
            size_t index=(size_t)n*inputs+k;
            float value=w->q[index]*w->s[index/gs];
            if (!isfinite(value) || fabsf(value)>65504) { ane_plan_free(p); return NULL; }
            packed[(size_t)k*p->n+n]=(__fp16)value;
        }
    }
    return p;
}

int ane_plan_run_batch(AnePlan *p, const float *input, float *output, int rows) {
    if (!p || !input || !output || rows<1 || rows>BATCH) return 0;
    AneDevice *d=p->device;
    __fp16 *source=d->input.map, *result=d->output.map;
    memset(source,0,(size_t)p->k*BATCH*2);
    for (int row=0;row<rows;row++) {
        for (int k=0;k<p->inputs;k++) {
            float x=input[(size_t)row*p->inputs+k];
            if (!isfinite(x) || fabsf(x)>65504) return 0;
            source[(size_t)row*p->k+k]=(__fp16)x;
        }
    }
    // Detect a successful ioctl that did not write its advertised output.
    uint16_t *bits=d->output.map;
    for (int i=0;i<rows*p->n;i++) bits[i]=0x7e00;
    memcpy(d->bootstrap.map,p->command.map,TD_SIZE);
    uint32_t *header=d->bootstrap.map;
    header[0]=(header[0]&~(0xffu<<16))|(0x40u<<16);
    struct submit request={.tsk_size=CMD_SIZE,.td_count=4,.td_size=TD_SIZE,
                           .btsp_handle=d->bootstrap.handle};
    request.handles[0]=p->command.handle;
    request.handles[2]=d->constants.handle;
    request.handles[3]=d->scratch.handle;
    request.handles[4]=d->input.handle;
    request.handles[5]=p->weights.handle;
    request.handles[6]=d->bias.handle;
    request.handles[7]=d->output.handle;
    if (ioctl(d->fd,SUBMIT,&request)) {
        fprintf(stderr,"ANE submission failed: %s\n",strerror(errno));
        return 0;
    }
    d->submissions++;
    // All input rows are in the device buffer before writing output: alias safe.
    for (int row=0;row<rows;row++) {
        for (int n=0;n<p->outputs;n++) {
            float value=result[(size_t)row*p->n+n];
            if (!isfinite(value)) return 0;
            output[(size_t)row*p->outputs+n]=value;
        }
    }
    return 1;
}

int ane_plan_run(AnePlan *p, const float *input, float *output) {
    return ane_plan_run_batch(p,input,output,1);
}

unsigned long long ane_device_submissions(const AneDevice *d) {
    return d ? d->submissions : 0;
}

typedef struct {
    AneDevice *device;
    int layers;
    AnePlan **plans;
} ModelPlans;

void npu_matmul_shutdown(NpuMatmulContext *ctx) {
    if (!ctx) return;
    ModelPlans *m = ctx->impl;
    if (m) {
        if (m->plans)
            for (int i = 0; i < 7 * m->layers + 1; i++) ane_plan_free(m->plans[i]);
        free(m->plans);
        ane_device_close(m->device);
        free(m);
    }
    ctx->impl = NULL;
    ctx->enabled = 0;
}

int npu_matmul_init(NpuMatmulContext *ctx, const Config *c,
                    const TransformerWeights *w) {
    memset(ctx, 0, sizeof(*ctx));
    const char *env = getenv("ANE");
    if (!env) env = getenv("NPU");
    if (!env || !strcmp(env, "0")) return 0;
    ModelPlans *m = calloc(1, sizeof(*m));
    if (!m) return 0;
    ctx->impl = m; m->layers = c->n_layers;
    m->device = ane_device_open();
    if (!m->device) { npu_matmul_shutdown(ctx); return 0; }
    m->plans = calloc((size_t)7 * c->n_layers + 1, sizeof(AnePlan *));
    if (!m->plans) { npu_matmul_shutdown(ctx); return 0; }
    int heads = c->n_heads * c->head_dim, kv = c->n_kv_heads * c->head_dim;
    const QuantizedTensor *weights[] = {w->wq, w->wk, w->wv, w->wo, w->w1, w->w2, w->w3};
    int inputs[] = {c->dim, c->dim, c->dim, heads, c->dim, c->hidden_dim, c->dim};
    int outputs[] = {heads, kv, kv, c->dim, c->hidden_dim, c->dim, c->hidden_dim};
    for (int kind = 0; kind < 7; kind++) {
        for (int layer = 0; layer < c->n_layers; layer++) {
            m->plans[kind * m->layers + layer] = ane_plan_create(m->device,
                weights[kind] + layer, inputs[kind], outputs[kind], c->group_size);
            if (!m->plans[kind * m->layers + layer]) {
                fprintf(stderr, "ANE: failed to prepare projection %d, layer %d.\n", kind, layer);
                npu_matmul_shutdown(ctx); return 0;
            }
        }
    }
    ctx->enabled = 1;
    fprintf(stderr, "ANE: resident FP16 projection plans ready (%d layers); vocabulary head %s.\n",
            m->layers, "CPU Q8");
    return 1;
}

int npu_matmul_run(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                   const float *in, float *out) {
    if (!ctx || !ctx->enabled || kind < 0 || kind > NPU_MATMUL_WCLS) return 0;
    ModelPlans *m = ctx->impl;
    if (layer < 0 || layer >= m->layers) return 0;
    AnePlan *p = m->plans[kind == NPU_MATMUL_WCLS ? 7 * m->layers : kind * m->layers + layer];
    if (!p) return 0;
    if (!ane_plan_run(p, in, out)) {
        fprintf(stderr, "ANE: projection failed; refusing an invalid inference result.\n");
        exit(EXIT_FAILURE);
    }
    ctx->npu_ops++;
    return 1;
}

int npu_matmul_run_batch(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                         const float *in, float *out, int rows) {
    if (!ctx || !ctx->enabled || kind < 0 || kind >= NPU_MATMUL_WCLS) return 0;
    ModelPlans *m=ctx->impl;
    if (layer<0 || layer>=m->layers || rows<1 || rows>BATCH) return 0;
    AnePlan *p=m->plans[kind*m->layers+layer];
    if (!ane_plan_run_batch(p,in,out,rows)) {
        fprintf(stderr,"ANE: batched projection failed.\n");
        exit(EXIT_FAILURE);
    }
    ctx->npu_ops++;
    return 1;
}

void npu_matmul_reset_stats(NpuMatmulContext *ctx) {
    if (ctx) ctx->npu_ops = ctx->cpu_ops = 0;
}
