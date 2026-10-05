#include "ane/ane_matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int GS = 64;
static unsigned rng = 42;
static unsigned next_random(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(void) {
    AneDevice *device = ane_device_open();
    if (!device) return 1;
    int shapes[][2] = {{512,512}, {1024,2048}, {1024,3072}, {3072,1024}, {192,640}};
    for (size_t shape = 0; shape < sizeof(shapes)/sizeof(shapes[0]); shape++) {
        int k = shapes[shape][0], n = shapes[shape][1];
        QuantizedTensor w = {.q = malloc((size_t)k*n), .s = malloc((size_t)k*n/GS*sizeof(float))};
        float *x = calloc(k > n ? k : n, sizeof(float)), *y = calloc(n, sizeof(float));
        if (!w.q || !w.s || !x || !y) return 1;
        for (int i = 0; i < k*n; i++) w.q[i] = (int)(next_random()%255)-127;
        for (int i = 0; i < k*n/GS; i++) w.s[i] = (0.001f + (next_random()%200)*0.00001f);
        AnePlan *plan = ane_plan_create(device, &w, k, n, GS);
        if (!plan) return 1;
        for (int trial = 0; trial < 3; trial++) {
            for (int i = 0; i < k; i++) x[i] = ((int)(next_random()%2001)-1000)*0.001f;
            if (trial == 2) memset(x, 0, k*sizeof(float));
            if (!ane_plan_run(plan, x, y)) return 1;
            double err2 = 0, ref2 = 0; float worst = 0;
            for (int row = 0; row < n; row++) {
                // Independent row-major dot product with the same FP16 rounding.
                double ref = 0;
                for (int col = 0; col < k; col++) {
                    int i = row*k+col;
                    ref += (float)(__fp16)(w.q[i]*w.s[i/GS]) * (float)(__fp16)x[col];
                }
                float diff = fabsf(y[row] - ref);
                if (!isfinite(y[row]) || diff > 0.008f + fabs(ref)*0.002f) {
                    fprintf(stderr,"FAIL K=%d N=%d trial=%d row=%d expected=%g got=%g\n", k,n,trial,row,ref,y[row]);
                    return 1;
                }
                if (diff > worst) worst = diff;
                err2 += diff*diff; ref2 += ref*ref;
            }
            printf("PASS K=%d N=%d trial=%d max_error=%g relative_rmse=%g\n", k,n,trial,worst,sqrt(err2/(ref2+1e-30)));
            // In-place projection must preserve every input partition.
            if (n <= k) {
                if (!ane_plan_run(plan, x, x)) return 1;
                for (int i = 0; i < n; i++) if (fabsf(x[i]-y[i]) > 1e-5f) return 1;
            }
        }
        ane_plan_free(plan); free(w.q); free(w.s); free(x); free(y);
    }
    printf("ANE submissions: %llu\n", ane_device_submissions(device));
    ane_device_close(device);
    return 0;
}
