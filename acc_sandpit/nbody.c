//
// [REF]
// https://XXX/~gakumu/.../2009XXXXX.pdf
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#define N 10000
#define TIME_STEP 100
#define EPSILON 1e-16
#define G 6.67384e-11
#define DT 0.25

float randf()
{
    return 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}

void nbody(float *m,
           float *p_x, float *p_y, float *p_z,
           float *v_x, float *v_y, float *v_z)
{
    #pragma acc data copyin  (p_x[N], p_y[N], p_z[N], m[N])
    #pragma acc data copyout (v_x[N], v_y[N], v_z[N])
    for (int t = 0; t < TIME_STEP; t++) {
        #pragma acc parallel
        #pragma acc loop independent
        for (int i = 0; i < N; i++) {
            float x_i, y_i, z_i, x_j, y_j, z_j;
            float dx, dy, dz, r2, a;
            float acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;

            x_i = p_x[i];
            y_i = p_y[i];
            z_i = p_z[i];

            //
            // reduction (+ : var1, var2 ...)
            //  or
            // reduction (+ : var1) reduction (+ : var2) ...
            //
            // #pragma acc loop
            #pragma acc loop reduction (+ : acc_x, acc_y, acc_z)
            for (int j = 0; j < N; j++) {
                x_j = p_x[j];
                y_j = p_y[j];
                z_j = p_z[j];

                dx = x_j - x_i;
                dy = y_j - y_i;
                dz = z_j - z_i;

                r2 = (dx * dx) + (dy * dy) + (dz * dz) + EPSILON;
                a = m[j] / sqrtf(r2 * r2 * r2);

                acc_x += dx * a;
                acc_y += dy * a;
                acc_z += dz * a;
            }

            v_x[i] += acc_x * DT;
            v_y[i] += acc_y * DT;
            v_z[i] += acc_z * DT;
        }

        #pragma acc parallel
        #pragma acc loop independent
        for (int i = 0; i < N; i++) {
            p_x[i] += v_x[i] * DT;
            p_y[i] += v_y[i] * DT;
            p_z[i] += v_z[i] * DT;
        }
    }
}

long double run_nbody()
{
    float  m[N];
    float p_x[N], p_y[N], p_z[N];
    float v_x[N], v_y[N], v_z[N];
    double cpu0, cpu1;

    for (int i = 0; i < N; i++) {
        m[i]   = G * randf();
        p_x[i] = randf();
        p_y[i] = randf();
        p_z[i] = randf();
        v_x[i] = randf();
        v_y[i] = randf();
        v_z[i] = randf();
    }

    cpu0 = second();
    nbody(m, p_x, p_y, p_z, v_x, v_y, v_z);
    cpu1 = second();

    return (19L * N + 6L + 6L) * N * TIME_STEP / (cpu1 - cpu0);
}

int main()
{
    printf("[%d bodies]\n", N);
    for (int i = 0; i < 3; i++)
        printf("(%d):\t" "%.1Lf flops\n", i, run_nbody());

    return 0;
}
