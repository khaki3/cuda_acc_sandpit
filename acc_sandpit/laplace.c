#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#define XSIZE 4096
#define YSIZE 4096
#define ITER 100

#define CPU
#define ACC1

void laplace_core(float *u, float *uu)
{
    #pragma acc data present(uu[0:XSIZE*YSIZE], u[0:XSIZE*YSIZE])
    #pragma acc parallel
    #pragma acc loop independent
    for (int x = 1; x < XSIZE - 1; x++) {
        #pragma acc loop independent
        for (int y = 1; y < YSIZE - 1; y++) {
            int offset = x * YSIZE + y;
            uu[offset] = (u[offset - YSIZE]
                          + u[offset + YSIZE]
                          + u[offset - 1]
                          + u[offset + 1]) * 0.25;
        }
    }
}

float laplace(float *u, float *uu)
{
    double cpu0, cpu1;

    #pragma acc data copy(uu[0:XSIZE*YSIZE]), create(u[0:XSIZE*YSIZE])
    {
        cpu0 = second();

        for (int iter = 0; iter < ITER; iter += 2) {
            laplace_core(u, uu);
            laplace_core(uu, u);
        }

        cpu1 = second();
    }

    return (cpu1 - cpu0) * 1000.;
}

double run_laplace()
{
    float *u, *uu;
    double elapsed_time;
    size_t map_size;

    map_size = XSIZE * YSIZE * sizeof(int);
    u  = (float*)malloc(map_size);
    uu = (float*)malloc(map_size);
    elapsed_time = laplace(u, uu);

    free(u);
    free(uu);
    return elapsed_time;
}

int main(int argc, char *argv[])
{
    printf("[(%d, %d) x %d]\n", XSIZE, YSIZE, ITER);
    for (int i = 0; i < 3; i++) {
        printf("(%d):\t", i);
        printf("%.1f ms\n", run_laplace());
    }
    return 0;
}
