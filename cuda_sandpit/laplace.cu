#include <stdio.h>
#include "common.h"
#define XSIZE 4096
#define YSIZE 4096
#define ITER 100

//#define CPU
#define GPU

#define BLOCK_SIZE 256
#define WORK_SIZE 2

#ifdef GPU
__global__
#endif
void laplace_core(float *u, float *uu)
{
#ifdef CPU
    for (int x = 1; x < XSIZE - 1; x++) {
        int base = x * YSIZE;
        for (int y = 1; y < YSIZE - 1; y++) {
            int offset = base + y;
            uu[offset] = (u[offset - YSIZE]
                          + u[offset + YSIZE]
                          + u[offset - 1]
                          + u[offset + 1]) * 0.25;
        }
    }

#elif defined(GPU)
    int tid
        = blockIdx.x * (blockDim.x * gridDim.y)
        + blockIdx.y * blockDim.x
        + threadIdx.x;

    for (int i = 0; i < WORK_SIZE; i++) {
        int p = tid * WORK_SIZE + i;
        int x = p / YSIZE + 1;
        int y = p % YSIZE + 1;

        if (x < XSIZE - 1 && y < YSIZE - 1) {
            int offset = x * YSIZE + y;
            uu[offset] = (u[offset - YSIZE]
                          + u[offset + YSIZE]
                          + u[offset - 1]
                          + u[offset + 1]) * 0.25;
        }
    }
#endif
}

float laplace(float *u, float *uu)
{
    cudaEvent_t start, stop;
    float elapsed_time;

#ifdef GPU
    float *dev_u, *dev_uu;
    size_t map_size = XSIZE * YSIZE * sizeof(int);

    HANDLE_ERROR(cudaMalloc(&dev_u,  map_size));
    HANDLE_ERROR(cudaMalloc(&dev_uu, map_size));
    HANDLE_ERROR(cudaMemcpy(dev_u, u, map_size, cudaMemcpyHostToDevice));

    int block = BLOCK_SIZE;
    dim3 grid(XSIZE, YSIZE / (block * WORK_SIZE));
#endif

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    for (int iter = 0; iter < ITER; iter += 2) {
#ifdef CPU
        laplace_core(u, uu);
        laplace_core(uu, u);
#elif defined(GPU)
        laplace_core<<<grid, block>>>(dev_u, dev_uu);
        laplace_core<<<grid, block>>>(dev_uu, dev_u);
#endif
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

#ifdef GPU
    HANDLE_ERROR(cudaMemcpy(u, dev_u, map_size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_u));
    HANDLE_ERROR(cudaFree(dev_uu));
#endif

    return elapsed_time;
}

float run_laplace()
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
