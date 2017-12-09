// Ref:
//
//   [1] https://XXX/~gakumu/.../2009XXXXX.pdf
//   [2] http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html
//
#include <stdio.h>
#include "common.h"
#define N 10000
#define TIME_STEP 100
#define EPSILON 1e-16
#define G 6.67384e-11
#define DT 0.25
#define THREAD_SIZE 1 // = N / (gridDim.x * blockDim.x)
//#define THREAD_SIZE 50 // = N / (gridDim.x * blockDim.x)

__global__ void kernel1(float *m, float3 *p, float3 *v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * THREAD_SIZE;

    for (int k = 0; k < THREAD_SIZE; k++) {
        int i = offset + k;
        float x_i, y_i, z_i, x_j, y_j, z_j;
        float dx, dy, dz, r2, a;
        float acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;

        x_i = p[i].x;
        y_i = p[i].y;
        z_i = p[i].z;

        for (int j = 0; j < N; j++) {
            x_j = p[j].x;
            y_j = p[j].y;
            z_j = p[j].z;

            dx = x_j - x_i;
            dy = y_j - y_i;
            dz = z_j - z_i;

            r2 = (dx * dx) + (dy * dy) + (dz * dz) + EPSILON;
            a = m[j] * rsqrtf(r2 * r2 * r2);

            acc_x += dx * a;
            acc_y += dy * a;
            acc_z += dz * a;
        }

        v[i].x += acc_x * DT;
        v[i].y += acc_y * DT;
        v[i].z += acc_z * DT;
    }
}

__global__ void kernel2(float *m, float3 *p, float3 *v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * THREAD_SIZE;

    for (int j = 0; j < THREAD_SIZE; j++) {
        int i = offset + j;
        p[i].x += v[i].x * DT;
        p[i].y += v[i].y * DT;
        p[i].z += v[i].z * DT;
    }
}

float randf()
{
    return 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}

void rand_body(float *m, float3 *p, float3 *v)
{
    *m   = G * randf();
    p->x = randf();
    p->y = randf();
    p->z = randf();
    v->x = randf();
    v->y = randf();
    v->z = randf();
}

void nbody (int thread_num)
{
    float  m[N];
    float3 p[N], v[N];
    float  *dev_m;
    float3 *dev_p, *dev_v;

    int block_num = N / (THREAD_SIZE * thread_num);

    for (int i = 0; i < N; i++)
        rand_body(&m[i], &p[i], &v[i]);

    cudaMalloc(&dev_m, N * sizeof(float));
    cudaMalloc(&dev_p, N * sizeof(float3));
    cudaMalloc(&dev_v, N * sizeof(float3));

    cudaMemcpy(dev_m, m, N * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, v, N * sizeof(float3), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsed_time;
    long double flops;
    
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    for (int t = 0; t < TIME_STEP; t++) {
        kernel1<<<block_num, thread_num>>>(dev_m, dev_p, dev_v);
        kernel2<<<block_num, thread_num>>>(dev_m, dev_p, dev_v);
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

    // [2]
    flops = (19L * N + 6L + 6L) * N * TIME_STEP * 1000 / elapsed_time;

    printf("(%d bodies, %d blocks, %d threads)\t",
           N, block_num, thread_num);
    printf("%.1Lf flops\n", flops);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}

int main() {
    nbody(10);
    nbody(50);
    nbody(100);

    return 0;
}
