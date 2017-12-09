// super slow program
#include <stdio.h>
#include "common.h"
#include "matrix.h"
//#define DEBUG

// A^T x B (TN)
__global__ void stage1 (float *result_stage1, Matrix a, Matrix b)
{
    __shared__ float a_sub[MATRIX_UNIT][MATRIX_UNIT];
    __shared__ float b_sub[MATRIX_UNIT][MATRIX_UNIT];

    int c_x = MATRIX_UNIT * blockIdx.x;
    int c_y = MATRIX_UNIT * blockIdx.y;
    int c_z = MATRIX_UNIT * blockIdx.z;

    // use A^T
    int a_x = c_y + THREAD_COLS * threadIdx.x;
    int a_y = c_z + threadIdx.y;

    int b_x = c_x + THREAD_COLS * threadIdx.x;
    int b_y = c_z + threadIdx.y;

    int a_offset = a_y * a.x + a_x;
    int b_offset = b_y * b.x + b_x;
    int tmp_x_offset = THREAD_COLS * threadIdx.x;

    for (int i = 0; i < THREAD_COLS; i++)
        a_sub[threadIdx.y][tmp_x_offset + i] = a.body[a_offset + i];
    for (int i = 0; i < THREAD_COLS; i++)
        b_sub[threadIdx.y][tmp_x_offset + i] = b.body[b_offset + i];

    __syncthreads();

    int r_offset
        = (MATRIX_UNIT * MATRIX_UNIT)
        * (  blockIdx.y * gridDim.x * gridDim.z
           + blockIdx.x * gridDim.z
           + blockIdx.z)
        + threadIdx.y * MATRIX_UNIT
        + tmp_x_offset;

    for (int i = 0; i < THREAD_COLS; i++) {
        float tmp = 0;

        int s_y = threadIdx.y;
        int s_x = tmp_x_offset + i;

        for (int j = 0; j < MATRIX_UNIT; j++)
            tmp += a_sub[j][s_y] * b_sub[j][s_x];

        //r_sub[s_y][s_x] = tmp;
        result_stage1[r_offset + i] = tmp;
    }
}

// collect RS1
__global__ void stage2 (float *result_stage2, float *result_stage1, Matrix b)
{
    float c[THREAD_COLS];

    int r_step = (MATRIX_UNIT * MATRIX_UNIT);
    int z = b.y / MATRIX_UNIT;

    int r1_base
        = r_step
        * (  blockIdx.y * gridDim.x * z
           + blockIdx.x * z)
        + threadIdx.y * MATRIX_UNIT
        + THREAD_COLS * threadIdx.x;

    for (int i = 0; i < THREAD_COLS; i++)
        c[i] = 0;

    for (int i = 0; i < z; i++) {
        int r1_offset = r1_base + r_step * i;
        for (int j = 0; j < THREAD_COLS; j++)
            c[j] += result_stage1[r1_offset + j];
    }

    int r2_y = MATRIX_UNIT * blockIdx.y + threadIdx.y;
    int r2_x = MATRIX_UNIT * blockIdx.x + THREAD_COLS * threadIdx.x;
    int r2_offset = r2_y * b.x + r2_x;

    for (int i = 0; i < THREAD_COLS; i++)
        result_stage2[r2_offset + i] = c[i];
}

// C = alpha * RS2 + beta * C
__global__ void stage3 (Matrix c, float *result_stage2, float alpha, float beta)
{
    int y = MATRIX_UNIT * blockIdx.y + threadIdx.y;
    int x = MATRIX_UNIT * blockIdx.x + THREAD_COLS * threadIdx.x;
    int offset = y * c.x + x;

    for (int i = 0; i < THREAD_COLS; i++) {
        int pos = offset + i;
        float tmp = c.body[pos];

        c.body[pos] = alpha * result_stage2[pos] + beta * tmp;
    }
}

float run_sgemm (int type, Matrix a, Matrix b, Matrix c,
                  float alpha, float beta)
{
    Matrix dev_a = matrix_into_device(a, !(type == TN || type == TT));
    Matrix dev_b = matrix_into_device(b, type == NT || type == TT);
    Matrix dev_c = matrix_into_device(c, false);

    // A^T x B
    dim3 blocks1(dev_b.x / MATRIX_UNIT,
                 dev_a.x / MATRIX_UNIT,
                 dev_a.y / MATRIX_UNIT);
    dim3 blocks2(dev_b.x / MATRIX_UNIT,
                 dev_a.x / MATRIX_UNIT);
    dim3 threads(MATRIX_UNIT / THREAD_COLS,
                 MATRIX_UNIT);

    float *result_stage1, *result_stage1_map, *result_stage2;
    size_t c_size = dev_b.x * dev_a.x * sizeof(float);

    HANDLE_ERROR(cudaHostAlloc((void**)&result_stage1,
                               c_size * dev_a.y * sizeof(float),
                               cudaHostAllocMapped));
    cudaHostGetDevicePointer(&result_stage1_map, result_stage1, 0);

    HANDLE_ERROR(cudaMalloc((void**)&result_stage2, c_size));

    cudaEvent_t start, stop;
    float elapsed_time;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    stage1<<<blocks1, threads>>>(result_stage1_map, dev_a, dev_b);

    stage2<<<blocks2, threads>>>(result_stage2, result_stage1, dev_b);

    stage3<<<blocks2, threads>>>(dev_c, result_stage2, alpha, beta);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

    HANDLE_ERROR(cudaFreeHost(result_stage1));
    HANDLE_ERROR(cudaFree(result_stage2));
    matrix_free_device(dev_a);
    matrix_free_device(dev_b);
    matrix_free_device(dev_c);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsed_time;
}

void test_sgemm (int size)
{
    Matrix a = random_matrix(size, size);
    Matrix b = random_matrix(size, size);
    Matrix c = random_matrix(size, size);
    
    float elapsed_time;
    long double flops;

    elapsed_time = run_sgemm(TN, a, b, c, 1.F, 1.F);
    flops = 2L * size * size * size * 1000 / elapsed_time;

    printf("(%d x %d)\t" "%.1Lf flops\n", size, size, flops);
}

int main (int argc, char *argv[])
{
    srand((unsigned)time(NULL));
    enableDeviceMapHost();

    test_sgemm(512);
    test_sgemm(1024);
    // test_sgemm(2048);
    // test_sgemm(4096);
    return 0;
}
