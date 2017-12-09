// Ref:
//  http://www.cc.u-tokyo.ac.jp/support/press/news/VOL12/No3/201005_gpgpu2.pdf
#include <stdio.h>
#include "common.h"
#include "matrix.h"
#define KERNEL2

#define BLOCKSIZE 64

#ifdef KERNEL0
// original
__global__ void kernel (Matrix a, Matrix b, Matrix c,
                        float alpha, float beta)
{
    int my_y = blockIdx.x * blockDim.x + threadIdx.x;
    int a_offset = my_y * a.x;
    int c_offset = my_y * c.x;

    for (int i = 0; i < c.x; i++) {
        int c_index = c_offset + i;
        float acc = 0;

        for (int j = 0; j < a.x; j++)
            acc += a.body[a_offset + j] * b.body[i + j * b.x];

        c.body[c_index] = alpha * acc + beta * c.body[c_index];
    }
}
#endif

#ifdef KERNEL1
// simd
__global__ void kernel (Matrix a, Matrix b, Matrix c,
                        float alpha, float beta)
{
    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y;

    int a_offset = my_y * a.x;
    float acc = 0;

    for (int i = 0; i < a.x; i++)
        acc += a.body[a_offset + i] * b.body[my_x + i * b.x];

    int c_index = my_y * c.x + my_x;
    c.body[c_index] = alpha * acc + beta * c.body[c_index];
}
#endif

#ifdef KERNEL2
// shared memory
__global__ void kernel (Matrix a, Matrix b, Matrix c,
                        float alpha, float beta)
{
    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y;

    __shared__ float sb[BLOCKSIZE];

    int a_offset = my_y * a.x;
    float acc = 0;

    for (int is = 0; is < a.x; is += BLOCKSIZE) {
        int b_offset = my_x + is * b.x;

        sb[threadIdx.x] = a.body[a_offset + is];
        __syncthreads();

        for (int i = 0; i < BLOCKSIZE; i++)
            acc += sb[i] * b.body[b_offset + i * b.x];

        __syncthreads();
    }

    int c_index = my_y * c.x + my_x;
    c.body[c_index] = alpha * acc + beta * c.body[c_index];
}
#endif

float run_sgemm (int type, Matrix a, Matrix b, Matrix c,
                  float alpha, float beta)
{
    #ifdef KERNEL0
    int block = BLOCKSIZE;
    int grid  = c.y / thread_num;
    #endif

    #if defined(KERNEL1) || defined(KERNEL2)
    int block = BLOCKSIZE;
    dim3 grid(c.x / block, c.y);
    #endif

    Matrix dev_a = matrix_into_device(a, type == TN || type == TT);
    Matrix dev_b = matrix_into_device(b, type == NT || type == TT);
    Matrix dev_c = matrix_into_device(c, false);

    cudaEvent_t start, stop;
    float elapsed_time;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    kernel<<<grid, block>>>(dev_a, dev_b, dev_c, alpha, beta);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

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

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

int main (int argc, char *argv[])
{
    srand((unsigned)time(NULL));

    test_sgemm(512);
    test_sgemm(1024);
    test_sgemm(2048);
    test_sgemm(4096);
    return 0;
}
