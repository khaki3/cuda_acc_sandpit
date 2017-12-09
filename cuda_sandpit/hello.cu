#include <stdio.h>
#include "common.h"

__constant__ char s[] = "HELLO WORLD.";

__device__ int is_tail_thread(int tid)
{
    return (tid == blockDim.x * gridDim.x - 1);
}

__global__ void project(char *dest)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int s_len;

    if (threadIdx.x == 0) {
        for (int i = 0; ; ++i) {
            if (s[i] == '\0') {
                s_len = i;
                break;
            }
        }
    }

    __syncthreads();

    dest[tid] = s[tid % s_len];

    if (is_tail_thread(tid))
        dest[tid] = '\0';    
}

void hello(int sm_num, int thread_per_block)
{
    char *dest;
    char *dev_dest;
    int thread_size = sm_num * thread_per_block;
    cudaEvent_t start, stop;
    float elapsedTime;

    HANDLE_ERROR(
        cudaHostAlloc((void**)&dest, thread_size, cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc((void**)&dev_dest, thread_size));

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    project<<<sm_num, thread_per_block>>>(dev_dest);
    HANDLE_ERROR(cudaMemcpy(dest, dev_dest,
                            thread_size,
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("result:\t" "%s (%d)\n", dest, thread_size);
    printf("time:\t"   "%3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaFreeHost(dest));
    HANDLE_ERROR(cudaFree(dev_dest));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}

void run_hello(int sm_num, int thread_per_block)
{
    printf("CASE: (%d, %d)\n", sm_num, thread_per_block);
    hello(sm_num, thread_per_block);
    printf("\n");
}

int main (int argc, char *argv[])
{    
    cudaDeviceProp prop = fetchDeviceProperties();

    run_hello(1, 10);
    run_hello(prop.multiProcessorCount, prop.maxThreadsPerBlock);
    run_hello(1, 10);
    return 0;
}
