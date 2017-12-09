#include <stdio.h>

#define N 5
#define M 10

__global__ void calc(int *a, int n)
{
    int tmp = 0;

    for (int i = 0; i < n; i++)
        tmp += a[i];
    a[0] = tmp;
}

__global__ void calc_pitch(int *a, int x, int y, size_t pitch)
{
    int tmp = 0;
    void *ptr = (void*)*a;

    for (int i = 0; i < y; i++)
        for (int j = 0; j < x; j++)
            tmp += *(int*)(ptr + i * pitch + j);
    *a = tmp;
}

void print_array(int a[N][M])
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    int a[N][M];
    int answer = 0;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            int tmp = i * M + j;
            a[i][j] = tmp;
            answer += tmp;
        }

    puts("(original)");
    print_array(a);

    puts("(host -> host) via cudaMemcpy2D");
    int b[N][M];
    cudaMemcpy2D(b, sizeof(int) * M,
                 a, sizeof(int) * M, sizeof(int) * M, N,
                 cudaMemcpyHostToHost);
    print_array(b);

    puts("(host -> device -> host) via cudaMemcpy using cudaMalloc");
    int *dev_a1;
    int c[N][M];
    size_t map_size = N * M * sizeof(int);
    cudaMalloc(&dev_a1, map_size);
    cudaMemcpy(dev_a1, a, map_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c, dev_a1, map_size, cudaMemcpyDeviceToHost);
    print_array(c);

    puts("(host -> device -> host) via cudaMemcpy2D using cudaMalloc");
    int *dev_a2;
    int d[N][M];
    cudaMalloc(&dev_a2, map_size);
    cudaMemcpy2D(dev_a2, sizeof(int) * M,
                 a, sizeof(int) * M, sizeof(int) * M, N,
                 cudaMemcpyHostToDevice);
    cudaMemcpy2D(d, sizeof(int) * M,
                 dev_a2, sizeof(int) * M, sizeof(int) * M, N,
                 cudaMemcpyDeviceToHost);
    print_array(d);

    puts("(host -> device -> host) via cudaMemcpy2D using cudaMallocPitch");
    int *dev_a3;
    int e[N][M];
    size_t pitch;
    cudaMallocPitch(&dev_a3, &pitch, sizeof(int) * M, N);
    cudaMemcpy2D(dev_a3, pitch,
                 a, sizeof(int) * M, sizeof(int) * M, N,
                 cudaMemcpyHostToDevice);
    cudaMemcpy2D(e, sizeof(int) * M,
                 dev_a3, pitch, sizeof(int) * M, N,
                 cudaMemcpyDeviceToHost);
    print_array(e);

    puts("(calculate test)");
    int result;
    calc<<<1,1>>>(dev_a2, M * N);
    cudaMemcpy(&result, dev_a2, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d == %d\n", answer, result);

    puts("(calculate test) according to pitch");
    calc_pitch<<<1,1>>>(dev_a3, M, N, pitch);
    cudaMemcpy(&result, dev_a3, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d == %d\n", answer, result);

    return 0;
}
