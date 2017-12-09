#include <stdio.h>
#include <stdlib.h>
#include "common.h"

/* GEMM TYPE */
typedef enum { NN, NT, TN, TT } GemmType;

typedef struct {
    size_t x;
    size_t y;
    float *body;
} Matrix;

Matrix construct_matrix(size_t x, size_t y, float *body)
{
    Matrix ret;

    ret.x = x;
    ret.y = y;
    ret.body = body;
    return ret;
}

Matrix new_matrix(size_t x, size_t y)
{
    return construct_matrix(x, y, (float*)malloc(x * y * sizeof(float)));
}

Matrix random_matrix(size_t x, size_t y)
{
    Matrix ret = new_matrix(x, y);
    size_t n = x * y;

    for (int i = 0; i < n; i++)
        ret.body[i] = (float)(rand() % 100 - 50) / 50;

    return ret;
}

void matrix_free (Matrix m)
{
    free(m.body);
}

Matrix matrix_transpose(Matrix m)
{
    Matrix ret = new_matrix(m.y, m.x);

    float *rb = ret.body, *mb = m.body;
    #pragma acc data copyout(rb[0:m.y*m.x]), copyin(mb[0:m.y*m.x])
    #pragma acc parallel
    #pragma acc loop independent
    for (int i = 0; i < m.y; i++) {
        #pragma acc loop independent
        for (int j = 0; j < m.x; j++) {
            rb[j * m.y + i] = mb[i * m.x + j];
        }
    }

    return ret;
}

double run_sgemm(GemmType type, Matrix a, Matrix b, Matrix c,
                 float alpha, float beta)
{
    double cpu0, cpu1;

    Matrix ret = new_matrix(c.x, c.y);

    switch (type) {
    case TN:
        a = matrix_transpose(a);
        break;
    case NT:
        b = matrix_transpose(b);
        break;
    case TT:
        a = matrix_transpose(a);
        b = matrix_transpose(b);
        break;
    default:
        break;
    }

    float *ab = a.body, *bb = b.body, *cb = c.body, *rb = ret.body;
    int n = a.y*a.x;
    #pragma acc data copyout(rb[0:n]), copyin(ab[0:n], bb[0:n], cb[0:n])
    {
        cpu0 = second();

        #pragma acc kernels
        #pragma acc loop independent
        for (int i = 0; i < c.y; i++) {
            #pragma acc loop independent gang(16), vector(256)
            for (int j = 0; j < c.x; j++) {
                int a_offset = i * a.x;
                int b_offset = j;
                float tmp = 0.f;
                for (int l = 0; l < a.x; l++)
                    tmp += ab[a_offset + l] * bb[b_offset + b.x * l];

                int c_offset = i * c.x + j;
                rb[c_offset] = alpha * tmp + beta * cb[c_offset];
            }
        }

        cpu1 = second();
    }

    switch (type) {
    case TN:
        matrix_free(a);
        break;
    case NT:
        matrix_free(b);
        break;
    case TT:
        matrix_free(a);
        matrix_free(b);
        break;
    default:
        break;
    }

    return (cpu1 - cpu0) * 1000.;
}

void test_sgemm(int size)
{
    Matrix a = random_matrix(size, size);
    Matrix b = random_matrix(size, size);
    Matrix c = random_matrix(size, size);
    
    double elapsed_time;
    long double flops;

    elapsed_time = run_sgemm(TN, a, b, c, 1.F, 1.F);
    flops = 2L * size * size * size * 1000 / elapsed_time;

    printf("(%d x %d)\t" "%.1Lf flops\n", size, size, flops);

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
}

int main(int argc, char *argv[])
{
    srand((unsigned)time(NULL));

    test_sgemm(512);
    test_sgemm(1024);
    test_sgemm(2048);
    test_sgemm(4096);
    return 0;
}
