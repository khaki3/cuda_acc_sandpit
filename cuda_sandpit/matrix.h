// matrix sizes must be a multiples of 32

#define MATRIX_UNIT 32
#define THREAD_COLS 8

struct Matrix {
    size_t x;
    size_t y;
    float *body;
};

void matrix_size_check (size_t x, size_t y)
{
    if (x % MATRIX_UNIT == 0 && y % MATRIX_UNIT == 0)
        return;
    printf("matrix sizes must be a multiples of 32 (%zu, %zu)\n", x, y);
    exit(EXIT_FAILURE); 
}

Matrix construct_matrix (size_t x, size_t y, float *body)
{
    Matrix ret;

    matrix_size_check(x, y);
    ret.x = x;
    ret.y = y;
    ret.body = body;
    return ret;
}

Matrix new_matrix (size_t x, size_t y)
{
    matrix_size_check(x, y);
    return construct_matrix(x, y, (float*)malloc(x * y * sizeof(float)));
}

Matrix random_matrix (size_t x, size_t y)
{
    Matrix ret = new_matrix(x, y);
    size_t n = x * y;

    for (int i = 0; i < n; i++)
        ret.body[i] = (float)(rand() % 100 - 50) / 50;

    return ret;
}

Matrix unit_matrix (size_t x)
{
    Matrix ret = new_matrix(x, x);

    for (int i = 0; i < x * x; i++)
        ret.body[i] = 0.f;
    for (int i = 0; i < x; i++)
        ret.body[i * x + i] = 1.f;

    return ret;
}

void matrix_free (Matrix m)
{
    free(m.body);
}

__global__ void matrix_transpose (Matrix m, float *tra_body)
{
    __shared__ float tmp[MATRIX_UNIT][MATRIX_UNIT];

    size_t m_y = MATRIX_UNIT * blockIdx.y + threadIdx.y;
    size_t m_x = MATRIX_UNIT * blockIdx.x + THREAD_COLS * threadIdx.x;
    size_t m_offset  = m_y * m.x + m_x;
    size_t tmp_x_offset = THREAD_COLS * threadIdx.x;

    for (int i = 0; i < THREAD_COLS; i++)
        tmp[threadIdx.y][tmp_x_offset + i] = m.body[m_offset + i];

    __syncthreads();

    size_t tra_y = MATRIX_UNIT * blockIdx.x + threadIdx.y;
    size_t tra_x = MATRIX_UNIT * blockIdx.y + THREAD_COLS * threadIdx.x;
    size_t tra_offset = tra_y * m.y + tra_x;

    for (int i = 0; i < THREAD_COLS; i++)
        tra_body[tra_offset + i] = tmp[tmp_x_offset + i][threadIdx.y];
}

Matrix matrix_into_device(Matrix m, bool tra)
{
    float *dev_body;
    size_t s = m.x * m.y * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void**)&dev_body, s));
    HANDLE_ERROR(cudaMemcpy(dev_body, m.body, s, cudaMemcpyHostToDevice));
    m = construct_matrix(m.x, m.y, dev_body);

    if (tra) {
        float *tra_body;
        dim3 blocks(m.x / MATRIX_UNIT, m.y / MATRIX_UNIT);
        dim3 threads(MATRIX_UNIT / THREAD_COLS, MATRIX_UNIT);

        HANDLE_ERROR(cudaMalloc((void**)&tra_body, s));
        matrix_transpose<<<blocks, threads>>>(m, tra_body);

        HANDLE_ERROR(cudaFree(dev_body));
        m = construct_matrix(m.y, m.x, tra_body);
    }

    return m;
}

Matrix matrix_from_device(Matrix dev_m)
{
    Matrix ret;
    size_t s = dev_m.x * dev_m.y * sizeof(float);

    ret = construct_matrix(dev_m.x, dev_m.y, (float*)malloc(s));
    HANDLE_ERROR(cudaMemcpy(ret.body, dev_m.body, s, cudaMemcpyDeviceToHost));

    return ret;
}

void matrix_free_device (Matrix m)
{
    HANDLE_ERROR(cudaFree(m.body));
}

void matrix_show (Matrix m)
{
    printf("(%d %d %p)\n", m.x, m.y, m.body);
    for (int i = 0; i < m.y; i++) {
        for (int j = 0; j < m.x; j++) {
            printf("%f ", m.body[i * m.x + j]);
        }
        printf("\n");
    }
}

Matrix matrix_transpose_cpu (Matrix m)
{
    Matrix ret = new_matrix(m.y, m.x);

    for (int i = 0; i < m.y; i++) {
        for (int j = 0; j < m.x; j++) {
            ret.body[j * m.y + i] = m.body[i * m.x + j];
        }
    }

    return ret;
}

// for debugging
bool matrix_equal (Matrix a, Matrix b)
{
    if (a.x == b.x && a.y == b.y) {
        int n = a.x * a.y;

        for (int i = 0; i < n; i++)
            if (fabs(a.body[i] - b.body[i]) > 0.0001)
                goto fail;

        return true;
    }

fail:
    return false;
}

float matrix_sum (Matrix m)
{
    float ret = 0;
    size_t n = m.y * m.x;
    for (int i = 0; i < n; i++) {
        ret += m.body[i];
    }
    return ret;
}

/* GEMM TYPE */
enum { NN, NT, TN, TT };

Matrix run_sgemm_cpu (int type, Matrix a, Matrix b, Matrix c,
                    float alpha, float beta)
{
    Matrix ret = new_matrix(c.x, c.y);

    switch (type) {
    case TN:
        a = matrix_transpose_cpu(a);
        break;
    case NT:
        b = matrix_transpose_cpu(b);
        break;
    case TT:
        a = matrix_transpose_cpu(a);
        b = matrix_transpose_cpu(b);
        break;
    default:
        break;
    }

    for (int i = 0; i < c.y; i++) {
        for (int j = 0; j < c.x; j++) {
            int a_offset = i * a.x;
            int b_offset = j;
            float tmp = 0.f;
            for (int l = 0; l < a.x; l++) {
                tmp += a.body[a_offset + l] * b.body[b_offset + b.x * l];
            }

            int c_offset = i * c.x + j;
            ret.body[c_offset] = alpha * tmp + beta * c.body[c_offset];
        }
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

    return ret;
}

/* Local Variables:  */
/* mode: cuda        */
/* End:              */
