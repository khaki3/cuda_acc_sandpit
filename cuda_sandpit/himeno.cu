//
// [FORK FROM]
// http://accc.riken.jp/supercom/himenobmt/download/win-mac/
//
// [REF]
// http://www.training.prace-ri.eu/uploads/tx_pracetmo/Himeno_Benchmark_on_GPU_Cluster.pdf
//

#include <stdio.h>
#include <sys/time.h>
#include "common.h"

#define MR(mt,n,r,c,d)  mt.m[(n) * mt.mrows * mt.mcols * mt.mdeps + (r) * mt.mcols* mt.mdeps + (c) * mt.mdeps + (d)]
#define GB (1024 * 1024 * 1024)

struct Mat {
    float* m;
    int mnums;
    int mrows;
    int mcols;
    int mdeps;
};

/* prototypes */
typedef struct Mat Matrix;

int newMat(Matrix* Mat, int mnums, int mrows, int mcols, int mdeps);
void clearMat(Matrix* Mat);
void set_param(int i[],char *size);
void mat_set(Matrix Mat,int l,float z);
void mat_set_init(Matrix Mat);
float jacobi(int n,Matrix M1,Matrix M2,Matrix M3,
             Matrix M4,Matrix M5,Matrix M6,Matrix M7);
double fflop(int,int,int);
double mflops(int,double,double);
double second();

__constant__ float omega=0.8;
Matrix  a,b,c,p,bnd,wrk1,wrk2;

int
main(int argc, char *argv[])
{
    int    nn;
    int    imax,jmax,kmax,mimax,mjmax,mkmax,msize[3];
    float  gosa, target;
    double  cpu0,cpu1,cpu,flop;
    char   size[10];

    if(argc == 2){
        strcpy(size,argv[1]);
    } else {
        printf("For example: \n");
        printf(" Grid-size= XS (32x32x64)\n");
        printf("\t    S  (64x64x128)\n");
        printf("\t    M  (128x128x256)\n");
        printf("\t    L  (256x256x512)\n");
        printf("\t    XL (512x512x1024)\n\n");
        printf("Grid-size = ");
        scanf("%s",size);
        printf("\n");
    }

    set_param(msize,size);
  
    mimax= msize[0];
    mjmax= msize[1];
    mkmax= msize[2];
    imax= mimax-1;
    jmax= mjmax-1;
    kmax= mkmax-1;

    target = 60.0;

    printf("mimax = %d mjmax = %d mkmax = %d\n",mimax,mjmax,mkmax);
    printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);

    /*
     *    Initializing matrixes
     */
    newMat(&p,1,mimax,mjmax,mkmax);
    newMat(&bnd,1,mimax,mjmax,mkmax);
    newMat(&wrk1,1,mimax,mjmax,mkmax);
    newMat(&wrk2,1,mimax,mjmax,mkmax);
    newMat(&a,4,mimax,mjmax,mkmax);
    newMat(&b,3,mimax,mjmax,mkmax);
    newMat(&c,3,mimax,mjmax,mkmax);

    mat_set_init(p);
    mat_set(bnd,0,1.0);
    mat_set(wrk1,0,0.0);
    mat_set_init(wrk2);
    mat_set(a,0,1.0);
    mat_set(a,1,1.0);
    mat_set(a,2,1.0);
    mat_set(a,3,1.0/6.0);
    mat_set(b,0,0.0);
    mat_set(b,1,0.0);
    mat_set(b,2,0.0);
    mat_set(c,0,1.0);
    mat_set(c,1,1.0);
    mat_set(c,2,1.0);

    /*
     *    Start measuring
     */
    gosa= jacobi(1,a,b,c,p,bnd,wrk1,wrk2); // wake up

    nn= 3;
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n",nn);

    cpu0= second();
    gosa= jacobi(nn,a,b,c,p,bnd,wrk1,wrk2);
    cpu1= second();
    cpu= cpu1 - cpu0;
    flop= fflop(imax,jmax,kmax);

    printf(" MFLOPS: %f time(s): %f %e\n\n",
           mflops(nn,cpu,flop),cpu,gosa);

    nn= (int)(target/(cpu/3.0));

    printf(" Now, start the actual measurement process.\n");
    printf(" The loop will be excuted in %d times\n",nn);
    printf(" This will take about one minute.\n");
    printf(" Wait for a while\n\n");

    cpu0 = second();
    gosa = jacobi(nn,a,b,c,p,bnd,wrk1,wrk2);
    cpu1 = second();
    cpu = cpu1 - cpu0;

    printf(" Loop executed for %d times\n",nn);
    printf(" Gosa : %e \n",gosa);
    printf(" MFLOPS measured : %f\tcpu : %f\n",mflops(nn,cpu,flop),cpu);
    printf(" Score based on Pentium III 600MHz using Fortran 77: %f\n",
           mflops(nn,cpu,flop)/82,84);

    /*
     *   Matrix free
     */ 
    clearMat(&p);
    clearMat(&bnd);
    clearMat(&wrk1);
    clearMat(&wrk2);
    clearMat(&a);
    clearMat(&b);
    clearMat(&c);
  
    return (0);
}

double
fflop(int mx,int my, int mz)
{
    return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
}

double
mflops(int nn,double cpu,double flop)
{
    return(flop/cpu*1.e-6*(double)nn);
}

void
set_param(int is[],char *size)
{
    if(!strcmp(size,"XS") || !strcmp(size,"xs")){
        is[0]= 32;
        is[1]= 32;
        is[2]= 64;
        return;
    }
    if(!strcmp(size,"S") || !strcmp(size,"s")){
        is[0]= 64;
        is[1]= 64;
        is[2]= 128;
        return;
    }
    if(!strcmp(size,"M") || !strcmp(size,"m")){
        is[0]= 128;
        is[1]= 128;
        is[2]= 256;
        return;
    }
    if(!strcmp(size,"L") || !strcmp(size,"l")){
        is[0]= 256;
        is[1]= 256;
        is[2]= 512;
        return;
    }
    if(!strcmp(size,"XL") || !strcmp(size,"xl")){
        is[0]= 512;
        is[1]= 512;
        is[2]= 1024;
        return;
    } else {
        printf("Invalid input character !!\n");
        exit(6);
    }
}

int
newMat(Matrix* Mat, int mnums,int mrows, int mcols, int mdeps)
{
    Mat->mnums= mnums;
    Mat->mrows= mrows;
    Mat->mcols= mcols;
    Mat->mdeps= mdeps;
    Mat->m= NULL;
    Mat->m= (float*) 
        malloc(mnums * mrows * mcols * mdeps * sizeof(float));
  
    return(Mat->m != NULL) ? 1:0;
}

void
clearMat(Matrix* Mat)
{
    if(Mat->m)
        free(Mat->m);
    Mat->m= NULL;
    Mat->mnums= 0;
    Mat->mcols= 0;
    Mat->mrows= 0;
    Mat->mdeps= 0;
}

void
mat_set(Matrix Mat, int l, float val)
{
    for(int i=0; i<Mat.mrows; i++)
        for(int j=0; j<Mat.mcols; j++)
            for(int k=0; k<Mat.mdeps; k++)
                MR(Mat,l,i,j,k)= val;
}

void
mat_set_init(Matrix Mat)
{
    for(int i=0; i<Mat.mrows; i++)
        for(int j=0; j<Mat.mcols; j++)
            for(int k=0; k<Mat.mdeps; k++)
                MR(Mat,0,i,j,k)= (float)(i*i)
                    /(float)((Mat.mrows - 1)*(Mat.mrows - 1));
}

double
second()
{
    struct timeval tm;
    double t ;

    static int base_sec = 0,base_usec = 0;

    gettimeofday(&tm, NULL);
  
    if(base_sec == 0 && base_usec == 0)
    {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    } else {
        t = (double) (tm.tv_sec-base_sec) + 
            ((double) (tm.tv_usec-base_usec))/1.0e6 ;
    }

    return t ;
}

////
////
////

int
newMatDev(Matrix* dev, int rows, int cols, int deps)
{
    dev->mnums = 1;
    dev->mrows = rows;
    dev->mcols = cols;
    dev->mdeps = deps;
    dev->m = NULL;

    size_t m_size
        = dev->mrows
        * dev->mcols
        * dev->mdeps
        * sizeof(float);

    HANDLE_ERROR(cudaMalloc(&(dev->m), m_size));

    return(dev->m != NULL) ? 1:0;
}

typedef enum { HostToDevice, DeviceToHost, DeviceToHostSkipHeadRow } transferKind;

void matTransfer(Matrix *dest, Matrix *src, int n, int r0, int r1, transferKind k)
{
    Matrix *dev = (k == HostToDevice) ? dest : src;
    Matrix *hst = (k == HostToDevice) ? src : dest;

    if (k == HostToDevice)
        dev->mrows = (r1 - r0 + 1);

    size_t rect          = hst->mcols * hst->mdeps;
    size_t transfer_size = dev->mrows * rect * sizeof(float);
    size_t offset        = (n * hst->mrows + r0) * rect;
    float *hst_pos       = hst->m + offset;

    switch (k) {
    case HostToDevice:
        HANDLE_ERROR(cudaMemcpy(dev->m, hst_pos, transfer_size, cudaMemcpyHostToDevice));
        break;

    case DeviceToHost:
        HANDLE_ERROR(cudaMemcpy(hst_pos, dev->m, transfer_size, cudaMemcpyDeviceToHost));
        break;

    case DeviceToHostSkipHeadRow:
        HANDLE_ERROR(cudaMemcpy(hst_pos + rect, dev->m + rect,
                                transfer_size - rect * sizeof(float), cudaMemcpyDeviceToHost));
        break;
    }
}

void
clearMatDev(Matrix* dev)
{
    if(dev->m)
        cudaFree(dev->m);
    dev->m = NULL;
    dev->mnums= 0;
    dev->mcols= 0;
    dev->mrows= 0;
    dev->mdeps= 0;
}

#define REDUCT_BLOCK_SIZE 256

__global__
void
sum_kernel (float *ptr)
{
    __shared__ float cache[REDUCT_BLOCK_SIZE];
    int cache_index = threadIdx.x;

    cache[cache_index] = ptr[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cache_index < i)
            cache[cache_index] += cache[cache_index + i];
        __syncthreads();
        i /= 2;
    }

    if (cache_index == 0)
        ptr[blockIdx.x] = cache[0];
}

__global__
void
jacobi_kernel (Matrix a0, Matrix a1, Matrix a2, Matrix a3,
               Matrix b0, Matrix b1, Matrix b2,
               Matrix c0, Matrix c1, Matrix c2,
               Matrix p, Matrix bnd, Matrix wrk1, Matrix wrk2,
               float *gosa);

#define BLOCK_SIZE_COLS 4
#define BLOCK_SIZE_DEPS 128

float
jacobi(int nn, Matrix a, Matrix b, Matrix c,
       Matrix p, Matrix bnd, Matrix wrk1, Matrix wrk2)
{
    float gosa = 0.f;

    int imax = p.mrows - 1;
    int jmax = p.mcols - 1;
    int kmax = p.mdeps - 1;

    int box  = p.mrows * p.mcols * p.mdeps;
    int divs = box * 14L * sizeof(float) / GB;
    int row_step = max(1, (imax - 1) / (divs + 1));
    int rows = row_step + 2;

    Matrix dev_a0, dev_a1, dev_a2, dev_a3;
    Matrix dev_b0, dev_b1, dev_b2;
    Matrix dev_c0, dev_c1, dev_c2;
    Matrix dev_p, dev_bnd, dev_wrk1, dev_wrk2;

    newMatDev(&dev_a0,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_a1,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_a2,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_a3,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_b0,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_b1,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_b2,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_c0,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_c1,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_c2,   rows, p.mcols, p.mdeps);
    newMatDev(&dev_p,    rows, p.mcols, p.mdeps);
    newMatDev(&dev_bnd,  rows, p.mcols, p.mdeps);
    newMatDev(&dev_wrk1, rows, p.mcols, p.mdeps);
    newMatDev(&dev_wrk2, rows, p.mcols, p.mdeps);

    int block_size_cols = BLOCK_SIZE_COLS;
    int block_size_deps = BLOCK_SIZE_DEPS;

    dim3 block(block_size_cols + 2, block_size_deps + 2);
    dim3 grid((jmax + block_size_cols - 1) / block_size_cols,
              (kmax + block_size_deps - 1) / block_size_deps);

    float *dev_gosas;
    size_t dev_gosas_size = box * sizeof(float);
    HANDLE_ERROR(cudaMalloc(&dev_gosas, dev_gosas_size));

    int reduct_block_size  = REDUCT_BLOCK_SIZE;
    int reduct_grid_size   = box / REDUCT_BLOCK_SIZE;
    size_t host_gosas_size = reduct_grid_size * sizeof(float);
    float *host_gosas      = (float*)malloc(host_gosas_size);

    for (int n = 0; n < nn; n++) {
        HANDLE_ERROR(cudaMemset(dev_gosas, 0, dev_gosas_size));

        int tail = imax - 1;

        for (int i = 0; i < tail; i += row_step) {

            int r1 = min(i + row_step + 1, imax);
            matTransfer(&dev_a0,   &a,    0, i, r1, HostToDevice);
            matTransfer(&dev_a1,   &a,    1, i, r1, HostToDevice);
            matTransfer(&dev_a2,   &a,    2, i, r1, HostToDevice);
            matTransfer(&dev_a3,   &a,    3, i, r1, HostToDevice);
            matTransfer(&dev_b0,   &b,    0, i, r1, HostToDevice);
            matTransfer(&dev_b1,   &b,    1, i, r1, HostToDevice);
            matTransfer(&dev_b2,   &b,    2, i, r1, HostToDevice);
            matTransfer(&dev_c0,   &c,    0, i, r1, HostToDevice);
            matTransfer(&dev_c1,   &c,    1, i, r1, HostToDevice);
            matTransfer(&dev_c2,   &c,    2, i, r1, HostToDevice);
            matTransfer(&dev_p,    &p,    0, i, r1, HostToDevice);
            matTransfer(&dev_bnd,  &bnd,  0, i, r1, HostToDevice);
            matTransfer(&dev_wrk1, &wrk1, 0, i, r1, HostToDevice);
            matTransfer(&dev_wrk2, &wrk2, 0, i, r1, HostToDevice);

            jacobi_kernel<<<grid, block>>>(
                dev_a0, dev_a1, dev_a2, dev_a3,
                dev_b0, dev_b1, dev_b2,
                dev_c0, dev_c1, dev_c2,
                dev_p, dev_bnd, dev_wrk1, dev_wrk2,
                dev_gosas
                );

            matTransfer(&wrk2, &dev_wrk2, 0, i, r1, DeviceToHostSkipHeadRow);
        }

        Matrix swap = p;
        p    = wrk2;
        wrk2 = swap;

        //
        // reduction
        //
        sum_kernel<<<reduct_grid_size, reduct_block_size>>>(dev_gosas);
        HANDLE_ERROR(cudaMemcpy(host_gosas, dev_gosas, host_gosas_size, cudaMemcpyDeviceToHost));

        gosa = 0.f;
        for (int i = 0; i < reduct_grid_size; i++)
            gosa += host_gosas[i];
    }

    //
    // free
    //
    clearMatDev(&dev_a0);
    clearMatDev(&dev_a1);
    clearMatDev(&dev_a2);
    clearMatDev(&dev_a3);
    clearMatDev(&dev_b0);
    clearMatDev(&dev_b1);
    clearMatDev(&dev_b2);
    clearMatDev(&dev_c0);
    clearMatDev(&dev_c1);
    clearMatDev(&dev_c2);
    clearMatDev(&dev_p);
    clearMatDev(&dev_bnd);
    clearMatDev(&dev_wrk1);
    clearMatDev(&dev_wrk2);
    cudaFree(dev_gosas);
    free(host_gosas);

    return gosa;
}

__global__
void
jacobi_kernel (Matrix a0, Matrix a1, Matrix a2, Matrix a3,
               Matrix b0, Matrix b1, Matrix b2,
               Matrix c0, Matrix c1, Matrix c2,
               Matrix p, Matrix bnd, Matrix wrk1, Matrix wrk2,
               float *gosas)
{
    __shared__ float p_cache[3][BLOCK_SIZE_COLS + 2][BLOCK_SIZE_DEPS + 2];
    float s0, ss;

    int imax = p.mrows - 1;
    int jmax = p.mcols - 1;
    int kmax = p.mdeps - 1;

    int line_step = p.mdeps;
    int rect_step = p.mcols * p.mdeps;

    int j = blockIdx.x * BLOCK_SIZE_COLS + threadIdx.x;
    int k = blockIdx.y * BLOCK_SIZE_DEPS + threadIdx.y;

    int cache_j = threadIdx.x;
    int cache_k = threadIdx.y;

    if (!(j <= jmax && k <= kmax))
        return;

    int offset = line_step * j + k;

    p_cache[1][cache_j][cache_k] = p.m[offset];
    p_cache[2][cache_j][cache_k] = p.m[rect_step + offset];

    bool inner =
        cache_j > 0 && cache_j <= BLOCK_SIZE_COLS && j < jmax &&
        cache_k > 0 && cache_k <= BLOCK_SIZE_DEPS && k < kmax;

    for (int i = 1; i < imax; i++) {
        int index = rect_step * i + offset;
        p_cache[0][cache_j][cache_k] = p_cache[1][cache_j][cache_k];
        p_cache[1][cache_j][cache_k] = p_cache[2][cache_j][cache_k];
        p_cache[2][cache_j][cache_k] = p.m[index + rect_step];

        __syncthreads();

        if (inner) {
            s0 =  a0.m[index] * p_cache[2][cache_j][cache_k]
                + a1.m[index] * p_cache[1][cache_j + 1][cache_k]
                + a2.m[index] * p_cache[1][cache_j][cache_k + 1]

                + b0.m[index]
                * (  p_cache[2][cache_j + 1][cache_k]
                   - p_cache[2][cache_j - 1][cache_k]
                   - p_cache[0][cache_j + 1][cache_k]
                   + p_cache[0][cache_j - 1][cache_k])

                + b1.m[index]
                * (  p_cache[1][cache_j + 1][cache_k + 1]
                   - p_cache[1][cache_j - 1][cache_k + 1]
                   - p_cache[1][cache_j + 1][cache_k - 1]
                   + p_cache[1][cache_j - 1][cache_k - 1])

                + b2.m[index]
                * (  p_cache[2][cache_j][cache_k + 1]
                   - p_cache[0][cache_j][cache_k + 1]
                   - p_cache[2][cache_j][cache_k - 1]
                   + p_cache[0][cache_j][cache_k - 1])

                + c0.m[index] * p_cache[0][cache_j][cache_k]
                + c1.m[index] * p_cache[1][cache_j - 1][cache_k]
                + c2.m[index] * p_cache[1][cache_j][cache_k - 1]
                + wrk1.m[index];

            ss = (s0 * a3.m[index] - p_cache[1][cache_j][cache_k]) * bnd.m[index];
            wrk2.m[index] = p_cache[1][cache_j][cache_k] + omega * ss;
            gosas[index] += ss * ss;
        }
    }
}
