//
// [FORK FROM]
// http://accc.riken.jp/supercom/himenobmt/download/win-mac/
//
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.h"

#define MR(mt,n,r,c,d)  mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]

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
void mat_set(Matrix* Mat,int l,float z);
void mat_set_init(Matrix* Mat);
float jacobi(int n,Matrix* M1,Matrix* M2,Matrix* M3,
             Matrix* M4,Matrix* M5,Matrix* M6,Matrix* M7);
double fflop(int,int,int);
double mflops(int,double,double);
double second();

float   omega=0.8;
Matrix  a,b,c,p,bnd,wrk1,wrk2;

int
main(int argc, char *argv[])
{
  int    i,j,k,nn;
  int    imax,jmax,kmax,mimax,mjmax,mkmax,msize[3];
  float  gosa,target;
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

  mat_set_init(&p);
  mat_set(&bnd,0,1.0);
  mat_set(&wrk1,0,0.0);
  mat_set(&wrk2,0,0.0);
  mat_set(&a,0,1.0);
  mat_set(&a,1,1.0);
  mat_set(&a,2,1.0);
  mat_set(&a,3,1.0/6.0);
  mat_set(&b,0,0.0);
  mat_set(&b,1,0.0);
  mat_set(&b,2,0.0);
  mat_set(&c,0,1.0);
  mat_set(&c,1,1.0);
  mat_set(&c,2,1.0);

  /*
   *    Start measuring
   */
  jacobi(1,&a,&b,&c,&p,&bnd,&wrk1,&wrk2); /* wake up */

  nn= 3;
  printf(" Start rehearsal measurement process.\n");
  printf(" Measure the performance in %d times.\n\n",nn);

  cpu0= second();
  gosa= jacobi(nn,&a,&b,&c,&p,&bnd,&wrk1,&wrk2);
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
  gosa = jacobi(nn,&a,&b,&c,&p,&bnd,&wrk1,&wrk2);
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
mat_set(Matrix* Mat, int l, float val)
{
  int i,j,k;

    for(i=0; i<Mat->mrows; i++)
      for(j=0; j<Mat->mcols; j++)
        for(k=0; k<Mat->mdeps; k++)
          MR(Mat,l,i,j,k)= val;
}

void
mat_set_init(Matrix* Mat)
{
  int  i,j,k,l;
  float tt;

  for(i=0; i<Mat->mrows; i++)
    for(j=0; j<Mat->mcols; j++)
      for(k=0; k<Mat->mdeps; k++)
        MR(Mat,0,i,j,k)= (float)(i*i)
          /(float)((Mat->mrows - 1)*(Mat->mrows - 1));
}

///
///
///

#undef MR
#define MR(mt,n,r,c,d)                          \
    mt##_m[(n) * mrows * mcols * mdeps +        \
           (r) * mcols * mdeps +                \
           (c) * mdeps +                        \
           (d)]

float
jacobi(int nn, Matrix* a,Matrix* b,Matrix* c,
       Matrix* p,Matrix* bnd,Matrix* wrk1,Matrix* wrk2)
{
  size_t mrows = p->mrows;
  size_t mcols = p->mcols;
  size_t mdeps = p->mdeps;

  int    i,j,k,n,imax,jmax,kmax;
  float  gosa,s0,ss;

  float *a_m, *b_m, *c_m, *p_m, *bnd_m, *wrk1_m, *wrk2_m;
  size_t bs; /* box size */

  imax= mrows-1;
  jmax= mcols-1;
  kmax= mdeps-1;

  a_m = a->m;
  b_m = b->m;
  c_m = c->m;
  p_m = p->m;
  bnd_m = bnd->m;
  wrk1_m = wrk1->m;
  wrk2_m = wrk2->m;

  bs = mrows * mcols * mdeps;

  #pragma acc data copy (p_m[bs])
  #pragma acc data copyin \
      (a_m[bs * 4], b_m[bs * 3], c_m[bs * 3], \
       bnd_m[bs], wrk1_m[bs], wrk2_m[bs])
  for(n=0 ; n<nn ; n++){
    gosa = 0.0;

    #pragma acc parallel loop reduction (+ : gosa)
    for(i=1 ; i<imax; i++)
      #pragma acc loop
      for(j=1 ; j<jmax ; j++)
        #pragma acc loop
        for(k=1 ; k<kmax ; k++){
          s0= MR(a,0,i,j,k)*MR(p,0,i+1,j,  k)
            + MR(a,1,i,j,k)*MR(p,0,i,  j+1,k)
            + MR(a,2,i,j,k)*MR(p,0,i,  j,  k+1)
            + MR(b,0,i,j,k)
             *( MR(p,0,i+1,j+1,k) - MR(p,0,i+1,j-1,k)
              - MR(p,0,i-1,j+1,k) + MR(p,0,i-1,j-1,k) )
            + MR(b,1,i,j,k)
             *( MR(p,0,i,j+1,k+1) - MR(p,0,i,j-1,k+1)
              - MR(p,0,i,j+1,k-1) + MR(p,0,i,j-1,k-1) )
            + MR(b,2,i,j,k)
             *( MR(p,0,i+1,j,k+1) - MR(p,0,i-1,j,k+1)
              - MR(p,0,i+1,j,k-1) + MR(p,0,i-1,j,k-1) )
            + MR(c,0,i,j,k) * MR(p,0,i-1,j,  k)
            + MR(c,1,i,j,k) * MR(p,0,i,  j-1,k)
            + MR(c,2,i,j,k) * MR(p,0,i,  j,  k-1)
            + MR(wrk1,0,i,j,k);

          ss= (s0*MR(a,3,i,j,k) - MR(p,0,i,j,k))*MR(bnd,0,i,j,k);

          gosa+= ss*ss;
          MR(wrk2,0,i,j,k)= MR(p,0,i,j,k) + omega*ss;
        }

    #pragma acc parallel loop independent
    for(i=1 ; i<imax ; i++)
      #pragma acc loop independent
      for(j=1 ; j<jmax ; j++)
        #pragma acc loop independent
        for(k=1 ; k<kmax ; k++)
          MR(p,0,i,j,k)= MR(wrk2,0,i,j,k);
    
  } /* end n loop */

  return(gosa);
}
