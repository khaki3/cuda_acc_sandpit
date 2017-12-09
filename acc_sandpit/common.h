#include <sys/time.h>

// copy from himeno benchmark
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
