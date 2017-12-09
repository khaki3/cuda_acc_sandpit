/*
  This software contains source code provided by NVIDIA Corporation.
  https://developer.nvidia.com/cuda-example
*/
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE); 
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

////
////
////

cudaDeviceProp fetchDeviceProperties()
{
    cudaDeviceProp prop;
    int dev_count;

    HANDLE_ERROR(cudaGetDeviceCount(&dev_count));

    if (dev_count == 0) {
        printf("cudaGetDeviceCount == 0\n");
        exit(EXIT_FAILURE);
    }

    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    return prop;
}

void enableDeviceMapHost()
{
    cudaDeviceProp prop = fetchDeviceProperties();

    if (prop.canMapHostMemory == 0)  {
        printf("canMapHostMemory == \n");
        exit(EXIT_FAILURE);
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);
}
