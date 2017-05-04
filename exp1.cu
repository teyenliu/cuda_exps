#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
  
#include <stdio.h>  
  
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);  
void printDevProp(cudaDeviceProp &devProp);

__global__ void addKernel(int *c, const int *a, const int *b)  
{
    // we only give 1 dimension of size so that only x value matters.
    int i = threadIdx.x;  
    c[i] = a[i] + b[i];  
}  
  
int main()  
{  
    const int arraySize = 5;  
    const int a[arraySize] = { 1, 2, 3, 4, 5 };  
    const int b[arraySize] = { 10, 20, 30, 40, 50 };  
    int c[arraySize] = { 0 };  
 
    // Add vectors in parallel.  
    cudaError_t cudaStatus;  
    int num = 0;  
    cudaDeviceProp prop;  
    cudaStatus = cudaGetDeviceCount(&num);  
    for(int i = 0;i<num;i++)  
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i); 
        cudaGetDeviceProperties(&prop,i);
        printDevProp(prop);  
    }  
 
    // Add vectors in parallel.  
    cudaStatus = addWithCuda(c, a, b, arraySize);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "addWithCuda failed!");  
        return 1;  
    }  
  
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",  
        c[0], c[1], c[2], c[3], c[4]);  
  
    // cudaThreadExit must be called before exiting in order for profiling and  
    // tracing tools such as Nsight and Visual Profiler to show complete traces.  
    cudaStatus = cudaThreadExit();  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaThreadExit failed!");  
        return 1;  
    }  
  
    return 0;  
}  
  
// Helper function for using CUDA to add vectors in parallel.  
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)  
{  
    int *dev_a = 0;  
    int *dev_b = 0;  
    int *dev_c = 0;  
    cudaError_t cudaStatus;  
  
    // Choose which GPU to run on, change this on a multi-GPU system.  
    cudaStatus = cudaSetDevice(0);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
        goto Error;  
    }  
  
    // Allocate GPU buffers for three vectors (two input, one output)    .  
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
  
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
  
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
  
    // Copy input vectors from host memory to GPU buffers.  
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
  
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
  
    // Launch a kernel on the GPU with one thread for each element.  
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);  
  
    // cudaThreadSynchronize waits for the kernel to finish, and returns  
    // any errors encountered during the launch.  
    cudaStatus = cudaThreadSynchronize();  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  
        goto Error;  
    }  
  
    // Copy output vector from GPU buffer to host memory.  
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }  
  
Error:  
    cudaFree(dev_c);  
    cudaFree(dev_a);  
    cudaFree(dev_b);  
      
    return cudaStatus;  
}


// Print device properties
void printDevProp(cudaDeviceProp &devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
