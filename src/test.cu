#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "test.h"

__global__ void vectorAdditionCUDA(const float* a, const float* b, float* c, int n)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < n)
        c[ii] = a[ii] + b[ii];
}

__global__ void cuprint()
{
    printf("Printing...\n");
}

void vectorAddition(const float* a, const float* b, float* c, int n) {

    float *a_cuda, *b_cuda, *c_cuda;
    unsigned int nBytes = sizeof(float) * n;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t err = cudaSuccess;

    // allocate and copy memory into the device
    err = cudaMalloc((void **)& a_cuda, nBytes);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector A (error code %s) %d!\n", cudaGetErrorString(err), err);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)& b_cuda, nBytes);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)& c_cuda, nBytes);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(a_cuda, a, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to cudaMemcpyHostToDevice vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(b_cuda, b, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to cudaMemcpyHostToDevice vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cuprint<<<1,1>>>();
    vectorAdditionCUDA<<<blocksPerGrid, threadsPerBlock>>>(a_cuda, b_cuda, c_cuda, n);
    // load the answer back into the host
    cudaMemcpy(c, c_cuda, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
}
