#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define KERNEL_LENGTH 11
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

int iDivUp2(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float Max2(int x, int y)
{
    return (x > y) ? x : y;
}

__device__ float Min2(int x, int y)
{
    return (x < y) ? x : y;
}

__device__ float clamp2(int x, int a, int b)
{
  return Max2(a, Min2(b, x));
}

__constant__ float fog_kernel[KERNEL_LENGTH*KERNEL_LENGTH];

__global__ void fogFilterKernel(unsigned char * img_in, unsigned char * img_out,
                                unsigned int width, unsigned int height,
                                unsigned int pitch, float scale)
{
    unsigned int X = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int Y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    if(X > 4 && X < width-5 && Y > 4 && Y < height-5)
    {
        float sum = 0.0;
        int kidx = 0;
        for(int i = -5;i<= 5;i++)
        {
            for(int j= -5;j<= 5;j++)
            {
                sum += fog_kernel[kidx++] * img_in[__umul24((Y+i),pitch) + X+j];
            }
        }
        sum = sum * scale;
        img_out[__umul24(Y,pitch) + X] = clamp2(sum,0,255);
    }
}

void fogFilter(unsigned char* src_h, unsigned char* dst_h,
                 unsigned int width, unsigned int height,
                 unsigned int pitch, float scale)
{
    unsigned char *src_d, *dst_d;
    unsigned int nBytes = sizeof(unsigned char) * width*height;

    cudaMalloc((void **)&src_d, nBytes);
    cudaMalloc((void **)&dst_d, nBytes);

    cudaMemcpy(src_d, src_h, nBytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp2(width, BLOCKDIM_X), iDivUp2(height, BLOCKDIM_Y));
    fogFilterKernel<<<grid, threads>>>(src_d, dst_d, width, height, pitch, scale);

    cudaMemcpy(dst_h, dst_d, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(src_d);
    cudaFree(dst_d);
}

void setFogKernel(float h_Kernel[])
{
    cudaMemcpyToSymbol(fog_kernel, h_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(float));
}
