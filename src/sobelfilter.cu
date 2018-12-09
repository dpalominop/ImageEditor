#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define KERNEL_LENGTH 5
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float Max(int x, int y)
{
    return (x > y) ? x : y;
}

__device__ float Min(int x, int y)
{
    return (x < y) ? x : y;
}

__device__ float clamp(int x, int a, int b)
{
  return Max(a, Min(b, x));
}

__constant__ int dgx_kernel[KERNEL_LENGTH*KERNEL_LENGTH];
__constant__ int dgy_kernel[KERNEL_LENGTH*KERNEL_LENGTH];

__global__ void sobelFilterKernel(unsigned char * img_in, unsigned char * img_out,
unsigned int width, unsigned int height,
unsigned int pitch, float scale)
{
    unsigned int X = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int Y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    if(X > 1 && X < width-2 && Y > 1 && Y < height-2)
    {
        int sum_gx = 0;
        int sum_gy = 0;
        int kidx = 0;
        for(int i = -2;i<= 2;i++)
        {
            for(int j= -2;j<= 2;j++)
            {
                sum_gx += dgx_kernel[kidx++] * img_in[__umul24((Y+i),pitch) + X+j];
                sum_gy += dgy_kernel[kidx++] * img_in[__umul24((Y+i),pitch) + X+j];
            }
        }
        int sum = (int)((float)sqrt((float)(sum_gx*sum_gx + sum_gy*sum_gy)) * scale);
        img_out[__umul24(Y,pitch) + X] = clamp(sum,0,255);
    }
}

void sobelFilter(unsigned char* src_h, unsigned char* dst_h,
                 unsigned int width, unsigned int height,
                 unsigned int pitch, float scale)
{
    unsigned char *src_d, *dst_d;
    unsigned int nBytes = sizeof(unsigned char) * width*height;

    cudaMalloc((void **)&src_d, nBytes);
    cudaMalloc((void **)&dst_d, nBytes);

    cudaMemcpy(src_d, src_h, nBytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));
    sobelFilterKernel<<<grid, threads>>>(src_d, dst_d, width, height, pitch, scale);

    cudaMemcpy(dst_h, dst_d, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(src_d);
    cudaFree(dst_d);
}

void setSobelKernel(int hgx_Kernel[], int hgy_Kernel[])
{
    cudaMemcpyToSymbol(dgx_kernel, hgx_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(int));
    cudaMemcpyToSymbol(dgy_kernel, hgy_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(int));
}
