#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cufft.h>
#include <stdlib.h>

__global__ void normalize(cufftComplex* src, unsigned char* dst, int w, int h)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < h*w)
    {
        const int luma32 = sqrt((src[ii].x)*(src[ii].x)+(src[ii].y)*(src[ii].y))/(1.0f*w);
        dst[ii] = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
    }
}

__global__ void swapRows(cufftComplex* src, int w, int h)
{
    cufftComplex *tmp;
    tmp = (cufftComplex *)malloc(sizeof(cufftComplex)*w/2);

    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < h/2)
    {
        cufftComplex* i_start = src+ii*w;
        cufftComplex* i_end = src + (ii+h/2)*w + w/2;
        memcpy(tmp, i_start, sizeof(cufftComplex)*w/2);
        memcpy(i_start, i_end, sizeof(cufftComplex)*w/2);
        memcpy(i_end, tmp, sizeof(cufftComplex)*w/2);
    }else if (ii < h) {
        cufftComplex* i_start = src+(ii-h/2)*w +w/2;
        cufftComplex* i_end = src + ii*w;
        memcpy(tmp, i_start, sizeof(cufftComplex)*w/2);
        memcpy(i_start, i_end, sizeof(cufftComplex)*w/2);
        memcpy(i_end, tmp, sizeof(cufftComplex)*w/2);
    }

    free(tmp);
}

__global__ void unnormalize(unsigned char* src, cufftComplex* dst, int w, int h)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < h*w)
    {
        dst[ii].x = (float)src[ii];
        dst[ii].y = 0;
    }
}

void img2fft(unsigned char *src, unsigned char *dst, int w, int h){
    cufftHandle plan;
    cufftComplex *src_d;
    cufftComplex *dst_d;
    unsigned char* uchar_d;

    checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(cufftComplex)*w*h));
    checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(cufftComplex)*w*h));
    checkCudaErrors(cudaMalloc((void**)&uchar_d, sizeof(unsigned char)*w*h));

    checkCudaErrors(cudaMemcpy(uchar_d, src, h*w *sizeof(unsigned char), cudaMemcpyHostToDevice));

    int threadsPerBlock = 32;
    int blocksPerGrid   = (h*w + threadsPerBlock - 1) / threadsPerBlock;
    unnormalize<<<blocksPerGrid,threadsPerBlock>>>(uchar_d, src_d, w, h);

    cufftPlan2d(&plan, h, w, CUFFT_C2C);
    cufftExecC2C(plan, src_d, dst_d, CUFFT_FORWARD);

    blocksPerGrid   = (h + threadsPerBlock - 1) / threadsPerBlock;
    swapRows<<<blocksPerGrid,threadsPerBlock>>>(dst_d, w, h);

    blocksPerGrid   = (h*w + threadsPerBlock - 1) / threadsPerBlock;
    normalize<<<blocksPerGrid,threadsPerBlock>>>(dst_d, uchar_d, w, h);

    checkCudaErrors(cudaMemcpy(dst, uchar_d, h*w *sizeof(unsigned char), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    cudaFree(src_d);
    cudaFree(dst_d);
    cudaFree(uchar_d);
}
