#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void rgb2yuvKernel(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n) {

    int r, g, b;
    int y, cb, cr;

    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n){
        r = imgr[index];
        g = imgg[index];
        b = imgb[index];

        y  = (int)( 0.299*r + 0.587*g +  0.114*b);
        cb = (int)(-0.147*r - 0.289*g +  0.436*b);
        cr = (int)( 0.615*r - 0.515*g - 0.1*b);

        imgy[index] = y;
        imgcb[index] = cb;
        imgcr[index] = cr;
    }
}


__global__ void yuv2rgbKernel(int *imgy,int *imgcb,int *imgcr, int *imgr,int *imgg,int *imgb, int n) {

    int r, g, b;
    int y, cb, cr;

    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n){
        y = imgy[index];
        cb = imgcb[index];
        cr = imgcr[index];

        r  = (int)( 1*y + 0*cb +  1.14*cr);
        g = (int)( 1*y - 0.396*cb -  0.581*cr);
        b = (int)( 1*y + 2.029*cb +  0*cr);

        imgr[index] = r;
        imgg[index] = g;
        imgb[index] = b;
    }
}

void rgb2yuv(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n){
    int *imgr_cuda, *imgg_cuda, *imgb_cuda;
    int *imgy_cuda, *imgcb_cuda, *imgcr_cuda;

    unsigned int nBytes = sizeof(int) * n;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&imgr_cuda, nBytes);
    cudaMalloc((void **)&imgg_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgb_cuda, sizeof(int)*n);

    cudaMalloc((void **)&imgy_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgcb_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgcr_cuda, sizeof(int)*n);

    cudaMemcpy(imgr_cuda, imgr, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(imgg_cuda, imgg, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(imgb_cuda, imgb, sizeof(int)*n, cudaMemcpyHostToDevice);

    rgb2yuvKernel<<<blocksPerGrid,threadsPerBlock>>>(imgr_cuda, imgg_cuda, imgb_cuda, imgy_cuda, imgcb_cuda, imgcr_cuda, n);

    cudaMemcpy(imgy, imgy_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(imgcb, imgcb_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(imgcr, imgcr_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);

    cudaFree(imgr_cuda);
    cudaFree(imgg_cuda);
    cudaFree(imgb_cuda);
    cudaFree(imgy_cuda);
    cudaFree(imgcb_cuda);
    cudaFree(imgcr_cuda);
}

void yuv2rgb(int *imgy,int *imgcb,int *imgcr, int *imgr,int *imgg,int *imgb, int n){
    int *imgy_cuda, *imgcb_cuda, *imgcr_cuda;
    int *imgr_cuda, *imgg_cuda, *imgb_cuda;

    unsigned int nBytes = sizeof(int) * n;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&imgy_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgcb_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgcr_cuda, sizeof(int)*n);

    cudaMalloc((void **)&imgr_cuda, nBytes);
    cudaMalloc((void **)&imgg_cuda, sizeof(int)*n);
    cudaMalloc((void **)&imgb_cuda, sizeof(int)*n);

    cudaMemcpy(imgy_cuda, imgy, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(imgcb_cuda, imgcb, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(imgcr_cuda, imgcr, sizeof(int)*n, cudaMemcpyHostToDevice);

    yuv2rgbKernel<<<blocksPerGrid,threadsPerBlock>>>(imgy_cuda, imgcb_cuda, imgcr_cuda, imgr_cuda, imgg_cuda, imgb_cuda, n);

    cudaMemcpy(imgr, imgr_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(imgg, imgg_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(imgb, imgb_cuda, sizeof(int)*n, cudaMemcpyDeviceToHost);

    cudaFree(imgy_cuda);
    cudaFree(imgcb_cuda);
    cudaFree(imgcr_cuda);
    cudaFree(imgr_cuda);
    cudaFree(imgg_cuda);
    cudaFree(imgb_cuda);
}
