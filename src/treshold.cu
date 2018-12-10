#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void rgb2binaryKernel(unsigned char *imgr,unsigned char *imgg,unsigned char *imgb,unsigned char *img_binary, int n, int umbral) {

    int r, g, b;
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n){
        r = imgr[index];
        g = imgg[index];
        b = imgb[index];

        img_binary[index] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b)>umbral?255:0;
    }
}

void rgb2binary(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb, unsigned char *img_binary, int n, int umbral){
    unsigned char *imgr_cuda, *imgg_cuda, *imgb_cuda;
    unsigned char *img_binary_cuda;

    unsigned int nBytes = sizeof(unsigned char) * n;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&imgr_cuda, nBytes);
    cudaMalloc((void **)&imgg_cuda, nBytes);
    cudaMalloc((void **)&imgb_cuda, nBytes);
    cudaMalloc((void **)&img_binary_cuda, nBytes);

    cudaMemcpy(imgr_cuda, imgr, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(imgg_cuda, imgg, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(imgb_cuda, imgb, nBytes, cudaMemcpyHostToDevice);

    rgb2binaryKernel<<<blocksPerGrid,threadsPerBlock>>>(imgr_cuda, imgg_cuda, imgb_cuda, img_binary_cuda, n, umbral);

    cudaMemcpy(img_binary, img_binary_cuda, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(imgr_cuda);
    cudaFree(imgg_cuda);
    cudaFree(imgb_cuda);
    cudaFree(img_binary_cuda);
}
