#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

typedef unsigned char uchar;

__global__ void addImageKernel(uchar *imgr, uchar *imgg, uchar *imgb,
                               uchar *imgr_k, uchar *imgg_k, uchar *imgb_k,
                               int w, int h, float index)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx < w && idy <h){
        imgr[idy*w + idx] = (1-index)*imgr[idy*w + idx] + index*imgr_k[idy*w + idx];
        imgg[idy*w + idx] = (1-index)*imgg[idy*w + idx] + index*imgg_k[idy*w + idx];
        imgb[idy*w + idx] = (1-index)*imgb[idy*w + idx] + index*imgb_k[idy*w + idx];
    }
}

void addImage(uchar *imgr, uchar *imgg, uchar *imgb,
              uchar *imgr_k, uchar *imgg_k, uchar *imgb_k,
              int w, int h, float index)
{
    uchar *d_imgr, *d_imgg, *d_imgb;
    uchar *d_imgr_k, *d_imgg_k, *d_imgb_k;

    checkCudaErrors(cudaMalloc((void **)&d_imgr, sizeof(uchar)*w*h));
    checkCudaErrors(cudaMalloc((void **)&d_imgg, sizeof(uchar)*w*h));
    checkCudaErrors(cudaMalloc((void **)&d_imgb, sizeof(uchar)*w*h));

    checkCudaErrors(cudaMalloc((void **)&d_imgr_k, sizeof(uchar)*w*h));
    checkCudaErrors(cudaMalloc((void **)&d_imgg_k, sizeof(uchar)*w*h));
    checkCudaErrors(cudaMalloc((void **)&d_imgb_k, sizeof(uchar)*w*h));

    checkCudaErrors(cudaMemcpy(d_imgr, imgr, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_imgg, imgg, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_imgb, imgb, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_imgr_k, imgr_k, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_imgg_k, imgg_k, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_imgb_k, imgb_k, sizeof(uchar)*w*h, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid((w + 7)/8, (h + 7)/8);
    addImageKernel<<<blocksPerGrid,threadsPerBlock>>>(d_imgr, d_imgg, d_imgb,
                                                      d_imgr_k, d_imgg_k, d_imgb_k,
                                                      w, h, index);

    checkCudaErrors(cudaMemcpy(imgr, d_imgr, sizeof(uchar)*w*h, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(imgg, d_imgg, sizeof(uchar)*w*h, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(imgb, d_imgb, sizeof(uchar)*w*h, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_imgr));
    checkCudaErrors(cudaFree(d_imgg));
    checkCudaErrors(cudaFree(d_imgb));

    checkCudaErrors(cudaFree(d_imgr_k));
    checkCudaErrors(cudaFree(d_imgg_k));
    checkCudaErrors(cudaFree(d_imgb_k));
}
