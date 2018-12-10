#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cufft.h>
#include <stdlib.h>

void img2fft(unsigned char *src, unsigned char *dst, int w, int h){
    cufftHandle plan;
    cufftComplex *src_d, *src_h;
    cufftComplex *dst_d, *dst_h;

    src_h = (cufftComplex *) malloc(sizeof(cufftComplex)*w*h);
    dst_h = (cufftComplex *) malloc(sizeof(cufftComplex)*w*h);

    for(int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            src_h[y*w +x].x = (float)(src[y*w +x]);
            src_h[y*w +x].y = 0;
        }
    }

    checkCudaErrors(cudaMalloc((void**)&src_d, sizeof(cufftComplex)*w*h));
    checkCudaErrors(cudaMalloc((void**)&dst_d, sizeof(cufftComplex)*w*h));
    checkCudaErrors(cudaMemcpy(src_d, src_h, h*w *sizeof(cufftComplex), cudaMemcpyHostToDevice));

    cufftPlan2d(&plan, h, w, CUFFT_C2C);
    cufftExecC2C(plan, src_d, dst_d, CUFFT_FORWARD);

    checkCudaErrors(cudaMemcpy(dst_h, dst_d, h*w *sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    for(int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            const int luma32 = sqrt((dst_h[y*w +x].x)*(dst_h[y*w +x].x)+(dst_h[y*w +x].y)*(dst_h[y*w +x].y))/(1.0f*w);
            dst[y*w +x] = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
        }
    }

    unsigned char *tmp = (unsigned char *)malloc(sizeof(unsigned char)*w/2);

    for(int i=0; i<h/2; i++){
        unsigned char* i_start = dst+i*w;
        unsigned char* i_end = dst + (i+h/2)*w + w/2;
        memcpy(tmp, i_start, sizeof(unsigned char)*w/2);
        memcpy(i_start, i_end, sizeof(unsigned char)*w/2);
        memcpy(i_end, tmp, sizeof(unsigned char)*w/2);
    }

    for(int i=0; i<h/2; i++){
        unsigned char* i_start = dst+i*w +w/2;
        unsigned char* i_end = dst + (i+h/2)*w;
        memcpy(tmp, i_start, sizeof(unsigned char)*w/2);
        memcpy(i_start, i_end, sizeof(unsigned char)*w/2);
        memcpy(i_end, tmp, sizeof(unsigned char)*w/2);
    }

    cufftDestroy(plan);
    cudaFree(src_d);
    cudaFree(dst_d);
    free(src_h);
    free(dst_h);
    free(tmp);
}
