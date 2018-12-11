#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

typedef unsigned char uchar;
typedef unsigned int uint;

__global__ void imageScaledKernel(uchar *output, uchar *input, int pitchOutput, int pitchInput, int bytesPerPixelInput, int bytesPerPixelOutput, float xRatio, float yRatio){
        int x = (int) (xRatio * blockIdx.x);
        int y = (int) (yRatio * blockIdx.y);

        uchar *a, *b, *c, *d;
        float xDist, yDist, blue, red, green;

        // X and Y distance difference
        xDist = (xRatio * blockIdx.x) - x;
        yDist = (yRatio * blockIdx.y) - y;

        // Points
        a = input + y * pitchInput + x * bytesPerPixelInput;
        b = input + y * pitchInput + (x+1) * bytesPerPixelInput;
        c = input + (y+1) * pitchInput + x * bytesPerPixelInput;
        d = input + (y+1) * pitchInput + (x+1) * bytesPerPixelInput;

        // red
        red = (a[2])*(1 - xDist)*(1 - yDist) + (b[2])*(xDist)*(1 - yDist) + (c[2])*(yDist)*(1 - xDist) + (d[2])*(xDist * yDist);

        // green
        green = ((a[1]))*(1 - xDist)*(1 - yDist) + (b[1])*(xDist)*(1 - yDist) + (c[1])*(yDist)*(1 - xDist) + (d[1])*(xDist * yDist);

        // blue
        blue = (a[0])*(1 - xDist)*(1 - yDist) + (b[0])*(xDist)*(1 - yDist) + (c[0])*(yDist)*(1 - xDist) + (d[0])*(xDist * yDist);

        uchar* p = output + blockIdx.y * pitchOutput + blockIdx.x * bytesPerPixelOutput;
        *((uint *) p) = (0xff000000 | ((((int)red) << 16)) | ((((int)green) << 8)) | ((int)blue));
//        p[0] = blue;
//        p[1] = green;
//        p[2] = red;
//        p[3] = 0xff;
}

void imageScaled(uchar *src, int w, int h, int pitch, uchar *dst, int nw, int nh, int npitch, int BytesPerPixel)
{
    int imageByteLength = w * h * sizeof(uchar)*BytesPerPixel;

    // New width of image
    dim3 grid(nw, nh);

    // Create scaled image surface
    int newImageByteLength = nw * nh * sizeof(uchar)*BytesPerPixel;

    float xRatio = ((float)(w-1))/nw;
    float yRatio = ((float)(h-1))/nh;

    // Create pointer to device pixels
    uchar *pixels_dyn;

    // Copy original image
    checkCudaErrors(cudaMalloc((void **) &pixels_dyn, imageByteLength));
    checkCudaErrors(cudaMemcpy(pixels_dyn, src,  imageByteLength, cudaMemcpyHostToDevice));

    // Allocate new image on DEVICE
    uchar *newPixels_dyn;

    checkCudaErrors(cudaMalloc((void **) &newPixels_dyn, newImageByteLength));

    // Do the bilinear transform on CUDA device
    imageScaledKernel<<<grid,1>>>(newPixels_dyn, pixels_dyn, npitch, pitch, BytesPerPixel, BytesPerPixel, xRatio, yRatio);

    // Copy scaled image to host
    checkCudaErrors(cudaMemcpy(dst, newPixels_dyn, newImageByteLength, cudaMemcpyDeviceToHost));
//    newImage->pixels = newPixels;

    // Free memory
    cudaFree(pixels_dyn);
    cudaFree(newPixels_dyn);
}
