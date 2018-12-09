#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

/*typedef unsigned int TColor;

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS   3
#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3
#define KNN_WINDOW_AREA     ( (2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1) )
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define INV_KNN_WINDOW_AREA ( 1.0f / (float)KNN_WINDOW_AREA )
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

#define KNN_WEIGHT_THRESHOLD    0.02f
#define KNN_LERP_THRESHOLD      0.79f
#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;

float Max(float x, float y)
{
    return (x > y) ? x : y;
}

float Min(float x, float y)
{
    return (x < y) ? x : y;
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c)
{
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b)
{
    return (
               (b.x - a.x) * (b.x - a.x) +
               (b.y - a.y) * (b.y - a.y) +
               (b.z - a.z) * (b.z - a.z)
           );
}

__device__ TColor make_color(float r, float g, float b, float a)
{
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__global__ void Copy(TColor *dst, int imageW, int imageH)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        float4 fresult = tex2D(texImage, x, y);
        dst[imageW * iy + ix] = make_color(fresult.x, fresult.y, fresult.z, 0);
    }
}

cudaError_t CUDA_MallocArray(uchar4 **h_Src, uchar4 **d_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray(&d_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpyToArray(*d_Src, 0, 0, *h_Src, imageW * imageH * sizeof(uchar4), cudaMemcpyHostToDevice);

    return error;
}

cudaError_t CUDA_FreeArray(cudaArray *d_Src)
{
    return cudaFreeArray(d_Src);
}

cuda_Copy(TColor *d_dst, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    Copy<<<grid, threads>>>(d_dst, imageW, imageH);
}


////////////////////////////////////////////////////////////////////////////////
// KNN kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void KNN(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        //Center of the KNN window
        float4 clr00 = tex2D(texImage, x, y);

        //Cycle through KNN window, surrounding (x, y) texel
        for (float i = -KNN_WINDOW_RADIUS; i <= KNN_WINDOW_RADIUS; i++)
            for (float j = -KNN_WINDOW_RADIUS; j <= KNN_WINDOW_RADIUS; j++)
            {
                float4     clrIJ = tex2D(texImage, x + j, y + i);
                float distanceIJ = vecLen(clr00, clrIJ);

                //Derive final weight from color distance
                float   weightIJ = __expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA));

                //Accumulate (x + j, y + i) texel color with computed weight
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        clr.x = lerpf(clr.x, clr00.x, lerpQ);
        clr.y = lerpf(clr.y, clr00.y, lerpQ);
        clr.z = lerpf(clr.z, clr00.z, lerpQ);
        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    };
}

void cuda_KNN(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    KNN<<<grid, threads>>>(d_dst, imageW, imageH, Noise, lerpC);
}

void runImageFilters(TColor *d_dst)
{
    switch (g_Kernel)
    {
        case 0:
            cuda_Copy(d_dst, imageW, imageH);
            break;

        case 1:
            cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC);
            break;
    }

    getLastCudaError("Filtering kernel execution failed.\n");
}

void runAutoTest(int argc, char **argv, const char *filename, int kernel_param)
{
//    printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

    int devID = findCudaDevice(argc, (const char **)argv);

    // First load the image, so we know what the size of the image (imageW and imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

    if (image_path == NULL)
    {
        printf("imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
        exit(EXIT_FAILURE);
    }

    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    TColor *d_dst = NULL;
    unsigned char *h_dst = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_dst, imageW*imageH*sizeof(TColor)));
    h_dst = (unsigned char *)malloc(imageH*imageW*4);

    {
        g_Kernel = kernel_param;
        printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);
        checkCudaErrors(CUDA_Bind2TextureArray());
        runImageFilters(d_dst);
        checkCudaErrors(CUDA_UnbindTexture());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost));
        sdkSavePPM4ub(filename, h_dst, imageW, imageH);
    }

    checkCudaErrors(CUDA_FreeArray());
    free(h_Src);

    checkCudaErrors(cudaFree(d_dst));
    free(h_dst);

    printf("\n[%s] -> Kernel %d, Saved: %s\n", sSDKsample, kernel_param, filename);

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
*/
