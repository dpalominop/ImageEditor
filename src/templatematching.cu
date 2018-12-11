#include "cuda_methods.h"

#include <device_launch_parameters.h>
// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_cuda.h>
#include <helper_functions.h>

typedef unsigned char uchar;


//#include <cuda_runtime.h>
//#include <stdio.h>
//#include "includes/kernel.cuh"
//#include "includes/utils.cuh"

#define L1Func(I, x, y) (I)
#define L2Func(I, x, y) (powf(I, 2))
#define LxFunc(I, x, y) (x * I)
#define LyFunc(I, x, y) (y * I)

#define RowCumSum(name, func)                                                \
  __global__ void name(const float *image, float *rowCumSum, int colNumberM, \
                       int rowNum) {                                         \
    float sum = 0;                                                           \
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;                      \
    if (xIndex >= rowNum) return;                                            \
    for (int i = 0; i < colNumberM; ++i) {                                   \
      sum += func(image[xIndex * colNumberM + i], i, xIndex);                \
      rowCumSum[xIndex * colNumberM + i] = sum;                              \
    }                                                                        \
  }

// Use macro to create kernels that compute L tables' rows
RowCumSum(calcL1RowCumSum, L1Func);
RowCumSum(calcL2RowCumSqrSum, L2Func);
RowCumSum(calcLxRowCumGradntSum, LxFunc);
RowCumSum(calcLyRowCumGradntSum, LyFunc);

// Sum up L tables by column
__global__ void calcSumTable(const float *rowCumSum, float *SumTable,
                             int rowNumberN, int colNumberM) {
  int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (xIndex >= colNumberM) return;
  for (int i = 1; i < rowNumberN; i++) {
    SumTable[i * colNumberM + xIndex] +=
        rowCumSum[(i - 1) * colNumberM + xIndex];
  }
}

// Helper function that computes S from certain L table
__device__ float computeS(float *sumTable, int rowNumberN, int colNumberM,
                          int startX, int startY, int Kx, int Ky) {
  startX--;
  startY--;
  float S =
      sumTable[startX + Kx + (Ky + startY) * colNumberM] -
      (startX < 0 ? 0 : sumTable[startX + (Ky + startY) * colNumberM]) -
      (startY < 0 ? 0 : sumTable[startX + Kx + startY * colNumberM]) +
      (startX < 0 || startY < 0 ? 0 : sumTable[startX + startY * colNumberM]);
  return S;
}

__global__ void calculateFeatureDifference(float *templateFeatures,
                                           int colNumberM, int rowNumberN,
                                           float *l1SumTable, float *l2SumTable,
                                           float *lxSumTable, float *lySumTable,
                                           int Kx, int Ky, float *differences) {
  int widthLimit = colNumberM - Kx + 1;
  int heightLimit = rowNumberN - Ky + 1;

  float meanVector;
  float varianceVector;
  float xGradientVector;
  float yGradientVector;
  int startX = threadIdx.x + blockIdx.x * blockDim.x;
  int startY = threadIdx.y + blockIdx.y * blockDim.y;
  if (startX >= widthLimit || startY >= heightLimit) return;
  float S1D =
      computeS(l1SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  float S2D =
      computeS(l2SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

  meanVector = S1D / (Kx * Ky);

  varianceVector = S2D / (Kx * Ky) - powf(meanVector, 2);

  float SxD =
      computeS(lxSumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

  xGradientVector = 4 * (SxD - (startX + Kx / 2.0) * S1D) / (Kx * Kx * Ky);

  float SyD =
      computeS(lySumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  yGradientVector = 4 * (SyD - (startY + Ky / 2.0) * S1D) / (Ky * Ky * Kx);

  differences[startX + startY * widthLimit] = norm4df(
      templateFeatures[0] - meanVector, templateFeatures[1] - varianceVector,
      templateFeatures[2] - xGradientVector,
      templateFeatures[3] - yGradientVector);
}

typedef struct SumTable_s {
  float* l1SumTable;
  float* l2SumTable;
  float* lxSumTable;
  float* lySumTable;
} SumTable;

inline int _ConvertSMVer2Cores(int major, int minor);

void AllocateCudaMem(float **pointer, int size) {
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void **)pointer, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void GetDeviceInfo(int *maxThreadsPerBlock, int *workingThreadsPerBlock) {
  int devid;
  cudaDeviceProp deviceProp;
  cudaGetDevice(&devid);
  cudaGetDeviceProperties(&deviceProp, devid);
  *maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  *workingThreadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
}

/*
 * Copy & modify from "helper_cuda.h" in the cuda samples, used to calculate the
 * number of cores per SM
 */
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
             // minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
      {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
      {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
      {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
      {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
      {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
      {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
      {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
      {0x70, 64},   // Volta Generation (SM 7.0) GV100 class

      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }
  return nGpuArchCoresPerSM[index - 1].Cores;
}

void preprocess(const float *I, const float *T, int M, int N, int Kx, int Ky,
                SumTable *sumTable, float *featuresT, int STableThread) {
  float *l1SumTable;
  float *l2SumTable;
  float *lxSumTable;
  float *lySumTable;

  AllocateCudaMem(&l1SumTable, sizeof(float) * M * N);
  AllocateCudaMem(&l2SumTable, sizeof(float) * M * N);
  AllocateCudaMem(&lxSumTable, sizeof(float) * M * N);
  AllocateCudaMem(&lySumTable, sizeof(float) * M * N);

  float *dev_I;

  AllocateCudaMem(&dev_I, sizeof(float) * M * N);

  cudaMemcpy(dev_I, I, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // Use streams to ensure the order
  cudaStream_t l1Stream, l2Stream, lxStream, lyStream;
  cudaStreamCreate(&l1Stream);
  cudaStreamCreate(&l2Stream);
  cudaStreamCreate(&lxStream);
  cudaStreamCreate(&lyStream);

  // calculate L tables first by row
  int rowBlocksize = (N + STableThread - 1) / STableThread;
  int sumTableBlocksize = (M + STableThread - 1) / STableThread;

  calcL1RowCumSum<<<rowBlocksize, STableThread, 0, l1Stream>>>(
      dev_I, l1SumTable, M, N);
  calcL2RowCumSqrSum<<<rowBlocksize, STableThread, 0, l2Stream>>>(
      dev_I, l2SumTable, M, N);
  calcLxRowCumGradntSum<<<rowBlocksize, STableThread, 0, lxStream>>>(
      dev_I, lxSumTable, M, N);
  calcLyRowCumGradntSum<<<rowBlocksize, STableThread, 0, lyStream>>>(
      dev_I, lySumTable, M, N);

  // then sum by column
  calcSumTable<<<sumTableBlocksize, STableThread, 0, l1Stream>>>(
      l1SumTable, l1SumTable, N, M);
  calcSumTable<<<sumTableBlocksize, STableThread, 0, l2Stream>>>(
      l2SumTable, l2SumTable, N, M);
  calcSumTable<<<sumTableBlocksize, STableThread, 0, lxStream>>>(
      lxSumTable, lxSumTable, N, M);
  calcSumTable<<<sumTableBlocksize, STableThread, 0, lyStream>>>(
      lySumTable, lySumTable, N, M);

  cudaStreamDestroy(l1Stream);
  cudaStreamDestroy(l2Stream);
  cudaStreamDestroy(lxStream);
  cudaStreamDestroy(lyStream);

  // Calculate features for the template
  for (int i = 0; i < Ky; i++) {
    for (int j = 0; j < Kx; j++) {
      featuresT[0] += T[i * Kx + j];
      featuresT[1] += T[i * Kx + j] * T[i * Kx + j];
      featuresT[2] += j * T[i * Kx + j];
      featuresT[3] += i * T[i * Kx + j];
    }
  }

  featuresT[0] /= (float)(Kx * Ky);
  featuresT[1] = featuresT[1] / (float)(Kx * Ky) - featuresT[0] * featuresT[0];
  //   4/Kx^2Ky*(Sx(D)-x*S1(D)), where x = Kx/2
  // = 4/Kx^2Ky*(f2-Kx/2*f0*Kx*Ky)
  // = 4/Kx^2Ky*f2-2*f0
  featuresT[2] = 4.0 / (Kx * Kx * Ky) * featuresT[2] - 2.0 * featuresT[0];
  featuresT[3] = 4.0 / (Ky * Kx * Ky) * featuresT[3] - 2.0 * featuresT[0];

  cudaDeviceSynchronize();
  sumTable->l1SumTable = l1SumTable;
  sumTable->l2SumTable = l2SumTable;
  sumTable->lxSumTable = lxSumTable;
  sumTable->lySumTable = lySumTable;
}

void getMinimum(float *target, int M, int N, int *x, int *y) {
  float minimum = *target;
  *x = 0;
  *y = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (target[i * M + j] < minimum) {
        minimum = target[i * M + j];
        *x = j;
        *y = i;
      }
    }
  }
}

void GetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x, int *y)
{
  int STableThread;
  int maxThreadsPerBlock;
  GetDeviceInfo(&maxThreadsPerBlock, &STableThread);

  SumTable sumTable;
  float featuresT[4] = {0, 0, 0, 0};
  preprocess(I, T, Iw, Ih, Tw, Th, &sumTable, featuresT, STableThread);
  float *dev_difference;
  float *difference;
  float *dev_featuresT;
  size_t difference_size = sizeof(float) * (Iw - Tw + 1) * (Ih - Th + 1);
  difference = (float *)malloc(difference_size);
  AllocateCudaMem(&dev_featuresT, sizeof(float) * 4);
  AllocateCudaMem(&dev_difference, difference_size);
  cudaMemcpy(dev_featuresT, featuresT, sizeof(float) * 4,
             cudaMemcpyHostToDevice);

  int differenceThreadsX = 32;
  int differenceThreadsY = maxThreadsPerBlock / differenceThreadsX;
  //                        Iw - Tw + 1 + differenceThreadsX - 1
  dim3 differenceBlockSize((Iw - Tw + differenceThreadsX) / differenceThreadsX,
                           // Ih - Th + 1 + differenceThreadsY - 1
                           (Ih - Th + differenceThreadsY) / differenceThreadsY);
  calculateFeatureDifference<<<differenceBlockSize,dim3(differenceThreadsX,differenceThreadsY)>>>(
                                                      dev_featuresT, Iw, Ih, sumTable.l1SumTable, sumTable.l2SumTable,
                                                      sumTable.lxSumTable, sumTable.lySumTable, Tw, Th, dev_difference);

  cudaMemcpy(difference, dev_difference, difference_size,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  getMinimum(difference, Iw - Tw + 1, Ih - Th + 1, x, y);
  cudaFree(sumTable.l1SumTable);
  cudaFree(sumTable.l2SumTable);
  cudaFree(sumTable.lxSumTable);
  cudaFree(sumTable.lySumTable);
  cudaFree(dev_difference);
  cudaFree(dev_featuresT);
  free(difference);
}
