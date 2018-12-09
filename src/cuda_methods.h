#ifndef CUDA_METHODS_H
#define CUDA_METHODS_H

extern "C" void rgb2yuv(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n);
extern "C" void img2fogeffect(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n);
extern "C" void img2gradientborder(const int* src, const int* dst, int w, int h);
extern "C" void img2fft(int *src,int *dst, int w, int h);

#endif // CUDA_METHODS_H
