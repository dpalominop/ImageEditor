#ifndef CUDA_METHODS_H
#define CUDA_METHODS_H

extern "C" void rgb2yuv(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n);
extern "C" void rgb2gray(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb, unsigned char *img_gray, int n);
extern "C" void rgb2binary(unsigned char *imgr, unsigned char *imgg, unsigned char *imgb, unsigned char *img_binary, int n, int umbral);

extern "C" void setFogKernel(float h_Kernel[]);
extern "C" void fogFilter(unsigned char* src_h, unsigned char* dst_h,
                                unsigned int width, unsigned int height,
                                unsigned int pitch, float scale);
extern "C" void img2gradientborder(const int* src, const int* dst, int w, int h);
extern "C" void img2fft(unsigned char *src, unsigned char *dst, int w, int h);
extern "C" void setSobelKernel(int hgx_Kernel[], int hgy_Kernel[]);
extern "C" void sobelFilter(unsigned char *src_h, unsigned char *dst_h,
                            unsigned int width, unsigned int height,
                            unsigned int pitch, float scale);

#endif // CUDA_METHODS_H
