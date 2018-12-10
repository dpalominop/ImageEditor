#include "fouriertransform.h"

FourierTransform::FourierTransform(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void FourierTransform::calcFFT()
{
    emit print_progress(0);
    emit print_message(QString("applying FFT..."));
    unsigned int w = srcImage->width();
    unsigned int h = srcImage->height();
    unsigned int sz = w*h;
    unsigned int nBytes = sizeof(unsigned char) * sz;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned char *src_h = (unsigned char *)malloc(nBytes);
    unsigned char *dst_h = (unsigned char *)malloc(nBytes);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            src_h[y*w + x] = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
        }

    emit print_progress(20);

    img2fft(src_h, dst_h, w, h);

    emit print_progress(70);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const int d_y = dst_h[x + w*y];
            QRgb pix = qRgb(d_y, d_y, d_y);
            dstImage->setPixel(x, y, pix);
        }

    emit print_progress(90);

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("applying FFT...finished"));

    free(src_h);
    free(dst_h);
    return;
}
