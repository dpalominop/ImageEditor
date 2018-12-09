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
    emit print_message(QString("applying convertion to YUV..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned int hist[256];
    unsigned int histDistribution[256];
    memset(hist, 0, 1024);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;

            // count intensity distribution
            ++hist[luma8];
        }

    // build a cumulative histogram as LUT
    unsigned int sum = 0;
    for (int i = 0; i < 256; ++i)
    {
        sum += hist[i];
        histDistribution[i] = sum;
    }

    // transform image using sum histogram as a LUT (Look Up Table)
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;

            const unsigned char level = (unsigned char)(histDistribution[luma8]*255/sz);

            pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying convertion to YUV...finished"));

    return;
}
