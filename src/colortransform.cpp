#include "colortransform.h"

ColorTransform::ColorTransform(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void ColorTransform::convertToYUV()
{
    emit print_progress(0);
    emit print_message(QString("applying convertion to YUV..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int *src_r, *src_g, *src_b;
    int *dst_y, *dst_cb, *dst_cr;

    src_r = (int *)malloc(sizeof(int)*w*h);
    src_g = (int *)malloc(sizeof(int)*w*h);
    src_b = (int *)malloc(sizeof(int)*w*h);
    dst_y = (int *)malloc(sizeof(int)*w*h);
    dst_cb = (int *)malloc(sizeof(int)*w*h);
    dst_cr = (int *)malloc(sizeof(int)*w*h);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);

            src_r[x + w*y] = qRed(pix);
            src_g[x + w*y] = qGreen(pix);
            src_b[x + w*y] = qBlue(pix);
        }

    // convert rgb to yuv
    rgb2yuv(src_r, src_g, src_b, dst_y, dst_cb, dst_cr, w*h);

    // transform image using yuv vectors
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            const int d_y = dst_cr[x + w*y];

            pix = qRgb(d_y, d_y, d_y);
            dstImage->setPixel(x, y, pix);
        }

    free(src_r);
    free(src_g);
    free(src_b);
    free(dst_y);
    free(dst_cb);
    free(dst_cr);

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying convertion to YUV...finished"));

    return;
}
