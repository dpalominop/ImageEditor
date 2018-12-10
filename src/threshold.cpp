#include "threshold.h"

Threshold::Threshold(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void Threshold::convertToBinary()
{
    emit print_progress(0);
    emit print_message(QString("applying convertion to Binary..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int nBytes = w*h*sizeof(unsigned char);
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned char *src_r, *src_g, *src_b;
    unsigned char *dst_gray;

    src_r = (unsigned char *)malloc(nBytes);
    src_g = (unsigned char *)malloc(nBytes);
    src_b = (unsigned char *)malloc(nBytes);
    dst_gray = (unsigned char *)malloc(nBytes);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);

            src_r[x + w*y] = qRed(pix);
            src_g[x + w*y] = qGreen(pix);
            src_b[x + w*y] = qBlue(pix);
        }

    // convert rgb to yuv
    rgb2binary(src_r, src_g, src_b, dst_gray, w*h, umbral);

    // transform image using yuv vectors
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const int d_y = dst_gray[x + w*y];
            QRgb pix = qRgb(d_y, d_y, d_y);
            dstImage->setPixel(x, y, pix);
        }

    free(src_r);
    free(src_g);
    free(src_b);
    free(dst_gray);

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying convertion to Binary...finished"));

    return;
}

void Threshold::setUmbral(const QString& u)
{
    bool ok;
    umbral = u.toInt(&ok);
    if (!ok)
        emit print_message(QString("error reading umbral"));
    else
        emit print_message(QString("umbral updated to ") + QString::number(umbral));
    return;
}
