#include "fogeffect.h"

/*static float gaussian[25] = {
0.037823,	0.039428,	0.039978,	0.039428,	0.037823,
0.039428,	0.041102,	0.041675,	0.041102,	0.039428,
0.039978,	0.041675,	0.042257,	0.041675,	0.039978,
0.039428,	0.041102,	0.041675,	0.041102,	0.039428,
0.037823,	0.039428,	0.039978,	0.039428,	0.037823
};*/

static float gaussian[11*11]{
    0.005374,	0.006088,	0.006708,	0.00719,	0.007495,	0.007599,	0.007495,	0.00719,	0.006708,	0.006088,	0.005374,
    0.006088,	0.006897,	0.007599,	0.008145,	0.00849,	0.008609,	0.00849,	0.008145,	0.007599,	0.006897,	0.006088,
    0.006708,	0.007599,	0.008374,	0.008974,	0.009355,	0.009486,	0.009355,	0.008974,	0.008374,	0.007599,	0.006708,
    0.00719,	0.008145,	0.008974,	0.009618,	0.010026,	0.010166,	0.010026,	0.009618,	0.008974,	0.008145,	0.00719,
    0.007495,	0.00849,	0.009355,	0.010026,	0.010452,	0.010598,	0.010452,	0.010026,	0.009355,	0.00849,	0.007495,
    0.007599,	0.008609,	0.009486,	0.010166,	0.010598,	0.010746,	0.010598,	0.010166,	0.009486,	0.008609,	0.007599,
    0.007495,	0.00849,	0.009355,	0.010026,	0.010452,	0.010598,	0.010452,	0.010026,	0.009355,	0.00849,	0.007495,
    0.00719,	0.008145,	0.008974,	0.009618,	0.010026,	0.010166,	0.010026,	0.009618,	0.008974,	0.008145,	0.00719,
    0.006708,	0.007599,	0.008374,	0.008974,	0.009355,	0.009486,	0.009355,	0.008974,	0.008374,	0.007599,	0.006708,
    0.006088,	0.006897,	0.007599,	0.008145,	0.00849,	0.008609,	0.00849,	0.008145,	0.007599,	0.006897,	0.006088,
    0.005374,	0.006088,	0.006708,	0.00719,	0.007495,	0.007599,	0.007495,	0.00719,	0.006708,	0.006088,	0.005374
};

FogEffect::FogEffect(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void FogEffect::calcFogEffect()
{
    emit print_progress(0);
    emit print_message(QString("applying Fog Effect..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned char *src_r, *src_g, *src_b;
    unsigned char *dst_r, *dst_g, *dst_b;

    src_r = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    src_g = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    src_b = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    dst_r = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    dst_g = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    dst_b = (unsigned char *)malloc(sizeof(unsigned char)*w*h);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);

            src_r[x + w*y] = qRed(pix);
            src_g[x + w*y] = qGreen(pix);
            src_b[x + w*y] = qBlue(pix);
        }

    setFogKernel(gaussian);
    fogFilter(src_r, dst_r, w, h, w, 1.0f);
    fogFilter(src_g, dst_g, w, h, w, 1.0f);
    fogFilter(src_b, dst_b, w, h, w, 1.0f);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = qRgb(0.7*dst_r[x + w*y]+0.3*src_r[x + w*y], 0.7*dst_g[x + w*y]+0.3*src_g[x + w*y], 0.7*dst_b[x + w*y]+0.3*src_b[x + w*y]);
//            QRgb pix = qRgb(dst_r[x + w*y], dst_g[x + w*y], dst_b[x + w*y]);
            dstImage->setPixel(x, y, pix);
        }

    free(src_r);
    free(src_g);
    free(src_b);
    free(dst_r);
    free(dst_g);
    free(dst_b);

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying Fog Effect...finished"));

    return;
}
