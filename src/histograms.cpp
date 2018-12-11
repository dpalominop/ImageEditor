#include "histograms.h"
#include "my_user_types.h"


histograms::histograms(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void histograms::apply_histogramEqualization()
{
    emit print_progress(0);
    emit print_message(QString("applying histogram equalization..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned int hist_r[256];
    unsigned int hist_g[256];
    unsigned int hist_b[256];
    unsigned int histDistribution_r[256];
    unsigned int histDistribution_g[256];
    unsigned int histDistribution_b[256];
    memset(hist_r, 0, 1024);
    memset(hist_g, 0, 1024);
    memset(hist_b, 0, 1024);

    unsigned char *img_r, *img_g, *img_b;
    img_r = (unsigned char *)malloc(sizeof(unsigned char)*sz);
    img_g = (unsigned char *)malloc(sizeof(unsigned char)*sz);
    img_b = (unsigned char *)malloc(sizeof(unsigned char)*sz);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            img_r[y*w+x] = qRed(pix);
            img_g[y*w+x] = qGreen(pix);
            img_b[y*w+x] = qBlue(pix);
        }

    histogram256(hist_r, img_r, h*w);
    histogram256(hist_g, img_g, h*w);
    histogram256(hist_b, img_b, h*w);

    // build a cumulative histogram as LUT
    unsigned int sum_r = 0;
    unsigned int sum_g = 0;
    unsigned int sum_b = 0;
    for (int i = 0; i < 256; ++i)
    {
        sum_r += hist_r[i];
        histDistribution_r[i] = sum_r;
        sum_g += hist_g[i];
        histDistribution_g[i] = sum_g;
        sum_b += hist_b[i];
        histDistribution_b[i] = sum_b;
    }

    // transform image using sum histogram as a LUT (Look Up Table)
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            //const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            //const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;

            const unsigned char level_r = (unsigned char)(histDistribution_r[qRed(pix)]*255/sz);
            const unsigned char level_g = (unsigned char)(histDistribution_g[qGreen(pix)]*255/sz);
            const unsigned char level_b = (unsigned char)(histDistribution_b[qBlue(pix)]*255/sz);

            pix = qRgb(level_r, level_g, level_b);
            dstImage->setPixel(x, y, pix);
        }

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying histogram equalization...finished"));

    return;
}

struct rect
{
    p2i v0;
    p2i v1;
};

void histograms::apply_histogramAdaptiveEqualization()
{
    emit print_message(QString("applying adaptive histogram equalization..."));
    emit print_progress(0);
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> inbuf(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {

            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            inbuf(y, x) = (unsigned char)(luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32);
        }

    unsigned int hist[256];
    unsigned int histDistribution[256];
    memset(hist, 0, 1024);

    int dx = 16;
    int dy = dx;

    int percent = 0;
    int prev_percent = percent;

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            memset(hist, 0, 1024);
            rect view;
            view.v0.y = y - dy;        view.v0.x = x - dx;
            view.v1.y = y + dy;        view.v1.x = x + dx;

            view.v0.y = view.v0.y < 0 ? 0 : view.v0.y;
            view.v0.x = view.v0.x < 0 ? 0 : view.v0.x;
            view.v1.y = view.v1.y > (h - 1) ? h - 1 : view.v1.y;
            view.v1.x = view.v1.x > (w - 1) ? w - 1 : view.v1.x;

            const int viewsize = (view.v1.y - view.v0.y + 1)*(view.v1.x - view.v0.x + 1);

            for (int py = view.v0.y; py <= view.v1.y; ++py)
            for (int px = view.v0.x; px <= view.v1.x; ++px)
            {
                ++hist[inbuf(py, px)];
            }

            // build a cumulative histogram as LUT
            unsigned int sum = 0;
            for (int i = 0; i < 256; ++i)
            {
                sum += hist[i];
                histDistribution[i] = sum;
            }

            // transform image using sum histogram as a LUT (Look Up Table)
            QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;

            const unsigned char level = (unsigned char)(histDistribution[luma8]*255/viewsize);

            pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }
        percent = y*20 / h;
        if (percent != prev_percent)
        {
            emit print_progress(percent*5);
            prev_percent = percent;
        }
    }

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying adaptive histogram equalization...finished"));

    return;
}
