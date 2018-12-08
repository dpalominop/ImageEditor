#include "gradients.h"

static const signed int Gxm[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

static const signed int Gym[3][3] = {
    {-1,-2,-1},
    { 0, 0, 0},
    { 1, 2, 1},
};

gradients::gradients(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    gradient_minimum_range = 31.0f;
    gradius = 8;
    mu0 = 0.8f;
    iter0 = 1;
    return;
}

void gradients::apply_gradient()
{
    emit print_progress(0);
    emit print_message(QString("applying gradient..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> img(h, w);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            img(y, x) = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
        }

    emit print_progress(20);

    rawArray2D<float> temp(h, w);

    // applying sobel mask
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int dx[3], dy[3];
            dx[0] = -(x>0);
            dx[1] = 0;
            dx[2] = (x<(w-1));
            dy[0] = -(y>0);
            dy[1] = 0;
            dy[2] = (y<(h-1));
            int Gx = 0, Gy = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    Gx += img(y+dy[i], x+dx[j]) * Gxm[i][j];
                    Gy += img(y+dy[i], x+dx[j]) * Gym[i][j];
                }
            temp(y, x) = sqrt(Gx*Gx + Gy*Gy);
        }

    emit print_progress(60);

    float min = temp[0];
    float max = min;
    for(int i = 1; i < sz; ++i)
    {
        const float val = temp[i];
        max = val > max ? val : max;
        min = val < min ? val : min;
    }

    emit print_progress(70);

    float range = max - min;
    range = range < 0.01 ? 0.01 : range;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const float val = temp(y, x);
            const unsigned char level = (unsigned char)(255*(val - min)/range);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit print_progress(90);

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("applying gradient...finished"));

    return;
}

void gradients::apply_nGradient()
{
    emit print_progress(0);
    emit print_message(QString("applying locally normalized gradient..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> img(h, w);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            img(y, x) = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
        }
    rawArray2D<float> temp(h, w);

    // applying sobel mask
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int dx[3], dy[3];
            dx[0] = -(x>0);
            dx[1] = 0;
            dx[2] = (x<(w-1));
            dy[0] = -(y>0);
            dy[1] = 0;
            dy[2] = (y<(h-1));
            int Gx = 0, Gy = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    Gx += img(y+dy[i], x+dx[j]) * Gxm[i][j];
                    Gy += img(y+dy[i], x+dx[j]) * Gym[i][j];
                }
            temp(y, x) = sqrt(Gx*Gx + Gy*Gy);
        }
    emit print_progress(4);

    int percent = 0;
    int prev_percent = percent;

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const int x1 = (x >= gradius) ? x - gradius : 0;
            const int y1 = (y >= gradius) ? y - gradius : 0;
            const int x2 = (x < (w - gradius)) ? x + gradius : w - 1;
            const int y2 = (y < (h - gradius)) ? y + gradius : h - 1;

            float buf = temp(y, x);
            float min = buf;
            float max = min;
            for (int yy = y1; yy <= y2; ++yy)
                for (int xx = x1; xx <= x2; ++xx)
                {
                    const float val = temp(yy, xx);
                    min = val < min ? val : min;
                    max = val > max ? val : max;
                }

            float range = max - min;
            range = range < gradient_minimum_range ? gradient_minimum_range : range;

            buf = (buf - min)/range;
            const unsigned char level = (unsigned char)(255*buf);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }
        percent = (y*32 + y*64)/h;
        if (percent != prev_percent)
        {
            emit print_progress(4 + percent);
            prev_percent = percent;
        }
    }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("applying locally normalized gradient...finished"));

    return;
}

void gradients::get_UV(const rawArray2D<unsigned char> & input,
                   rawArray2D<float> *const __restrict Ugvf,
                   rawArray2D<float> *const __restrict Vgvf)
{
    const int w = input.width;
    const int h = input.height;

    const float mu4 = mu0*4;
    int ITER = iter0;

    int percent = 0;
    int prev_percent = percent;

    struct gvfw {
        float u;
        float v;
        float f;
        float Lu;
        float Lv;
        float c1;
        float c2;
    };
    rawArray2D<gvfw> tmp(h, w);

    float pixmax = input[0];
    float pixmin = input[0];
    const int sz = h*w;
    for (int i = 1;  i < sz; ++i)
    {
        const float pix = input[i];
        pixmax = pix > pixmax ? pix : pixmax;
        pixmin = pix < pixmin ? pix : pixmin;
    }

    for (int i = 0; i < sz; i++)
    {
        float buf = input[i];
        buf = (buf-pixmin)/(pixmax-pixmin);
        tmp[i].f = buf;
    }

    const int ihw = sz - 1;
    const int ih0 = sz - w;

    tmp[0].u = 0;     tmp[w - 1].u = 0;
    tmp[ih0].u = 0;   tmp[ihw].u = 0;

    tmp[0].v = 0;     tmp[w - 1].v = 0;
    tmp[ih0].v = 0;   tmp[ihw].v = 0;

    for(int x = 1; x < (w - 1); ++x)
    {
        const int ihx = sz - w + x;
        tmp[x].v = 0.0f;
        tmp[ihx].v = 0.0f;
        float l = tmp[ihx - 1].f;
        float r = tmp[ihx + 1].f;
        float buf = 0.5f*(r - l);
        tmp[x].u = buf;
        buf = 0.5f*(tmp[ihx + 1].f - tmp[ihx - 1].f);
        tmp[ihx - 1].u = buf;
    }
    int iy0 = 0;
    for(int y = 1; y < (h - 1); ++y)
    {
        iy0 += w;
        tmp[iy0].v = 0.0f;
        tmp[iy0 + w - 1].v = 0.0f;
        float buf = 0.5f*(tmp[iy0 + w].f - tmp[iy0 - w].f);
        tmp[iy0].u = buf;
        buf = 0.5f*(tmp[iy0 + 2*w - 1].f - tmp[iy0 - 1].f);
        tmp[iy0 + w - 1].u = buf;
    }
    iy0 = 0;
    for(int y = 1; y < (h - 1); ++y)
    {
        iy0 += w;
        for(int x = 1; x < (w - 1); ++x)
        {
            float buf = 0.5*(tmp[iy0 + x + 1].f - tmp[iy0 + x - 1].f);
            tmp[iy0 + x].u = buf;
            buf = 0.5*(tmp[iy0 + w + x].f - tmp[iy0 - w + x].f);
            tmp[iy0 + x].v = buf;
        }
    }
    iy0 = 0;
    for(int y = 1; y < (h - 1); ++y)
    {
        iy0 += w;
        for(int x = 1; x < (w - 1); ++x)
        {
            const int iyx = iy0 + x;
            float buf = tmp[iyx - w - 1].f * Gxm[0][0] +
                        tmp[iyx - w].f * Gxm[0][1] +
                        tmp[iyx - w - 1].f * Gxm[0][2] +
                        tmp[iyx - 1].f * Gxm[1][0] +
                        tmp[iyx].f * Gxm[1][1] +
                        tmp[iyx + 1].f * Gxm[1][2] +
                        tmp[iyx + w - 1].f * Gxm[2][0] +
                        tmp[iyx + w].f * Gxm[2][1] +
                        tmp[iyx + 2 + 1].f * Gxm[2][2];
            tmp[iyx].u = buf;
            buf = tmp[iyx - w - 1].f * Gym[0][0] +
                  tmp[iyx - w].f * Gym[0][1] +
                  tmp[iyx - w + 1].f * Gym[0][2] +
                  tmp[iyx - 1].f * Gym[1][0] +
                  tmp[iyx].f * Gym[1][1] +
                  tmp[iyx + 1].f * Gym[1][2] +
                  tmp[iyx + w - 1].f * Gym[2][0] +
                  tmp[iyx + w].f * Gym[2][1] +
                  tmp[iyx + w + 1].f * Gym[2][2];
            tmp[iyx].v = buf;
        }
    }

    iy0 = 0;
    for(int y = 1; y < (h - 1); ++y)
    {
        iy0 += w;
        for(int x = 1; x < (w - 1); ++x)
        {
            const int iyx = iy0 + x;
            const float tempx = tmp[iyx].u;
            const float tempy = tmp[iyx].v;
            const float mag2 = tempx*tempx + tempy*tempy;
            tmp[iyx].f = (1 - mag2);
            tmp[iyx].c1 = mag2 * tempx;
            tmp[iyx].c2 = mag2 * tempy;
        }
    }

    emit print_progress(4);

    for (int count=1; count <= ITER; count++)
    {
        tmp[0].Lu = (tmp[1].u + tmp[w].u)*0.5 - tmp[0].u;
        tmp[0].Lv = (tmp[1].v + tmp[w].v)*0.5 - tmp[0].v;
        tmp[w - 1].Lu = (tmp[w - 2].u + tmp[w*2 - 1].u)*0.5 - tmp[w - 1].u;
        tmp[w - 1].Lv = (tmp[w - 2].v + tmp[w*2 - 1].v)*0.5 - tmp[w - 1].v;
        tmp[ih0].Lu = (tmp[ih0 - w + 1].u + tmp[ih0 - w*2].u)*0.5 - tmp[ih0 - w].u;
        tmp[ih0].Lv = (tmp[ih0 - w + 1].v + tmp[ih0 - w*2].v)*0.5 - tmp[ih0 - w].v;
        tmp[ih0 + w - 1].Lu = (tmp[ih0 - 2].u + tmp[ih0 - w - 1].u)*0.5 - tmp[ih0 - 1].u;
        tmp[ih0 + w - 1].Lv = (tmp[ih0 - 2].v + tmp[ih0 - w - 1].v)*0.5 - tmp[ih0 - 1].v;
        iy0 = 0;
        for (int y = 1; y <= (h - 2); y++)
        {
            iy0 += w;
            tmp[iy0].Lu = (2*tmp[iy0 + 1].u + tmp[iy0 - w].u + tmp[iy0 + w].u)*0.25 - tmp[iy0].u;
            tmp[iy0].Lv = (2*tmp[iy0 + 1].v + tmp[iy0 - w].v + tmp[iy0 + w].v)*0.25 - tmp[iy0].v;
            tmp[iy0 + w - 1].Lu = (2*tmp[iy0 + w - 2].u + tmp[iy0 - 1].u + tmp[iy0 + w*2 - 1].u)*0.25 - tmp[iy0 + w - 1].u;
            tmp[iy0 + w - 1].Lv = (2*tmp[iy0 + w - 2].v + tmp[iy0 - 1].v + tmp[iy0 + w*2 - 1].v)*0.25 - tmp[iy0 + w - 1].v;
        }
        for (int x = 1; x <= (w - 2); x++)
        {
            tmp[x].Lu = (2*tmp[w + x].u + tmp[x-1].u + tmp[x+1].u)*0.25 - tmp[x].u;
            tmp[x].Lv = (2*tmp[w + x].v + tmp[x-1].v + tmp[x+1].v)*0.25 - tmp[x].v;
            tmp[ih0 + x].Lu = (2*tmp[ih0 + x - w].u + tmp[ih0 + x - 1].u + tmp[ih0 + x + 1].u)*0.25 - tmp[ih0 + x].u;
            tmp[ih0 + x].Lv = (2*tmp[ih0 + x - w].v + tmp[ih0 + x - 1].v + tmp[ih0 + x + 1].v)*0.25 - tmp[ih0 + x].v;
        }
        iy0 = 0;
        for (int y = 1; y <= (h - 2); y++)
        {
            iy0 += w;
            for (int x = 1; x <= (w - 2); x++)
            {
                const int iyx = iy0 + x;
                tmp[iyx].Lu = (tmp[iyx - 1].u + tmp[iyx + 1].u + tmp[iyx - w].u + tmp[iyx + w].u)*0.25 - tmp[iyx].u;
                tmp[iyx].Lv = (tmp[iyx - 1].v + tmp[iyx + 1].v + tmp[iyx - w].v + tmp[iyx + w].v)*0.25 - tmp[iyx].v;
            }
        }
        iy0 = -w;
        for (int y = 0; y < h; y++)
        {
            iy0 += w;
            for (int x = 0; x < w; x++)
            {
                const int iyx = iy0 + x;
                tmp[iyx].u = tmp[iyx].f*tmp[iyx].u + mu4*tmp[iyx].Lu + tmp[iyx].c1;
                tmp[iyx].v = tmp[iyx].f*tmp[iyx].v + mu4*tmp[iyx].Lv + tmp[iyx].c2;
            }
        }
        float rmax = sqrt(tmp[0].v*tmp[0].v + tmp[0].u*tmp[0].u);
        for (int i = 1;  i < sz; ++i)
        {
            const float r = sqrt(tmp[i].v*tmp[i].v + tmp[i].u*tmp[i].u);
            rmax = r > rmax ? r : rmax;
        }
        rmax *= 1.01f;
        rmax = rmax < 0.01f ? 0.01f : rmax;
        for (int i = 0; i < sz; i++)
        {
            tmp[i].u /= rmax;
            tmp[i].v /= rmax;
        }//*/

        percent = (count*32 + count*64)/(ITER+1);
        if (percent != prev_percent)
        {
            emit print_progress(4 + percent);
            prev_percent = percent;
        }
    }

    Ugvf->resize_without_copy(h,w);
    Vgvf->resize_without_copy(h,w);
    iy0 = -w;
    for (int y = 0; y < h; y++)
    {
        iy0 += w;
        for (int x = 0; x < w; x++)
        {
            (*Ugvf)[iy0 + x] = tmp[iy0 + x].u;
            (*Vgvf)[iy0 + x] = tmp[iy0 + x].v;
        }
    }
}


void gradients::apply_GVF()
{
    emit print_progress(0);
    emit print_message(QString("applying gradient vector flow..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> img(h, w);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            img(y, x) = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
        }
    rawArray2D<float> Ux;
    rawArray2D<float> Vy;

    get_UV(img, &Ux, &Vy);

    float u = Ux[0];
    float v = Vy[0];
    float min = sqrt(u*u + v*v);
    float max = min;

    for(int i = 1; i < sz; ++i)
    {
        u = Ux[i];
        v = Vy[i];
        const float val = sqrt(u*u + v*v);
        max = val > max ? val : max;
        min = val < min ? val : min;
        Ux[i] = val;
    }

    float range = max - min;
    range = range < 0.01f ? 0.01f : range;
    const float irange = 1.0f/range;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const float val = (Ux(y, x) - min)*irange;
            const unsigned char level = (unsigned char)(255*val);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(0);
    emit print_message(QString("applying gradient vector flow...finished"));

    return;
}

void gradients::update_gradius(const QString& gr)
{
    bool ok;
    gradius = gr.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading gradius"));
    else
        emit print_message(QString("gradius updated to ") + QString::number(gradius));
    return;
}

void gradients::update_mu0(const QString& gr)
{
    bool ok;
    mu0 = gr.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading mu0"));
    else
        emit print_message(QString("mu0 updated to ") + QString::number(mu0));
    return;
}

void gradients::update_iter0(const QString& gr)
{
    bool ok;
    iter0 = gr.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading iter0"));
    else
        emit print_message(QString("iter0 updated to ") + QString::number(iter0));
    return;
}


