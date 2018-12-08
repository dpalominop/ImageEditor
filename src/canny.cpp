#include "canny.h"

// pi is 360 steps, not degrees
static const int _pi = 360;         // 180
static const int _2pi = _pi*2;      // 360
static const int _pi2 = _pi/2;      // 90
static const int _pi4 = _pi/4;      // 45
static const int _3pi4 = _pi2 + _pi4; // 135
static const int _pi8 = _pi/8;      // 22.5
static const int _3pi8 = _pi4 + _pi8; // 67.5
static const int _5pi8 = _pi2 + _pi8; // 112.5
static const int _7pi8 = _pi - _pi8;  // 157.5
#define LSZ 3
#define DIRS 10
#define RANGE 32

static const unsigned int Gaublu[5][5] = {
    {2,  4,  5,  4,  2},
    {4,  9, 12,  9,  4},
    {5, 12, 15, 12,  5},
    {4,  9, 12,  9,  4},
    {2,  4,  5,  4,  2},
};

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
//Gy inversed in comparison to wikipedia because here Y-axis points down

static float adapt(const rawArray2D<float> &gradient, const int y0, const int x0, float base)
{
    const int h = gradient.height;
    const int w = gradient.width;

    const int dx = 29;
    const int dy = 29;
    const int x1 = (x0 > dx) ? x0 - dx : 0;
    const int y1 = (y0 > dy) ? y0 - dy : 0;
    const int x2 = (x0 < (w - dx - 1)) ? x0 + dx : (w - 1);
    const int y2 = (y0 < (h - dy - 1)) ? y0 + dy : (h - 1);

    float sum = 0.0f;
    for (int y = y1; y <= y2; ++y)
        for (int x = x1; x <= x2; ++x)
        {
            sum += gradient(y, x);
        }
    sum /= (x2 - x1 + 1)*(y2 - y1 + 1);

    return base + sum;
}

canny::canny(QImage *const src, QImage *const dst, QObject * parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    lowerThreshold = -10.0f;
    upperThreshold = 10.0f;
    return;
}

void canny::apply_canny()
{
    emit print_progress(0);
    emit print_message(QString("applying canny..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int sz = w*h;
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int percent = 0;
    int prev_percent = percent;

    rawArray2D<unsigned char> img(h, w);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
            img(y, x) = luma8;
        }

    emit print_progress(1);

    rawArray2D<unsigned char> edgeDir(h, w);
    rawArray2D<float> gradient(h, w);
    rawArray2D<unsigned char> temp(h, w);

    // Gaussian blur
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int dx[5], dy[5];
            dx[1] = -(x > 0);
            dx[0] = (x > 1) ? -2 : dx[1];
            dx[2] = 0;
            dx[3] = (x < (w - 1));
            dx[4] = (x < (w - 2)) ? +2 : dx[3];
            dy[1] = -(y > 0);
            dy[0] = (y > 1) ? -2 : dy[1];
            dy[2] = 0;
            dy[3] = (y < (h - 1));
            dy[4] = (y < (h - 2)) ? +2 : dy[3];
            unsigned int pix = 0;
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    pix += img(y+dy[i], x+dx[i])*Gaublu[i][j];
            pix /= 159;
            temp(y, x) = (unsigned char)(pix);
        }

    emit print_progress(3);

    memset(img.data, 0, w*h);

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
                    Gx += temp(y+dy[i], x+dx[j]) * Gxm[i][j];
                    Gy += temp(y+dy[i], x+dx[j]) * Gym[i][j];
                }
            unsigned char dir = 4;

            const float angle0 = atan2(Gy,Gx);
            int angle = _pi*(angle0/3.1416); //<- normal angle (not tangencial)
            angle %= _2pi;
            angle += (angle < 0) ? _pi : 0;

            if ((angle <= _pi8)||(angle > _7pi8))
                dir = 0;
            else if ((angle > _pi8)&&(angle <= _3pi8))
                dir = 1;
            else if ((angle > _3pi8)&&(angle <= _5pi8))
                dir = 2;
            else if ((angle > _5pi8)&&(angle <= _7pi8))
                dir = 3;
            edgeDir(y, x) = dir;
            gradient(y, x) = sqrt(Gx*Gx + Gy*Gy);
        }

    emit print_progress(4);

    // processing edges
    //const unsigned char rowShift[4] = {0, 1, 1, 1};
    //const unsigned char colShift[4] = {1, 1, 0, -1};
    // NORMAL ANGLE :               0     45     90   135
    const signed char delta[4][2]={{0,1},{1,1},{1,0},{1,-1}}; //this is correct
    // delta is perpendicular to edge orientation here

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const unsigned char dir = edgeDir(y, x);
            const int dy = delta[dir][0];
            const int dx = delta[dir][1];
            // do not search for gradient local maximum over the edge
            const int yedge = (y==0)|(y==(h-1));
            const int xedge = (x==0)|(x==(w-1));
            const int skip = ((dy!=0)&yedge)|((dx!=0)&(xedge));
            temp(y, x) = 0;
            if (skip) continue;

            int py[LSZ], ny[LSZ], px[LSZ], nx[LSZ];
            py[0] = ny[0] = y;
            px[0] = nx[0] = x;
            for (int i = 1; i < LSZ; ++i)
            {
                py[i] = py[i - 1] - dy;
                ny[i] = ny[i - 1] + dy;
                px[i] = px[i - 1] - dx;
                nx[i] = nx[i - 1] + dx;
                py[i] = ((py[i] < 0)|(py[i] >= h)) ? py[i-1] : py[i];
                ny[i] = ((ny[i] < 0)|(ny[i] >= h)) ? ny[i-1] : ny[i];
                px[i] = ((px[i] < 0)|(px[i] >= w)) ? px[i-1] : px[i];
                nx[i] = ((nx[i] < 0)|(nx[i] >= w)) ? nx[i-1] : nx[i];
            }

            const int hit1 = (gradient(y, x) > gradient(py[1], px[1]));
            const int hit2 = (gradient(y, x) >= gradient(ny[1], nx[1]));
            int hit3 = 1;
            int hit4 = 1;
            for (int i = 2; i < LSZ; ++i)
            {
                hit3 &= gradient(y, x) > gradient(py[i], px[i]);
                hit4 &= gradient(y, x) > gradient(ny[i], nx[i]);
            }

            const int hit = hit1&hit2&hit3&hit4;

            temp(y, x) = hit ? (unsigned char)255 : (unsigned char)0;
        }

    emit print_progress(5);

    const signed char directions[4][2]={{1,0},{1,-1},{0,1},{1,1}}; //y, x order
    // directions of edges, not normals

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            const int thr = adapt(gradient, y, x, upperThreshold);
            if ((temp(y, x) > 0) && (gradient(y, x) > thr) && (img(y, x) == 0))
            {
                unsigned char dir = edgeDir(y, x);
                //first in one direction
                int l1 = 0, yn = y, xn = x;
                while (1)
                {
                    xn += directions[dir][0];
                    yn += directions[dir][1];
                    if ((xn < 0)|(yn < 0)|(xn > (w - 1))|(yn > (h - 1)))
                        break;
                    const float thrsh = adapt(gradient, yn, xn, lowerThreshold);
                    if ((temp(yn, xn) == 0)|(gradient(yn, xn) <= thrsh)|(img(yn, xn) == 255))
                        break;
                    ++l1;
                    dir = edgeDir(yn, xn);
                }
                // the other direction
                dir = edgeDir(y, x);
                xn = x;
                yn = y;
                int l2 = 0;
                while (1)
                {
                    xn -= directions[dir][0];
                    yn -= directions[dir][1];
                    if ((xn < 0)|(yn < 0)|(xn > (w - 1))|(yn > (h - 1)))
                        break;
                    const float thrsh = adapt(gradient, yn, xn, lowerThreshold);
                    if ((temp(yn, xn) == 0)|(gradient(yn, xn) <= thrsh)|(img(yn, xn) == 255))
                        break;
                    ++l2;
                    dir = edgeDir(yn, xn);
                }
                if ((l1+l2)>=0)
                {
                    img(y, x) = 255;
                    temp(y, x) = 0;
                    dir = edgeDir(y, x);
                    xn = x;  yn = y;
                    while (l1 > 0)
                    {
                        xn += directions[dir][0];
                        yn += directions[dir][1];
                        --l1;
                        dir = edgeDir(yn, xn);
                        img(yn, xn) = 255;
                        temp(yn, xn) = 0;
                    }
                    // the other direction
                    dir = edgeDir(y, x);
                    xn = x;  yn = y;
                    while (l2 > 0)
                    {
                        xn -= directions[dir][0];
                        yn -= directions[dir][1];
                        --l2;
                        dir = edgeDir(yn, xn);
                        img(yn, xn) = 255;
                        temp(yn, xn) = 0;
                    }
                    continue;
                }
            }
        }
        percent = y*90/h;
        if (percent != prev_percent)
        {
            emit print_progress(5+percent);
            prev_percent = percent;
        }
    }

    emit print_progress(95);

    for(int i = 0; i < sz; ++i)
        temp[i] = 0;

    emit print_progress(96);

    //const signed char directions[4][2]={{1,0},{1,-1},{0,1},{1,1}}; //y, x order
    const signed char extra_dir[4][4][2]=
    {
        {   {1, 0}, {2, 0}, {2, 1}, { 2,-1} },
        {   {1,-1}, {2,-2}, {1,-2}, { 2,-1} },
        {   {0, 1}, {0, 2}, {1, 2}, {-1, 2} },
        {   {1, 1}, {2, 2}, {2, 1}, { 1, 2} }
    };

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            if (img(y, x) == 255)
            {
                const int dir = edgeDir(y, x);
                for(int i = 0; i < 4; ++i)
                {
                    int yn = y + extra_dir[dir][i][0];
                    int xn = x + extra_dir[dir][i][1];
                    int fall = (xn < 0)|(xn >= w)|(yn < 0)|(yn >= h);
                    if (fall)
                        continue;
                    if (img(yn, xn) == 255)
                    {
                        temp(yn,xn) = 255;
                        temp(y,x) = 255;
                        temp(y+(yn - y)/2,x+(xn - x)/2) = 255;
                        break;
                    }
                    yn = y - extra_dir[dir][i][0];
                    xn = x - extra_dir[dir][i][1];
                    fall = (xn < 0)|(xn >= w)|(yn < 0)|(yn >= h);
                    if (fall)
                        continue;
                    if (img(yn, xn) == 255)
                    {
                        temp(yn,xn) = 255;
                        temp(y,x) = 255;
                        temp(y+(yn - y)/2,x+(xn - x)/2) = 255;
                        break;
                    }
                }
            }
        }

    emit print_progress(99);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level  = temp(y, x);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("applying canny...finished"));

    return;

    /*for(int i = 0; i < sz; ++i)
        temp[i] = 0;

    // connect dots
    const int diap = (90+DIRS/2)/DIRS;
    int dir_weights[DIRS];
    int lp[DIRS][2];
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            if ((*dst)(y,x) == 0)
                continue;
            const int y2 = y + ((((h - 1) - y) > RANGE) ? RANGE : (h - 1) - y);
            const int x2 = x + ((((w - 1) - x) > RANGE) ? RANGE : (w - 1) - x);
            for (int i = 0; i < DIRS; ++i)
            {
                lp[i][0] = y;
                lp[i][1] = x;
                dir_weights[i] = 0;
            }
            for (int yy = y; yy <= y2; ++yy)
                for (int xx = x; xx <= x2; ++xx)
                {
                    if ((*dst)(yy,xx) == 255)
                    {
                        const float ang_rad = ((xx - x) == 0) ? 3.1416/2.0 : atan((float)(yy- y)/(float)(xx - x));
                        const int ang_deg = (int)(ang_rad/3.1416*180);
                        const int i = (ang_deg)/diap;
                        dir_weights[i]+= 1;
                    }
                }
            //continue;
            int max_i1 = 0, max_i2 = 0;
            int max_w1 = dir_weights[0], max_w2 = dir_weights[0];
            int sum = 0;
            for (int i = 0; i < DIRS; ++i)
            {
                sum += dir_weights[i];
                if (max_w1 < dir_weights[i])
                {
                    max_w2 = max_w1;
                    max_i2 = max_i1;
                    max_w1 = dir_weights[i];
                    max_i1 = i;
                }
                else if (max_w2 < dir_weights[i])
                {
                    max_w2 = dir_weights[i];
                    max_i2 = i;
                }
            }
            for (int yy = y; yy <= y2; ++yy)
            {
                for (int xx = x; xx <= x2; ++xx)
                {
                    if ((*dst)(yy,xx) == 255)
                    {
                        const float ang_rad = ((xx - x) == 0) ? 3.1416/2.0 : atan((float)(yy- y)/(float)(xx - x));
                        const int ang_deg = (int)(ang_rad/3.1416*180);
                        const int i = (ang_deg)/diap;
                        if (((i == max_i1)|(i == max_i2))&(dir_weights[i]>(2*RANGE/3)))
                        {
                            int y0 = lp[i][0];
                            int x0 = lp[i][1];
                            temp(y, x) = 255;
                            temp(yy, xx) = 255;
                            y0 = y0 + (yy - y0)/2;
                            x0 = x0 + (xx - x0)/2;
                            temp(y0, x0) = 255;
                            lp[i][0] = yy;
                            lp[i][1] = xx;
                        }
                    }
                }
            }

        }

    for(int i = 0; i < sz; ++i)
        (*dst)[i] = temp[i];

    return;*/

}
#undef LSZ
#undef DIRS
#undef RANGE

void canny::updateUpperThreshold(const QString& thr)
{
    bool ok;
    upperThreshold = thr.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading upperThreshold"));
    else
        emit print_message(QString("upperThreshold updated to ") + QString::number(upperThreshold));
    return;
}

void canny::updateLowerThreshold(const QString& thr)
{
    bool ok;
    lowerThreshold = thr.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading lowerThreshold"));
    else
        emit print_message(QString("lowerThreshold updated to ") + QString::number(lowerThreshold));

    return;
}
