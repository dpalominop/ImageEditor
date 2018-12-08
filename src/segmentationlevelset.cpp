#include "segmentationlevelset.h"

segmentationLevelSet::segmentationLevelSet(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    threshold = 10;
    limit_thickness = 2;
    limit_steps = 2000;
    return;
}

void segmentationLevelSet::apply_levelSetSegmentation()
{
    const int w0 = srcImage->width();
    const int h0 = srcImage->height();
    const int sz0 = h0*w0;
    segmentsCount = 0;

    segmentImage.resize_without_copy(h0, w0);
    memset(segmentImage.data, 0, sz0*sizeof(int));
    currentImage.resize_without_copy(h0, w0);
    incrementMap.resize_without_copy(h0, w0);
    *dstImage = QImage(w0, h0, QImage::Format_RGB32);
    for(int y = 0; y < h0; ++y)
        for(int x = 0; x < w0; ++x)
        {
            dstImage->setPixel(x, y, qRgb(0,0,0));
            QRgb pix = srcImage->pixel(x, y);
            const int luma32 = (int)(0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix));
            currentImage(y, x) = (unsigned char)(luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32);
            incrementMap(y, x) = {0, 0};
            segmentImage(y, x) = 0;
        }

    current_thickness = 0;
    while (updateUpperImageSize() > 0)
    {
        ++current_thickness;
        std::cout << "step [" << current_thickness << "]" <<std::endl;
        const int w = currentImage.width;
        const int h = currentImage.height;

        for(int y = 0; y < h; y+=2)
            for(int x = 0; x < w; x+=2)
            {
                // decides which pixels is root for a new segment
                // and constructs above pixel from others
                // then updates segments numbers
                solveQuad(x, y);
            }

        if (current_thickness == limit_steps)
            break;

        currentImage = std::move(upperImage);
    }
    drawSegments();
    emit image_ready();
}

void segmentationLevelSet::drawSegments()
{
    const int w = segmentImage.width;
    const int h = segmentImage.height;

    struct segment_color
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };
    std::map<int, segment_color> segments_colors;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            const int xl = x > 0 ? x - 1 : 0;
            const int ya = y > 0 ? y - 1 : 0;
            const int left = segmentImage(y, xl);
            const int above = segmentImage(ya, x);
            segment_color sg;
            const p2i inc = incrementMap(y,x);
            const int sega = above + inc.y;
            const int segl = left + inc.x;
            const int seg = (sega + segl)/2;
            segmentImage(y, x) = seg;

            if (segments_colors.find(seg) == segments_colors.end())
            {
                sg.r = qrand()%256;
                sg.g = qrand()%256;
                sg.b = qrand()%256;
                segments_colors[seg] = sg;
            }
            else
                sg = segments_colors[seg];
            dstImage->setPixel(x, y, qRgb(sg.r, sg.g, sg.b));

        }
    }

    return;
}

void segmentationLevelSet::solveQuad(const int x, const int y)
{
    const int w = currentImage.width;
    const int h = currentImage.height;
    const int w0 = segmentImage.width;
    const int h0 = segmentImage.height;

    int A[4];
    const int x1 = x < (currentImage.width - 1) ? x + 1 : x;
    const int y1 = y < (currentImage.height - 1) ? y + 1 : y;
    A[0] = currentImage(y, x);
    A[1] = currentImage(y, x1);
    A[2] = currentImage(y1, x);
    A[3] = currentImage(y1, x1);

    p2i B[4];
    B[0].x = 0;
    int val = A[1] - A[0];
    B[1].x = val > threshold ? +1 : 0;
    B[1].x = val < -threshold ? -1 : B[1].x;
    B[2].x = 0;
    val = A[3] - A[2];
    B[3].x = val > threshold ? +1 : 0;
    B[3].x = val < -threshold ? -1 : B[3].x;
    B[0].y = 0;
    B[1].y = 0;
    val = A[2] - A[0];
    B[2].y = val > threshold ? +1 : 0;
    B[2].y = val < -threshold ? -1 : B[2].y;
    val = A[3] - A[1];
    B[3].y = val > threshold ? +1 : 0;
    B[3].y = val < -threshold ? -1 : B[3].y;

    /*if ((B[0].x!=0)||(B[1].x!=0)||(B[2].x!=0)||(B[3].x!=0)||
        (B[0].y!=0)||(B[1].y!=0)||(B[2].y!=0)||(B[3].y!=0))
    {
        std::cout << x*w0/w << ";" << y*h0/h << " <<>> ";
        std::cout << x << ";" << y << std::endl;
    }*/

    int x0, y0;
    //x0 = x*w0/w;
    //y0 = y*h0/h;
    //incrementMap(y0, x0).x += B[0].x;
    //incrementMap(y0, x0).y += B[0].y; //zeros
    x0 = x1*w0/w;
    y0 = y*h0/h;
    incrementMap(y0, x0).x += B[1].x;
    //incrementMap(y0, x0).y += B[1].y;
    x0 = x*w0/w;
    y0 = y1*h0/h;
    //incrementMap(y0, x0).x += B[2].x;
    incrementMap(y0, x0).y += B[2].y;
    x0 = x1*w0/w;
    y0 = y1*h0/h;
    incrementMap(y0, x0).x += B[3].x;
    incrementMap(y0, x0).y += B[3].y;

    upperImage(y/2, x/2) = A[0];

    return;
}

int segmentationLevelSet::updateUpperImageSize()
{
    int w = currentImage.width;
    int h = currentImage.height;

    if ((w <= 1)||(h <= 1))
        return 0;

    w = (w % 2) ? w/2 + 1 : w/2;
    h = (h % 2) ? h/2 + 1 : h/2;

    upperImage.resize_without_copy(h, w);
    return 1;
}

void segmentationLevelSet::update_threshold(const QString& thr)
{
    bool ok;
    threshold = thr.toUInt(&ok);
    if (!ok)
        std::cout << "error reading threshold" << std::endl;
    else
        std::cout << "threshold updated to " << threshold << std::endl;
    return;
}

void segmentationLevelSet::update_thicknessLimit(const QString& tl)
{
    bool ok;
    limit_thickness = tl.toUInt(&ok);
    if (!ok)
        std::cout << "error reading limit_thickness" << std::endl;
    else
        std::cout << "limit_thickness updated to " << limit_thickness << std::endl;
    return;
}

void segmentationLevelSet::update_stepsLimit(const QString& sl)
{
    bool ok;
    limit_steps = sl.toUInt(&ok);
    if (!ok)
        std::cout << "error reading limit_steps" << std::endl;
    else
        std::cout << "limit_steps updated to " << limit_steps << std::endl;
    return;
}



