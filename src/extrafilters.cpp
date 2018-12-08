#include "extrafilters.h"
#include "my_user_types.h"

extraFilters::extraFilters(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    min_length = 8;
    return;
}

static bool check(std::list<p2i> & line, int yy, int xx)
{
    for (auto it = line.begin(); it != line.end(); ++it)
        if ((it->y==yy)&(it->x==xx))
            return false;
    return true;
}

void extraFilters::apply_deleteShortLines()
{
    emit print_progress(0);
    emit print_message(QString("deleting short lines..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> img(h, w);

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 128 ? 255 : 0;
            img(y, x) = luma8;

            pix = qRgb(0, 0, 0);
            dstImage->setPixel(x, y, pix);
        }
    }
    std::list<p2i> line;

    int percent = 0;
    int prev_percent = percent;

    for (int y0 = 0; y0 < h; ++y0)
    {
        for (int x0 = 0; x0 < w; ++x0)
        {
            if (img(y0,x0) > 128)
            {
                int x = x0;
                int y = y0;
                line.clear();
                line.push_back({x,y});
                int len = 1, final_length = 1;
                // start collecting line
                int end, no_new_branch, level = 0;
                // this cycle wooud collect points lying close to each other starting from (y0;x0)
                // it will move this points to output while removing from input
                // when it remove all the points connected to (y0;x0) it will quit
                do
                {
                    end = 0;
                    while (!end)
                    {
                        end = 1;
                        const int x1 = (x > 0) ? x - 1 : 0;
                        const int x2 = (x < (w - 1)) ? x + 1 : w - 1;
                        const int y1 = (y > 0) ? y - 1 : 0;
                        const int y2 = (y < (h - 1)) ? y + 1 : h - 1;
                        // get next point (the first one caught)
                        for (int yy = y1; (yy <= y2)&end; ++yy)
                        {
                            for (int xx = x1; xx <= x2; ++xx)
                            {
                                if ((x==xx)&(y==yy))
                                    continue;
                                if ((img(yy, xx) == 255)&&check(line, yy, xx))
                                {
                                    // if we found a point which is not in line already(prevent loops)
                                    // we save it to line
                                    line.push_back({xx, yy});
                                    // increment line length
                                    ++len;
                                    // mark, that we haven't reached end of line yet
                                    end = 0;
                                    // choose this new point as new center for 1-radius search for next new point
                                    x = xx;
                                    y = yy;
                                    break;
                                }
                            }
                        }
                    }
                    final_length = len;
                    if (final_length >= 2)
                    {
                        // go backward through found points
                        if (len >= min_length)
                        {
                            // TODO: save line as possible snake
                            level = 255;
                        }
                        auto it = line.end(); --it; // point iterator to last element
                        do
                        {
                            // move last point to resulting image (as 255 if line was long enough, 0 if not)
                            y = it->y;
                            x = it->x;
                            QRgb pix = qRgb(level, level, level);
                            dstImage->setPixel(x, y, pix);
                            img(y, x) = 0;
                            // decrease length and remove last point from our temporary line
                            --it;
                            --len;
                            line.pop_back();
                            // also black this point in source (because we already linked it to some line)
                            y = it->y;
                            x = it->x;
                            no_new_branch = 1;
                            // now we look at new last point
                            // if it have another continuation we break from cycle and go to cycle above (appending line)
                            // if no next point found
                            const int x1 = (x > 0) ? x - 1 : 0;
                            const int x2 = (x < (w - 1)) ? x + 1 : w - 1;
                            const int y1 = (y > 0) ? y - 1 : 0;
                            const int y2 = (y < (h - 1)) ? y + 1 : h - 1;
                            // get next point (the first one caught)
                            for (int yy = y1; (yy <= y2)&(no_new_branch); ++yy)
                                for (int xx = x1; xx <= x2; ++xx)
                                {
                                    if ((x==xx)&(y==yy))
                                        continue;
                                    if ((img(yy, xx) == 255)&&check(line, yy, xx))
                                    {
                                        line.push_back({xx, yy});
                                        ++len;
                                        x = xx;
                                        y = yy;
                                        no_new_branch = 0;
                                        break;
                                    }
                                }
                        } while((len>1)&(no_new_branch));
                    }
                } while ((!no_new_branch)&(len>1));
                img(y, x) = 0;
                QRgb pix = qRgb(level, level, level);
                dstImage->setPixel(x, y, pix);
            }
        }
        percent = y0*10/h;
        if (percent != prev_percent)
        {
            emit print_progress(percent*10);
            prev_percent = percent;
        }
    }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("deleting short lines...finished"));

    return;
}

#define LIM 6
void extraFilters::apply_smooth()
{
    emit print_progress(0);
    emit print_message(QString("smoothing image..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    rawArray2D<unsigned char> img(h, w);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
            img(y, x) = luma8;
            dstImage->setPixel(x, y, pix);
        }

    int percent = 0;
    int prev_percent = percent;

    rawArray2D<unsigned char> mark(h+1, w+1);

    for (int y = 1; y < (h - 1); ++y)
    {
        for (int x = 1; x < (w - 1); ++x)
        {
            const unsigned char p = img(y,x);
            mark(y, x) = (p<img(y-1,x-1))
                        +(p<img(y-1,x))
                        +(p<img(y-1,x+1))
                        +(p<img(y,x-1))
                        +(p<img(y,x+1))
                        +(p<img(y+1,x-1))
                        +(p<img(y+1,x))
                        +(p<img(y+1,x+1));
            if (mark(y, x) < LIM)
            {
                mark(y, x) =(p>img(y-1,x-1))
                            +(p>img(y-1,x))
                            +(p>img(y-1,x+1))
                            +(p>img(y,x-1))
                            +(p>img(y,x+1))
                            +(p>img(y+1,x-1))
                            +(p>img(y+1,x))
                            +(p>img(y+1,x+1));
            }
        }
        percent = y*10/(h - 2);
        if (percent != prev_percent)
        {
            emit print_progress(percent*5);
            prev_percent = percent;
        }
    }

    for (int y = 1; y < (h - 1); ++y)
    {
        for (int x = 1; x < (w - 1); ++x)
        {
            unsigned char val;
            if (mark(y, x) >= LIM)
            {
                val = (img(y-1,x-1)+img(y-1,x)+img(y-1,x+1)+img(y,x-1)+
                                 img(y,x+1)+img(y+1,x-1)+img(y+1,x)+img(y+1,x+1)+4)/8;
            }
            else
                val = img(y, x);
            QRgb pix = qRgb(val, val, val);
            dstImage->setPixel(x, y, pix);
        }
        percent = y*10/(h - 2);
        if (percent != prev_percent)
        {
            emit print_progress(50 + percent*5);
            prev_percent = percent;
        }
    }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("smoothing image...finished"));

    return;
}
#undef LIM

void extraFilters::update_minLineLength(const QString& mll)
{
    bool ok;
    min_length = mll.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading min_length"));
    else
        emit print_message(QString("min_length updated to ") + QString::number(min_length));
    return;
}
