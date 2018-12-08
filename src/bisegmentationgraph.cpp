#include "bisegmentationgraph.h"
#include <map>
#include <QString>

biSegmentationGraph::biSegmentationGraph(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    level = 10.0f;
    minSegSize = 100;
    Cr = 0.299f;
    Cg = 0.587f;
    Cb = 0.114f;

    return;
}

int biSegmentationGraph::find_root(const int i)
{
    int j = i; // get previous element
    int k = vertices[j].root;
    while (j != k)
    {
        j = k;
        k = vertices[k].root;
    };
    // also connect to the rootto make tree shorter
    vertices[i].root = k;
    return k;
}

void biSegmentationGraph::apply_bisegmentationGraph()
{
    emit print_progress(0);
    emit print_message(QString("finding bi-segments..."));
    // here ew convert QImages to vertices and edges with their weights
    // TODOIT
    const int w = srcImage->width();
    const int h = srcImage->height();
    rawArray2D<unsigned char> img(h, w);
    vertices.resize_without_copy(h, w);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            const int luma32 = Cr * qRed(pix) + Cg * qGreen(pix) + Cb * qBlue(pix);
            const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
            img(y, x) = luma8;
            const int ind = y*w + x;
            vertices[ind].root = ind;
            vertices[ind].size = 1;
        }
    // it's recommended to smooth image

    edges.clear();
    edges.reserve(w*h*4); // w*h*4 - w*3 - h*3 + 2 - exactly
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            biedge e;
            int x1, y1;
            x1 = x + 1;
            y1 = y - 1;
            if ((y1 >= 0) && (x1 < w)) // if above-right pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                e.smooth = (img[e.v1] > level)==(img[e.v2] > level);
                edges.push_back(e);
            }
            x1 = x + 1;
            y1 = y;
            if (x1 < w) // if right pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                e.smooth = (img[e.v1] > level)==(img[e.v2] > level);
                edges.push_back(e);
            }
            x1 = x + 1;
            y1 = y + 1;
            if ((x1 < w) && (y1 < h)) // if right-below pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                e.smooth = (img[e.v1] > level)==(img[e.v2] > level);
                edges.push_back(e);
            }
            x1 = x;
            y1 = y;
            if (y1 < h) // if below pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                e.smooth = (img[e.v1] > level)==(img[e.v2] > level);
                edges.push_back(e);
            }
        }
    const int edges_size = edges.size();
    const int sz = w*h;
    int segments_count = sz;
    std::cout << "segments count: " << segments_count << std::endl;

    for(int i = 0; i < edges_size; ++i)
    {
        biedge e = edges[i];
        int a = find_root(e.v1);
        int b = find_root(e.v2);
        if (a != b)
        {
            // if threshold allow
            if (e.smooth)
            {
                if (vertices[a].size < vertices[b].size)
                {
                    const int c = a;
                    a = b;
                    b = c;
                }
                // connect these two (order of connection? with the biggest in the root for optimization?)
                vertices[b].root = a;
                vertices[a].size += vertices[b].size;
                --segments_count;
            }
        }
    }
    std::cout << "segments count: " << segments_count << std::endl;

    if (minSegSize > 0)
    {
        // merge small segments
        bool more_small_segments = true;
        while (more_small_segments)
        {
            more_small_segments = false;
            for(int i = 0; i < edges_size; ++i)
            {
                biedge e = edges[i];
                int a = find_root(e.v1);
                int b = find_root(e.v2);
                //if (e.smooth)  continue;
                if (a != b)
                {
                    // if threshold allow
                    if ((vertices[a].size < minSegSize) || (vertices[b].size < minSegSize))
                    {
                        if (vertices[a].size < vertices[b].size)
                        {
                            const int c = a;
                            a = b;
                            b = c;
                        }
                        // connect these two (order of connection? with the biggest in the root for optimization?)
                        vertices[b].root = a;
                        vertices[a].size += vertices[b].size;
                        --segments_count;
                        more_small_segments = true;
                    }
                }
            }
            std::cout << "segments count: " << segments_count << std::endl;
        }
    }

    *dstImage = QImage(w, h, QImage::Format_RGB32);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            int seg = find_root(y*w + x);

            unsigned char r = (img[seg] > level)*255;

            QRgb pix = qRgb(r, 0, 255 - r);
            dstImage->setPixel(x, y, pix);
        }

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("finding bi-segments...") + QString::number(segments_count));

    return;
}

void biSegmentationGraph::update_level(const QString& lvl)
{
    bool ok;
    level = lvl.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading level"));
    else
        emit print_message(QString("level updated to ") + QString::number(level));
    return;
}

void biSegmentationGraph::update_minSegSize(const QString& mss)
{
    bool ok;
    minSegSize = mss.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading minSegSize"));
    else
        emit print_message(QString("minSegSize updated to ") + QString::number(minSegSize));
    return;
}

void biSegmentationGraph::set_source(const int index)
{
    if (index == 1)
    {
        Cr = 1.0f;
        Cg = 0.0f;
        Cb = 0.0f;
    }
    else if (index == 2)
    {
        Cr = 0.0f;
        Cg = 1.0f;
        Cb = 0.0f;
    }
    else if (index == 3)
    {
        Cr = 0.0f;
        Cg = 0.0f;
        Cb = 1.0f;
    }
    else
    {
        Cr = 0.299f;
        Cg = 0.587f;
        Cb = 0.114f;
    }
    return;
}
