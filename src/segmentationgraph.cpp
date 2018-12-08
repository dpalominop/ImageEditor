#include "segmentationgraph.h"
#include <map>

segmentationGraph::segmentationGraph(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    K = 10.0f;
    sigma = 10;
    minSegSize = 100;
    return;
}

int segmentationGraph::find_root(const int i)
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

bool operator<(const edge &a, const edge &b)
{
    return a.weight < b.weight;
}

void segmentationGraph::apply_segmentationGraph()
{
    emit print_progress(0);
    emit print_message(QString("finding segments..."));
    // here ew convert QImages to vertices and edges with their weights
    // TODOIT
    const int w = srcImage->width();
    const int h = srcImage->height();
    rawArray2D<c3f> img(h, w);
    vertices.resize_without_copy(h, w);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const int ind = y*w + x;
            QRgb pix = srcImage->pixel(x, y);
            img(y, x).r = (float)qRed(pix);
            img(y, x).g = (float)qGreen(pix);
            img(y, x).b = (float)qBlue(pix);
            vertices[ind].root = ind;
            vertices[ind].size = 1;
            vertices[ind].threshold = K;
            vertices[ind].max_weight = 0;
        }
    // it's recommended to smooth image

    edges.clear();
    edges.reserve(w*h*4); // w*h*4 - w*3 - h*3 + 2 - exactly

    int percent = 0;
    int prev_percent = percent;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            edge e;
            int x1, y1;
            x1 = x + 1;
            y1 = y - 1;
            if ((y1 >= 0) && (x1 < w)) // if above-right pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                const float dr = img[e.v1].r - img[e.v2].r;
                const float dg = img[e.v1].g - img[e.v2].g;
                const float db = img[e.v1].b - img[e.v2].b;
                e.weight = sqrt(dr*dr + dg*dg + db*db);
                edges.push_back(e);
            }
            x1 = x + 1;
            y1 = y;
            if (x1 < w) // if right pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                const float dr = img[e.v1].r - img[e.v2].r;
                const float dg = img[e.v1].g - img[e.v2].g;
                const float db = img[e.v1].b - img[e.v2].b;
                e.weight = sqrt(dr*dr + dg*dg + db*db);
                edges.push_back(e);
            }
            x1 = x + 1;
            y1 = y + 1;
            if ((x1 < w) && (y1 < h)) // if right-below pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                const float dr = img[e.v1].r - img[e.v2].r;
                const float dg = img[e.v1].g - img[e.v2].g;
                const float db = img[e.v1].b - img[e.v2].b;
                e.weight = sqrt(dr*dr + dg*dg + db*db);
                edges.push_back(e);
            }
            x1 = x;
            y1 = y;
            if (y1 < h) // if below pixel exists
            {
                e.v1 = y*w + x;
                e.v2 = y1*w + x1;
                const float dr = img[e.v1].r - img[e.v2].r;
                const float dg = img[e.v1].g - img[e.v2].g;
                const float db = img[e.v1].b - img[e.v2].b;
                e.weight = sqrt(dr*dr + dg*dg + db*db);
                edges.push_back(e);
            }
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            emit print_progress(percent*4);
            prev_percent = percent;
        }
    }

    emit print_progress(40);
    const int edges_size = edges.size();
    const int sz = w*h;
    int segments_count = sz;
    std::cout << "segments count: " << segments_count << std::endl;

    std::sort(&edges[0], &edges[edges_size]);

    for(int i = 0; i < edges_size; ++i)
    {
        edge e = edges[i];
        int a = find_root(e.v1);
        int b = find_root(e.v2);
        if (a != b)
        {
            // if threshold allow
            if ((e.weight <= vertices[a].threshold) && (e.weight <= vertices[b].threshold))
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
                if (vertices[a].max_weight < e.weight)
                {
                    vertices[a].threshold = e.weight + K/vertices[a].size;
                    vertices[a].max_weight = e.weight;
                }
                --segments_count;
            }
        }
        percent = i*10/edges_size;
        if (percent != prev_percent)
        {
            emit print_progress(40+percent*4);
            prev_percent = percent;
        }
    }
    std::cout << "segments count: " << segments_count << std::endl;
    emit print_progress(80);

    // merge small segments?
    // TODOIT
if (sigma > 50.0f)
{
    std::cout << "stable" << std::endl;
    bool more_small_segments = true;
    while (more_small_segments)
    {
        percent = 0;
        prev_percent = percent;
        more_small_segments = false;
        for(int i = 0; i < edges_size; ++i)
        {
            edge e = edges[i];
            int a = find_root(e.v1);
            int b = find_root(e.v2);
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
            percent = i*10/edges_size;
            if (percent != prev_percent)
            {
                emit print_progress(80+percent*2);
                prev_percent = percent;
            }
        }
        std::cout << "segments count: " << segments_count << std::endl;
    }
}
else
{
    percent = 0;
    prev_percent = percent;
    std::cout << "test" << std::endl;
    for (int i = 0; i < sz; ++i)
    {
        int a = find_root(i);
        vertice v = vertices[a];
        if (v.size < minSegSize)
        {
            const int x = i % w;
            const int y = i / w;
            int b[8]; float minw, testw;
            b[0] = ((x > 0)&&(y > 0)) ? i - w - 1 : -1;
            b[1] = (y > 0) ? i - w : -1;
            b[2] = ((x < (w - 1))&&(y > 0)) ? i - w + 1 : -1;
            b[3] = (x > 0) ? i - 1 : -1;
            b[4] = (x < (w - 1)) ? i + 1 : -1;
            b[5] = ((x > 0)&&(y < (h - 1))) ? i + w - 1 : -1;
            b[6] = (y < (h - 1)) ? i + w : -1;
            b[7] = ((x < (w - 1))&&(y < (h - 1))) ? i + w + 1 : -1;
            for(int j = 0; j < 8; ++j)
            {
                if (b[j]>=0)
                    if (a == find_root(b[j]))
                        b[j] = -1;
            }

            float dr,dg,db;
            if (b[0] >= 0)
            {
                dr = img[i].r - img[b[0]].r;
                dg = img[i].g - img[b[0]].g;
                db = img[i].b - img[b[0]].b;
                minw = sqrt(dr*dr + dg*dg + db*db);
            }
            else minw = 1000.0f;

            for (int j = 1; j < 8; ++j)
            {
                if (b[j] >= 0)
                {
                    dr = img[i].r - img[b[j]].r;
                    dg = img[i].g - img[b[j]].g;
                    db = img[i].b - img[b[j]].b;
                    testw = sqrt(dr*dr + dg*dg + db*db);
                }
                else testw = 1000.0f;
                if (testw < minw)
                {
                    minw = testw;
                    b[0] = b[j];
                }
            }

            if (b[0] >= 0)
            {
                vertices[a].root = b[0];
                vertices[b[0]].size += vertices[a].size;
                --segments_count;
            }
        }
        percent = i*10/sz;
        if (percent != prev_percent)
        {
            emit print_progress(80+percent*2);
            prev_percent = percent;
        }
    }
    std::cout << "segments count: " << segments_count << std::endl;
}

    struct segment_color
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };
    std::map<int, segment_color> root_colors;
    for(int i = 0; i < sz; ++i)
    {
        if (vertices[i].root == i) // found segment base pixel
        {
            segment_color buf;
            if (vertices[i].size > minSegSize)
            {
                buf.r = qrand()%256;
                buf.g = qrand()%256;
                buf.b = qrand()%256;
            }
            else
            {
                buf.r = 0;
                buf.g = 0;
                buf.b = 0;
            }
            root_colors[i] = buf;
        }
    }

    *dstImage = QImage(w, h, QImage::Format_RGB32);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            int segr = find_root(y*w + x);
            segment_color segc = root_colors[segr];
            QRgb pix = qRgb(segc.r, segc.g, segc.b);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("finding segments...") + QString::number(segments_count));

    return;
}

void segmentationGraph::update_K(const QString& nK)
{
    bool ok;
    K = nK.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading K"));
    else
        emit print_message(QString("K updated to ") +QString::number(K));
    return;
}

void segmentationGraph::update_sigma(const QString& s)
{
    bool ok;
    sigma = s.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading sigma"));
    else
        emit print_message(QString("sigma updated to ") + QString::number(sigma));
    return;
}

void segmentationGraph::update_minSegSize(const QString& mss)
{
    bool ok;
    minSegSize = mss.toFloat(&ok);
    if (!ok)
        emit print_message(QString("error reading minSegSize"));
    else
        emit print_message(QString("minSegSize updated to ") + QString::number(minSegSize));
    return;
}
