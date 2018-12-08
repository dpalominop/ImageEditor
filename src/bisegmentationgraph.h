#ifndef BISEGMENTATIONGRAPH_H
#define BISEGMENTATIONGRAPH_H

#include <QObject>
#include <QImage>
#include <QString>
#include "my_user_types.h"

struct biedge
{
    int v1;
    int v2;
    int smooth;
};

struct bivertice
{
    int root;
    int size;
};

class biSegmentationGraph : public QObject
{
    Q_OBJECT
public:
    biSegmentationGraph(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_bisegmentationGraph();
    void update_level(const QString&);
    void update_minSegSize(const QString&);
    void set_source(const int index);
private:
    QImage * srcImage;
    QImage * dstImage;
    unsigned char level;
    float minSegSize;
    float Cb, Cr, Cg;

    rawArray2D<bivertice> vertices;
    std::vector<biedge> edges;

    int find_root(const int i);
};

#endif // BISEGMENTATIONGRAPH_H
