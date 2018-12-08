#ifndef SEGMENTATIONGRAPH_H
#define SEGMENTATIONGRAPH_H

#include <QObject>
#include <QImage>
#include <QString>
#include <cmath>
#include "my_user_types.h"

struct edge
{
    int v1;
    int v2;
    float weight;
};

struct vertice
{
    int root;
    int size;
    float threshold;
    float max_weight;
    float candidate_weight;
    int root_candidate;
};

class segmentationGraph : public QObject
{
    Q_OBJECT
public:
    segmentationGraph(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_segmentationGraph();
    void update_K(const QString&);
    void update_sigma(const QString&);
    void update_minSegSize(const QString&);
private:
    QImage * srcImage;
    QImage * dstImage;
    float K;
    float sigma;
    float minSegSize;

    rawArray2D<vertice> vertices;
    std::vector<edge> edges;

    int find_root(const int i);
};

#endif // SEGMENTATIONGRAPH_H
