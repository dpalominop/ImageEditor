#ifndef SEGMENTATIONLEVELSET_H
#define SEGMENTATIONLEVELSET_H

#include <QObject>
#include <QImage>
#include <map>
#include <iostream>
#include "my_user_types.h"

enum node_type
{
    NO_ROOT = 0,
    ROOT = 1,
    HALF_ROOT = 2,
};

struct vpixel
{
    unsigned char value;
    unsigned char root;
};

class segmentationLevelSet : public QObject
{
    Q_OBJECT
public:
    segmentationLevelSet(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_levelSetSegmentation();
    void update_threshold(const QString &thr);
    void update_thicknessLimit(const QString &tl);
    void update_stepsLimit(const QString &sl);

private:
    rawArray2D<unsigned char> upperImage;
    rawArray2D<unsigned char> currentImage;
    rawArray2D<p2i> incrementMap;
    rawArray2D<int> segmentImage;
    QImage * srcImage;
    QImage * dstImage;
    int segmentsCount;
    int threshold;
    int limit_thickness;
    int limit_steps;
    int current_thickness;

    void solveQuad(const int x, const int y);
    int updateUpperImageSize();
    void drawSegments();
};

#endif // SEGMENTATIONLEVELSET_H
