#ifndef MERGEIMAGES_H
#define MERGEIMAGES_H

#include <QObject>
#include <QLabel>
#include <QImage>
#include <QFileDialog>
#include <QString>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <iterator>
#include "cuda_methods.h"

class MergeImages : public QObject
{
    Q_OBJECT
public:
    explicit MergeImages(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void mergeImages();
    void setIndex(const QString& u);
    void loadKernel();

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

private:
    QImage *srcImage;
    QImage *dstImage;
    QImage kernelImage;
    float index=0.5;
};

#endif // MERGEIMAGES_H
