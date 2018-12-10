#ifndef GRAYTRANSFORM_H
#define GRAYTRANSFORM_H

#include <QObject>
#include <QImage>
#include "cuda_methods.h"

class GrayTransform : public QObject
{
    Q_OBJECT
public:
    explicit GrayTransform(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void convertToGray();

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

private:
    QImage *srcImage;
    QImage *dstImage;
};

#endif // GRAYTRANSFORM_H
