#ifndef COLORTRANSFORM_H
#define COLORTRANSFORM_H


#include <QImage>
#include "cuda_methods.h"

class ColorTransform: public QObject
{
    Q_OBJECT

    public:
        ColorTransform(QImage *const src, QImage *const dst, QObject *parent = 0);

    public slots:
        void convertToYUV();

    signals:
        void image_ready();
        void print_message(const QString&);
        void print_progress(const int);

    private:
        QImage *srcImage;
        QImage *dstImage;

};

#endif // COLORTRANSFORM_H
