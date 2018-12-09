#ifndef COLORTRANSFORM_H
#define COLORTRANSFORM_H

#include <QObject>
#include <QImage>
#include "cuda_methods.h"

class ColorTransform: public QObject
{
    Q_OBJECT

    public:
        ColorTransform(QImage *const src, QImage *const dst, QObject *parent = 0);

    public slots:
        void convertToYUV();
        void setY(const int state);
        void setCb(const int state);
        void setCr(const int state);

    signals:
        void image_ready();
        void print_message(const QString&);
        void print_progress(const int);

    private:
        QImage *srcImage;
        QImage *dstImage;
        int show_Y=1, show_Cb=0, show_Cr=0;

};

#endif // COLORTRANSFORM_H
