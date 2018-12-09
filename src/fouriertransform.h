#ifndef FOURIERTRANSFORM_H
#define FOURIERTRANSFORM_H

#include <QObject>
#include <QImage>
#include <cuda_methods.h>

class FourierTransform : public QObject
{
    Q_OBJECT

    public:
        FourierTransform(QImage *const src, QImage *const dst, QObject *parent = 0);

    public slots:
        void calcFFT();

    signals:
        void image_ready();
        void print_message(const QString&);
        void print_progress(const int);

    private:
        QImage *srcImage;
        QImage *dstImage;
};

#endif // FOURIERTRANSFORM_H
