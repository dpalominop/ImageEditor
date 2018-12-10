#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <QObject>
#include <QImage>
#include "cuda_methods.h"

class Threshold : public QObject
{
    Q_OBJECT
public:
    explicit Threshold(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void convertToBinary();
    void setUmbral(const QString& u);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

private:
    QImage *srcImage;
    QImage *dstImage;
    int umbral=100;
};

#endif // THRESHOLD_H
