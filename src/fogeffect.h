#ifndef FOGEFFECT_H
#define FOGEFFECT_H

#include <QImage>
#include <QString>
#include "cuda_methods.h"

class FogEffect : public QObject
{
    Q_OBJECT
public:
    FogEffect(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void calcFogEffect();

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

private:
    QImage *srcImage;
    QImage *dstImage;
};

#endif // FOGEFFECT_H
