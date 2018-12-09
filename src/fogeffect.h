#ifndef FOGEFFECT_H
#define FOGEFFECT_H

#include <QImage>
#include <QString>
#include "cuda_methods.h"

class FogEffect : public QObject
{
    Q_OBJECT
public:
    explicit FogEffect(QObject *parent = nullptr);

signals:

public slots:
};

#endif // FOGEFFECT_H
