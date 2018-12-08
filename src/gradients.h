#ifndef GRADIENTS_H
#define GRADIENTS_H

#include <QObject>
#include <QImage>
#include <QString>
#include <cmath>
#include "my_user_types.h"

class gradients : public QObject
{
    Q_OBJECT
public:
    explicit gradients(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_nGradient();
    void apply_gradient();
    void apply_GVF();
    void update_gradius(const QString &);
    void update_mu0(const QString &);
    void update_iter0(const QString &);
private:
    QImage * srcImage;
    QImage * dstImage;
    int gradius; //8
    float gradient_minimum_range; //31.0f;
    float mu0;
    int iter0;
    void get_UV(const rawArray2D<unsigned char> &img, rawArray2D<float> *const __restrict Ugvf, rawArray2D<float> *const __restrict Vgvf);

};

#endif // GRADIENTS_H
