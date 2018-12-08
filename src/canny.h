#ifndef CANNY_H
#define CANNY_H
#include "my_user_types.h"
#include <cmath>
#include <QImage>


class canny : public QObject
{
    Q_OBJECT
public:
    canny(QImage *const src, QImage *const dst, QObject *parent = 0);
public slots:
    void apply_canny();
    void updateLowerThreshold(const QString & thr);
    void updateUpperThreshold(const QString & thr);
signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
private:
    QImage * srcImage;
    QImage * dstImage;
    float upperThreshold;
    float lowerThreshold;
};

#endif // CANNY_H
