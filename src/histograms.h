#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <QObject>
#include <QImage>

class histograms : public QObject
{
    Q_OBJECT
public:
    histograms(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void apply_histogramEqualization();
    void apply_histogramAdaptiveEqualization();
signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
private:
    QImage *srcImage;
    QImage *dstImage;

};

#endif // HISTOGRAM_H
