#ifndef TEMPLATEMATCHING_H
#define TEMPLATEMATCHING_H

#include <QObject>
#include <QImage>
#include <QPainter>
#include <QFileDialog>
#include <fstream>
#include "cuda_methods.h"

class TemplateMatching : public QObject
{
    Q_OBJECT
public:
    explicit TemplateMatching(QImage *const src, QImage *const dst, QObject *parent = 0);

public slots:
    void findTemplate();
    void loadTemplate();

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);

private:
    QImage *srcImage;
    QImage *dstImage;
    QImage tmpImage;
};

#endif // TEMPLATEMATCHING_H
