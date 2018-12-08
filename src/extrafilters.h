#ifndef EXTRAFILTERS_H
#define EXTRAFILTERS_H

#include <QObject>
#include <QImage>

class extraFilters : public QObject
{
    Q_OBJECT
public:
    extraFilters(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_smooth();
    void apply_deleteShortLines();
    void update_minLineLength(const QString& mll);
private:
    QImage * srcImage;
    QImage * dstImage;
    int min_length;

};

#endif // EXTRAFILTERS_H
