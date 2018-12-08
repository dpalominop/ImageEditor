#ifndef IMAGE_INTERFACE_H
#define IMAGE_INTERFACE_H

#include <QObject>
#include <QLabel>
#include <QImage>
#include <QString>

class image_interface : public QObject
{
    Q_OBJECT
public:
    image_interface(QLabel *const srclbl, QLabel *const dstlbl,
                    QImage *const srcimg, QImage *const dstimg,
                    QObject *parent = 0);

signals:
    void print_message(const QString&);
public slots:
    void updateSrcImage();
    void updateDstImage();
    void load();
    void copy();
    void updateHeight(const QString& h);
    void updateWidth(const QString& w);
    void rescaleSrcImage();
private:
    QLabel *srcLabel;
    QLabel *dstLabel;
    QImage *srcImage;
    QImage *dstImage;
    int newWidth;
    int newHeight;

};

#endif // IMAGE_INTERFACE_H
