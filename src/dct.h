#ifndef DCT_H
#define DCT_H

#include <QObject>
#include <QImage>
#include <cmath>

class dct : public QObject
{
    Q_OBJECT
public:
    dct(QImage *const src, QImage *const dst, QObject *parent = 0);

signals:
    void image_ready();
    void print_message(const QString&);
    void print_progress(const int);
public slots:
    void apply_dctidct();
    void apply_AVG();
    void apply_LF();
    void apply_MF();
    void apply_HF();
    void set_DC(const int state);
    void set_LAC(const int state);
    void set_MAC(const int state);
    void set_HAC(const int state);
private:
    QImage * srcImage;
    QImage * dstImage;
    unsigned char show_highAC;
    unsigned char show_mediumAC;
    unsigned char show_lowAC;
    unsigned char show_DC;

};

#endif // DCT_H
