#include "templatematching.h"

TemplateMatching::TemplateMatching(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void TemplateMatching::findTemplate()
{
    emit print_progress(0);
    emit print_message(QString("applying Template Matching..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    const int wt = tmpImage.width();
    const int ht = tmpImage.height();

    const int nBytes = w*h*sizeof(float);
    const int nBytesT = wt*ht*sizeof(float);
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    float *src_gray;
    float *tmp_gray;

    src_gray = (float *)malloc(nBytes);
    tmp_gray = (float *)malloc(nBytesT);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            src_gray[x + w*y] = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
        }

    for(int y = 0; y < ht; ++y)
        for(int x = 0; x < wt; ++x)
        {
            const QRgb pix_t = tmpImage.pixel(x, y);
            tmp_gray[x + wt*y] = 0.299 * qRed(pix_t) + 0.587 * qGreen(pix_t) + 0.114 * qBlue(pix_t);
        }

    // find template
    int x, y;
    GetMatch(src_gray, tmp_gray, w, h, wt, ht, &x, &y);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            QRgb pix = srcImage->pixel(x, y);
            dstImage->setPixel(x, y, pix);
        }

    QPainter qPainter(dstImage);
    qPainter.setBrush(Qt::NoBrush);
    qPainter.setPen(Qt::red);
    qPainter.drawRect(x, y, wt, ht);
    qPainter.end();

    free(src_gray);
    free(tmp_gray);

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying Template Matching...finished"));

    return;
}

void TemplateMatching::loadTemplate()
{
    QString filename = QFileDialog::getOpenFileName(0, "Load", "", "Images (*.png *.bmp *.jpg *.jpeg *.gif)");
    std::ifstream bmp(filename.toStdString().c_str(), std::ios::binary);

    if(!filename.isEmpty() && bmp){
        tmpImage.load(filename);
        emit print_message(QString("loaded template ") + QString::number(tmpImage.height()) + QString("x") + QString::number(tmpImage.width()));
    }
    return;
}
