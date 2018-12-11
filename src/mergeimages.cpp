#include "mergeimages.h"

MergeImages::MergeImages(QImage *const src, QImage *const dst, QObject *parent) :  QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    return;
}

void MergeImages::mergeImages()
{
    emit print_progress(0);
    emit print_message(QString("applying Merge of Images..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    kernelImage = kernelImage.scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    const int nBytes = w*h*sizeof(unsigned char);
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    unsigned char *src_r, *src_g, *src_b;
    unsigned char *src_r_k, *src_g_k, *src_b_k;

    src_r = (unsigned char *)malloc(nBytes);
    src_g = (unsigned char *)malloc(nBytes);
    src_b = (unsigned char *)malloc(nBytes);
    src_r_k = (unsigned char *)malloc(nBytes);
    src_g_k = (unsigned char *)malloc(nBytes);
    src_b_k = (unsigned char *)malloc(nBytes);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const QRgb pix = srcImage->pixel(x, y);
            src_r[x + w*y] = qRed(pix);
            src_g[x + w*y] = qGreen(pix);
            src_b[x + w*y] = qBlue(pix);

            const QRgb pix_k = kernelImage.pixel(x, y);
            src_r_k[x + w*y] = qRed(pix_k);
            src_g_k[x + w*y] = qGreen(pix_k);
            src_b_k[x + w*y] = qBlue(pix_k);
        }

    // merge images
    addImage(src_r, src_g, src_b, src_r_k, src_g_k, src_b_k, w, h, index);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const int r = src_r[x + w*y];
            const int g = src_g[x + w*y];
            const int b = src_b[x + w*y];
            QRgb pix = qRgb(r, g, b);
            dstImage->setPixel(x, y, pix);
        }

    free(src_r);
    free(src_g);
    free(src_b);
    free(src_r_k);
    free(src_g_k);
    free(src_b_k);

    emit print_progress(100);
    emit image_ready();
    emit print_message(QString("applying Merge of Images...finished"));

    return;
}

void MergeImages::setIndex(const QString& u)
{
    bool ok;
    index = u.toFloat(&ok);
    index = (index>1)?1:((index<0)?0:index);
    if (!ok)
        emit print_message(QString("error reading index"));
    else
        emit print_message(QString("index updated to ") + QString::number(index));
    return;
}

void MergeImages::loadKernel()
{
    QString filename = QFileDialog::getOpenFileName(0, "Load", "", "Images (*.png *.bmp *.jpg *.jpeg *.gif)");
    std::ifstream bmp(filename.toStdString().c_str(), std::ios::binary);

    if(!filename.isEmpty() && bmp){
        kernelImage.load(filename);
        emit print_message(QString("loaded image ") + QString::number(kernelImage.height()) + QString("x") + QString::number(kernelImage.width()));
    }
    return;
}
