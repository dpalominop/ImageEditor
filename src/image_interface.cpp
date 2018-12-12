#include "image_interface.h"
#include <QLabel>
#include <QImage>
#include <QFileDialog>
#include <QString>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <iterator>

image_interface::image_interface(QLabel *const srclbl, QLabel *const dstlbl,
                                 QImage *const srcimg, QImage *const dstimg,
                                 QObject *parent) : QObject(parent)
{
    srcLabel = srclbl;
    srcImage = srcimg;
    dstLabel = dstlbl;
    dstImage = dstimg;
    newWidth = 640;
    newHeight = 480;
}

void image_interface::load()
{
    QString filename = QFileDialog::getOpenFileName(0, "Load", "", "Images (*.png *.bmp *.jpg *.jpeg *.gif)");
    std::ifstream bmp(filename.toStdString().c_str(), std::ios::binary);

    if(!filename.isEmpty() && bmp){

        static constexpr size_t HEADER_SIZE = 54;
        std::array<char, HEADER_SIZE> header;

        bmp.read(header.data(), header.size());

        if(header[0] == 'B' && header[1]== 'M'){
            auto fileSize = *reinterpret_cast<uint32_t *>(&header[2]);
            auto dataOffset = *reinterpret_cast<uint32_t *>(&header[10]);
            auto headerSize = *reinterpret_cast<uint32_t *>(&header[14]);
            auto width = *reinterpret_cast<uint32_t *>(&header[18]);
            auto height = *reinterpret_cast<uint32_t *>(&header[22]);
            auto depth = *reinterpret_cast<uint16_t *>(&header[28]);

            std::vector<char> img(dataOffset - HEADER_SIZE);
            bmp.read(img.data(), img.size());

            switch (depth) {
                case 24:
                case 32:
                {
                    auto dataSize = ((width * 3 + 3) & (~3)) * height;
                    auto width_ext = ((width * 3 + 3) & (~3));
                    img.resize(dataSize);
                    bmp.read(img.data(), img.size());

                    QImage qimg(width, height, QImage::Format_RGB32);
                    srcImage->operator=(qimg);

                    for (auto i=0; i<height; i++){
                        for(auto j=0; j<width; j++){
                            QRgb value = qRgb(img[i*width_ext+j*3+2], img[i*width_ext+j*3+1], img[i*width_ext+j*3]);
                            srcImage->setPixel(j, height -1 -i, value);
                        }
                    }
                    break;
                }
                case 4:
                {
                    srcImage->load(filename);
                    break;
                }
            }

        }else{
            srcImage->load(filename);
        }

        emit print_message(QString("loaded image ") + QString::number(srcImage->height()) + QString("x") + QString::number(srcImage->width()));
        updateSrcImage();

    }
    return;
}

void image_interface::copy()
{
    *srcImage = *dstImage;
    updateSrcImage();
    return;
}

void image_interface::rescaleSrcImage()
{
    *srcImage = srcImage->scaled(newWidth, newHeight, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    updateSrcImage();
    return;
}

void image_interface::rescaleDstImage()
{
    *dstImage = QImage(newWidth, newHeight, QImage::Format_RGB32);
    uchar *dst = dstImage->bits();
    imageScaled(srcImage->bits(), srcImage->width(), srcImage->height(), srcImage->bytesPerLine(),
                dst, dstImage->width(), dstImage->height(), dstImage->bytesPerLine(), 4);
    updateDstImage();
    return;
}

void image_interface::updateSrcImage()
{
    srcLabel->setPixmap(QPixmap::fromImage(*srcImage));
    srcLabel->resize(srcImage->width(), srcImage->height());
    srcLabel->show();
    return;
}

void image_interface::updateDstImage()
{
    dstLabel->setPixmap(QPixmap::fromImage(*dstImage));
    dstLabel->resize(dstImage->width(), dstImage->height());
    dstLabel->show();
    return;
}

void image_interface::updateWidth(const QString& w)
{
    bool ok;
    newWidth = w.toUInt(&ok);
    if (!ok)
        emit print_message(QString("error reading newWidth"));
    else
        emit print_message(QString("newWidth updated to ") + QString::number(newWidth));
    return;
}

void image_interface::updateHeight(const QString& h)
{
    bool ok;
    newHeight = h.toUInt(&ok);
    if (!ok)
        emit print_message(QString("error reading newHeight"));
    else
        emit print_message(QString("newHeight updated to ") + QString::number(newHeight));
    return;
}
