#include <QApplication>
#include <QImage>
#include <QWidget>
#include <QLabel>
#include <QBoxLayout>
#include <QPixmap>
#include <iostream>
#include <QColor>
#include <QPushButton>
#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QProgressBar>
#include <QDebug>
#include <QGroupBox>
#include <QButtonGroup>
#include "gradients.h"
#include "histograms.h"
#include "image_interface.h"
#include "colortransform.h"
#include "fouriertransform.h"
#include "fogeffect.h"
#include "graytransform.h"
#include "threshold.h"
#include "mergeimages.h"
#include "templatematching.h"
#include "myqlineedit.h"

#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

void printArray(const float* a, const unsigned int n) {
    QString s = "(";
    unsigned int ii;
    for (ii = 0; ii < n - 1; ++ii)
        s.append(QString::number(a[ii])).append(", ");
    s.append(QString::number(a[ii])).append(")");

    qDebug() << s;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QBoxLayout* phbxLayout0 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *loadButt = new QPushButton("Load");
        QPushButton *copyButt = new QPushButton("Filtered->Source");
        phbxLayout0->addWidget(loadButt);
        phbxLayout0->addWidget(copyButt);

    QBoxLayout* phbxLayout1 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *rescaleButt = new QPushButton("rescale input");
        myQLineEdit* ptxtSrcWidth = new myQLineEdit("Rescale source image with this width", "640");
        myQLineEdit* ptxtSrcHeight = new myQLineEdit("Rescale source image with this height", "480");
        phbxLayout1->addWidget(rescaleButt);
        phbxLayout1->addWidget(ptxtSrcWidth);
        phbxLayout1->addWidget(ptxtSrcHeight);

    QBoxLayout* phbxLayout2 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *grayButt = new QPushButton("Gray Format");
        phbxLayout2->addWidget(grayButt);
    QBoxLayout* phbxLayout3 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *binaryButt = new QPushButton("Threshold");
        myQLineEdit* ptxtBinaryRadius = new myQLineEdit("Binary Umbral", "100");
        phbxLayout3->addWidget(binaryButt);
        phbxLayout3->addWidget(ptxtBinaryRadius);
    QBoxLayout* phbxLayout4 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *histeqButt = new QPushButton("Histogram Equalization (HE)");
        phbxLayout4->addWidget(histeqButt);
    QBoxLayout* phbxLayout5 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *mergeButt = new QPushButton("Add Images");
        QPushButton *mergeAddButt = new QPushButton("Load");
        myQLineEdit *ptxtmergeIndexRadius = new myQLineEdit("Merge Index", "0.5");
        phbxLayout5->addWidget(mergeButt);
        phbxLayout5->addWidget(mergeAddButt);
        phbxLayout5->addWidget(ptxtmergeIndexRadius);
    QBoxLayout* phbxLayout6 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *fogButt = new QPushButton("Fog Effect");
        phbxLayout6->addWidget(fogButt);
    QBoxLayout* phbxLayout7 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *gradButt = new QPushButton("Gradient");
        phbxLayout7->addWidget(gradButt);
    QBoxLayout* phbxLayout8 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *yuvButt = new QPushButton("YUV");
        QGroupBox *groupBox = new QGroupBox;
        QButtonGroup *buttonGroup = new QButtonGroup;
        QHBoxLayout *vbox = new QHBoxLayout;
        QCheckBox* pchkY = new QCheckBox("Y");
        QCheckBox* pchkCb = new QCheckBox("Cb");
        QCheckBox* pchkCr = new QCheckBox("Cr");
        pchkY->setChecked(true);
        pchkCb->setChecked(false);
        pchkCr->setChecked(false);
        buttonGroup->addButton(pchkY);
        buttonGroup->addButton(pchkCb);
        buttonGroup->addButton(pchkCr);
        buttonGroup->setExclusive(true);
        phbxLayout8->addWidget(yuvButt);
        vbox->addWidget(pchkY);
        vbox->addWidget(pchkCb);
        vbox->addWidget(pchkCr);
        groupBox->setLayout(vbox);
        phbxLayout8->addWidget(groupBox);
    QBoxLayout* phbxLayout9 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *fftButt = new QPushButton("FFT");
        phbxLayout9->addWidget(fftButt);
    QBoxLayout* phbxLayout10 = new QBoxLayout(QBoxLayout::LeftToRight);
        QPushButton *templateButt = new QPushButton("Template Matching");
        QPushButton *tempAddButt = new QPushButton("Load Template");
        phbxLayout10->addWidget(templateButt);
        phbxLayout10->addWidget(tempAddButt);

    QProgressBar * ppbrProgress = new QProgressBar();
    ppbrProgress->setMaximum(100);
    ppbrProgress->setMinimum(0);
    ppbrProgress->setValue(0);
    QLineEdit * ptxtInfo = new QLineEdit();
    ptxtInfo->setReadOnly(true);
    ptxtInfo->setAlignment(Qt::AlignCenter);

    QBoxLayout* pbxLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    pbxLayout->addLayout(phbxLayout0);
    pbxLayout->addLayout(phbxLayout1);
    pbxLayout->addLayout(phbxLayout2);
    pbxLayout->addLayout(phbxLayout3);
    pbxLayout->addLayout(phbxLayout4);
    pbxLayout->addLayout(phbxLayout5);
    pbxLayout->addLayout(phbxLayout6);
    pbxLayout->addLayout(phbxLayout7);
    pbxLayout->addLayout(phbxLayout8);
    pbxLayout->addLayout(phbxLayout9);
    pbxLayout->addLayout(phbxLayout10);
    pbxLayout->addWidget(ppbrProgress);
    pbxLayout->addWidget(ptxtInfo);

    QLabel originalLabel; QImage originalImage;
    QLabel filteredLabel; QImage filteredImage;

    image_interface ii(&originalLabel, &filteredLabel, &originalImage, &filteredImage);
    GrayTransform graytrans(&originalImage, &filteredImage);
    Threshold bintrans(&originalImage, &filteredImage);
    histograms histeq(&originalImage, &filteredImage);
    MergeImages mergimag(&originalImage, &filteredImage);
    FogEffect fogeff(&originalImage, &filteredImage);
    gradients gradtrans(&originalImage, &filteredImage);
    ColorTransform coltrans(&originalImage, &filteredImage);
    FourierTransform foutrans(&originalImage, &filteredImage);
    TemplateMatching tempmatch(&originalImage, &filteredImage);

    QObject::connect(loadButt, SIGNAL(clicked()), &ii, SLOT(load()));
    QObject::connect(copyButt, SIGNAL(clicked()), &ii, SLOT(copy()));
    QObject::connect(rescaleButt, SIGNAL(clicked()), &ii, SLOT(rescaleDstImage()));
        QObject::connect(ptxtSrcWidth, SIGNAL(textChanged(const QString &)), &ii, SLOT(updateWidth(const QString &)));
        QObject::connect(ptxtSrcHeight, SIGNAL(textChanged(const QString &)), &ii, SLOT(updateHeight(const QString &)));

    QObject::connect(grayButt, SIGNAL(clicked()), &graytrans, SLOT(convertToGray()));
    QObject::connect(binaryButt, SIGNAL(clicked()), &bintrans, SLOT(convertToBinary()));
        QObject::connect(ptxtBinaryRadius, SIGNAL(textChanged(const QString&)), &bintrans, SLOT(setUmbral(const QString&)));
    QObject::connect(histeqButt, SIGNAL(clicked()), &histeq, SLOT(apply_histogramEqualization()));
    QObject::connect(mergeButt, SIGNAL(clicked()), &mergimag, SLOT(mergeImages()));
        QObject::connect(mergeAddButt, SIGNAL(clicked()), &mergimag, SLOT(loadKernel()));
        QObject::connect(ptxtmergeIndexRadius, SIGNAL(textChanged(const QString&)), &mergimag, SLOT(setIndex(const QString&)));
    QObject::connect(gradButt, SIGNAL(clicked()), &gradtrans, SLOT(apply_gradient()));
    QObject::connect(fogButt, SIGNAL(clicked()), &fogeff, SLOT(calcFogEffect()));
    QObject::connect(yuvButt, SIGNAL(clicked()), &coltrans, SLOT(convertToYUV()));
        QObject::connect(pchkY, SIGNAL(stateChanged(const int)), &coltrans, SLOT(setY(const int)));
        QObject::connect(pchkCb, SIGNAL(stateChanged(const int)), &coltrans, SLOT(setCb(const int)));
        QObject::connect(pchkCr, SIGNAL(stateChanged(const int)), &coltrans, SLOT(setCr(const int)));
    QObject::connect(fftButt, SIGNAL(clicked()), &foutrans, SLOT(calcFFT()));
    QObject::connect(templateButt, SIGNAL(clicked()), &tempmatch, SLOT(findTemplate()));
        QObject::connect(tempAddButt, SIGNAL(clicked()), &tempmatch, SLOT(loadTemplate()));

    QObject::connect(&graytrans, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&bintrans, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&histeq, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&mergimag, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&fogeff, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&gradtrans, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&coltrans, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&foutrans, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));
    QObject::connect(&tempmatch, SIGNAL(image_ready()), &ii, SLOT(updateDstImage()));

    QObject::connect(&ii, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(ptxtSrcWidth, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(ptxtSrcHeight, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));

    QObject::connect(&graytrans, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&bintrans, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&histeq, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&mergimag, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(ptxtBinaryRadius, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&fogeff, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&gradtrans, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&coltrans, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&foutrans, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));
    QObject::connect(&tempmatch, SIGNAL(print_message(const QString&)), ptxtInfo, SLOT(setText(const QString&)));

    QObject::connect(&graytrans, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&bintrans, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&histeq, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&mergimag, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&fogeff, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&gradtrans, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&coltrans, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&foutrans, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));
    QObject::connect(&tempmatch, SIGNAL(print_progress(const int)), ppbrProgress, SLOT(setValue(const int)));

    QWidget wgt;
    wgt.resize(256, 100);
    wgt.setMinimumWidth(430);
    wgt.setLayout(pbxLayout);
    wgt.show();
    wgt.move(1400, 20);
    originalLabel.move(0,0);
    filteredLabel.move(0,100);

    int cudaDevice = 0;
    char cudaDeviceName [100];

    cuInit(0);
    cuDeviceGet(&cudaDevice, 0);
    cuDeviceGetName(cudaDeviceName, 100, cudaDevice);

    wgt.setWindowTitle(QString("ImageEditor - ")+QString(cudaDeviceName));
    return app.exec();
}
