#include "dct.h"

dct::dct(QImage *const src, QImage *const dst, QObject *parent) : QObject(parent)
{
    srcImage = src;
    dstImage = dst;
    show_DC = 1;
    show_lowAC = 1;
    show_mediumAC = 1;
    show_highAC = 1;
    return;
}

#include "my_user_types.h"
static const int cospi8sqrt2minus1=20091;
static const int sinpi8sqrt2 =35468;

static void idct4x4(short *const input, short *const output)
{
    int i;
    int a1, b1, c1, d1;
    int ip0, ip4, ip8, ip12;
    short tmp_block[16];
    short *ip=input;
    short *tp=tmp_block;
    int temp1, temp2;

    for (i = 0; i < 4; ++i)
    {
        ip0 = ip[0];
        ip4 = ip[4];
        ip8 = ip[8];
        ip12 = ip[12];

        a1 = ip0+ip8;
        b1 = ip0-ip8;

        temp1 = (ip4 * sinpi8sqrt2)>>16;
        temp2 = ip12 + ((ip12 * cospi8sqrt2minus1)>>16);
        c1 = temp1 - temp2;

        temp1 = ip4 + ((ip4 * cospi8sqrt2minus1)>>16);
        temp2 = (ip12 * sinpi8sqrt2)>>16;
        d1 = temp1 + temp2;

        tp[0] = a1 + d1;
        tp[12] = a1 - d1;
        tp[4] = b1 + c1;
        tp[8] = b1 - c1;

        ++ip;
        ++tp;
    }

    short *op = output;
    tp = tmp_block;
    for(i = 0; i < 4; ++i)
    {
        a1 = tp[0]+tp[2];
        b1 = tp[0]-tp[2];
        temp1 = (tp[1] * sinpi8sqrt2)>>16;
        temp2 = tp[3]+((tp[3] * cospi8sqrt2minus1)>>16);
        c1 = temp1 - temp2;
        temp1 = tp[1] + ((tp[1] * cospi8sqrt2minus1)>>16);
        temp2 = (tp[3] * sinpi8sqrt2)>>16;
        d1 = temp1 + temp2;

        /* after adding this results to predictors - clamping maybe needed */
        tp[0] = ((a1 + d1 + 4) >> 3);
        op[0] = (short)((tp[0] > 255) ? 255 : ((tp[0] < 0) ? 0 : tp[0] ));
        tp[3] = ((a1 - d1 + 4) >> 3);
        op[3] = (short)((tp[3] > 255) ? 255 : ((tp[3] < 0) ? 0 : tp[3] ));
        tp[1] = ((b1 + c1 + 4) >> 3);
        op[1] = (short)((tp[1] > 255) ? 255 : ((tp[1] < 0) ? 0 : tp[1] ));
        tp[2] = ((b1 - c1 + 4) >> 3);
        op[2] = (short)((tp[2] > 255) ? 255 : ((tp[2] < 0) ? 0 : tp[2] ));

        op+=4;
        tp+=4;
    }

    return;
}


static void dct4x4(short *input, short *output)
{
    // input - pointer to start of block in raw frame. I-line of the block will be input + I*width
    // output - pointer to encoded_macroblock.block[i] data.
    int i;
    int a1, b1, c1, d1;
    short *ip = input;
    short *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ((ip[0] + ip[3])<<3);
        b1 = ((ip[1] + ip[2])<<3);
        c1 = ((ip[1] - ip[2])<<3);
        d1 = ((ip[0] - ip[3])<<3);

        op[0] = (short)(a1 + b1);
        op[2] = (short)(a1 - b1);

        op[1] = (short)((c1 * 2217 + d1 * 5352 +  14500)>>12);
        op[3] = (short)((d1 * 2217 - c1 * 5352 +   7500)>>12);

        ip += 4;
        op += 4;

    }
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = op[0] + op[12];
        b1 = op[4] + op[8];
        c1 = op[4] - op[8];
        d1 = op[0] - op[12];

        op[0] = (( a1 + b1 + 7)>>4); // quant using dc_q only first time
        op[8] = (( a1 - b1 + 7)>>4);
        op[4]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
        op[12] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);

        ++op;
    }
    return;
}

void dct::apply_dctidct()
{
    emit print_progress(0);
    emit print_message(QString("DCT filtering..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);
    short min = 0x7FFF;
    short max = -min;
    rawArray2D<short> temp(h, w);

    int percent = 0;
    int prev_percent = percent;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            short A[16], B[16];
            int i = 0;
            for (int yy = y - 1; yy < y + 3; ++yy)
                for (int xx = x - 1; xx < x + 3; ++xx)
                {
                    int xxx = xx < 0 ? 0 : xx;
                    xxx = xxx >= w ? w - 1 : xxx;
                    int yyy = yy < 0 ? 0 : yy;
                    yyy = yyy >= h ? h -1 : yyy;
                    const QRgb pix = srcImage->pixel(xxx, yyy);
                    const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
                    const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
                    A[i] = luma8;
                    ++i;
                }
            dct4x4(A, B);
            B[0]  *= show_DC;       B[1]  *= show_lowAC;    B[2]  *= show_mediumAC; B[3]  *= show_highAC;
            B[4]  *= show_lowAC;    B[5]  *= show_lowAC;    B[6]  *= show_mediumAC; B[7]  *= show_highAC;
            B[8]  *= show_mediumAC; B[9]  *= show_mediumAC; B[10] *= show_mediumAC; B[11] *= show_highAC;
            B[12] *= show_highAC;   B[13] *= show_highAC;   B[14] *= show_highAC;   B[15] *= show_highAC;
            idct4x4(B,A);
            max = A[5] > max ? A[5] : max;
            min = A[5] < min ? A[5] : min;
            temp(y, x) = A[5];
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            print_progress(percent*10);
            prev_percent = percent;
        }
    }

    if (show_DC == 1)
    {
        max = 255;
        min = 0;
    }

    const int range = (max - min) < 1 ? 1 : max - min;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            short level16 = (short)((temp(y, x) - min)*255/range);
            level16 = level16 > 255 ? 255 : level16;
            level16 = level16 < 0 ? 0 : level16;
            const unsigned char level8 = (unsigned char)level16;
            QRgb pix = qRgb(level8, level8, level8);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("DCT filtering...finished"));

    return;
}

void dct::apply_AVG()
{
    emit print_progress(0);
    emit print_message(QString("averaging points..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int percent = 0;
    int prev_percent = percent;

    short A[16], B[16];
    rawArray2D<short> temp(h, w);
    short min = 0x7FFF;
    short max = -min;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            int i = 0;
            short val = 0;
            for (int yy = y - 1; yy < y + 3; ++yy)
                for (int xx = x - 1; xx < x + 3; ++xx)
                {
                    int xxx = xx < 0 ? 0 : xx;
                    xxx = xxx >= w ? w - 1 : xxx;
                    int yyy = yy < 0 ? 0 : yy;
                    yyy = yyy >= h ? h - 1 : yyy;
                    const QRgb pix = srcImage->pixel(xxx, yyy);
                    const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
                    const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
                    A[i] = luma8;
                    ++i;
                }
            dct4x4(A, B);
            val = B[0]; //AVG
            max = max < val ? val : max;
            min = min > val ? val : min;
            temp(y, x) = val;
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            print_progress(percent*10);
            prev_percent = percent;
        }
    }

    const int range = max - min;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level = (unsigned char)((temp(y, x) - min)*255/range)&(0xE0);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("averaging points...finished"));

    return;
}

void dct::apply_LF()
{
    emit print_progress(0);
    emit print_message(QString("extracting low frequency..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int percent = 0;
    int prev_percent = percent;

    short A[16], B[16];
    rawArray2D<int> temp(h, w);
    int min = 0x0FFFFFFF;
    int max = -min;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            int i = 0;
            int val = 0;
            for (int yy = y - 1; yy < y + 3; ++yy)
                for (int xx = x - 1; xx < x + 3; ++xx)
                {
                    int xxx = xx < 0 ? 0 : xx;
                    xxx = xxx >= w ? w - 1 : xxx;
                    int yyy = yy < 0 ? 0 : yy;
                    yyy = yyy >= h ? h -1 : yyy;
                    const QRgb pix = srcImage->pixel(xxx, yyy);
                    const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
                    const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
                    A[i] = luma8;
                    ++i;
                }
            dct4x4(A, B);
            val = abs(B[1]) + abs(B[4]) + abs(B[5]); //LF
            //val = val*val;
            val = (int)sqrt(val);
            max = max < val ? val : max;
            min = min > val ? val : min;
            temp(y, x) = val;
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            print_progress(percent*10);
            prev_percent = percent;
        }
    }

    const int range = max - min;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level = (unsigned char)((temp(y, x) - min)*255/range)&(0xE0);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }


    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("extracting low frequency...finished"));

    return;
}

void dct::apply_MF()
{
    emit print_progress(0);
    emit print_message(QString("extracting medium frequency..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int percent = 0;
    int prev_percent = percent;

    short A[16], B[16];
    rawArray2D<int> temp(h, w);
    int min = 0x0FFFFFFF;
    int max = -min;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            int i = 0;
            int val = 0;
            for (int yy = y - 1; yy < y + 3; ++yy)
                for (int xx = x - 1; xx < x + 3; ++xx)
                {
                    int xxx = xx < 0 ? 0 : xx;
                    xxx = xxx >= w ? w - 1 : xxx;
                    int yyy = yy < 0 ? 0 : yy;
                    yyy = yyy >= h ? h -1 : yyy;
                    const QRgb pix = srcImage->pixel(xxx, yyy);
                    const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
                    const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
                    A[i] = luma8;
                    ++i;
                }
            dct4x4(A, B);
            val = abs(B[2]) + abs(B[6]) + abs(B[8]) + abs(B[9]) + abs(B[10]); //MF
            //val = val*val;
            val = (int)sqrt(val);
            max = max < val ? val : max;
            min = min > val ? val : min;
            temp(y, x) = val;
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            print_progress(percent*10);
            prev_percent = percent;
        }
    }

    const int range = max - min;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level = (unsigned char)((temp(y, x) - min)*255/range)&(0xE0);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("extracting medium frequency...finished"));

    return;
}

void dct::apply_HF()
{
    emit print_progress(0);
    emit print_message(QString("extracting hidh frequency..."));
    const int w = srcImage->width();
    const int h = srcImage->height();
    *dstImage = QImage(w, h, QImage::Format_RGB32);

    int percent = 0;
    int prev_percent = percent;

    short A[16], B[16];
    rawArray2D<int> temp(h, w);
    int min = 0x0FFFFFFF;
    int max = -min;

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
            int i = 0;
            int val = 0;
            for (int yy = y - 1; yy < y + 3; ++yy)
                for (int xx = x - 1; xx < x + 3; ++xx)
                {
                    int xxx = xx < 0 ? 0 : xx;
                    xxx = xxx >= w ? w - 1 : xxx;
                    int yyy = yy < 0 ? 0 : yy;
                    yyy = yyy >= h ? h -1 : yyy;
                    const QRgb pix = srcImage->pixel(xxx, yyy);
                    const int luma32 = 0.299 * qRed(pix) + 0.587 * qGreen(pix) + 0.114 * qBlue(pix);
                    const unsigned char luma8 = luma32 > 255 ? 255 : luma32 < 0 ? 0 : luma32;
                    A[i] = luma8;
                    ++i;
                }
            dct4x4(A, B);
            val = abs(B[3]) + abs(B[7]) + abs(B[11]) + abs(B[12]) + abs(B[13]) + abs(B[14]) + abs(B[15]); //HF
            //val = val*val;
            val = (int)sqrt(val);
            max = max < val ? val : max;
            min = min > val ? val : min;
            temp(y, x) = val;
        }
        percent = y*10/h;
        if (percent != prev_percent)
        {
            print_progress(percent*10);
            prev_percent = percent;
        }
    }

    const int range = max - min;
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
        {
            const unsigned char level = (unsigned char)((temp(y, x) - min)*255/range)&(0xE0);
            QRgb pix = qRgb(level, level, level);
            dstImage->setPixel(x, y, pix);
        }

    emit image_ready();
    emit print_progress(100);
    emit print_message(QString("extracting hidh frequency...finished"));

    return;
}

void dct::set_DC(const int state)
{
    if (state == Qt::Checked)
    {
        emit print_message(QString("will display DC level"));
        show_DC = 1;
    }
    else
    {
        emit print_message(QString("won't display DC level"));
        show_DC = 0;
    }
    return;
}

void dct::set_LAC(const int state)
{
    if (state == Qt::Checked)
    {
        emit print_message(QString("will display low frequency AC level"));
        show_lowAC = 1;
    }
    else
    {
        emit print_message(QString("won't display low frequency AC level"));
        show_lowAC = 0;
    }
    return;
}

void dct::set_MAC(const int state)
{
    if (state == Qt::Checked)
    {
        emit print_message(QString("will display medium frequency AC level"));
        show_mediumAC = 1;
    }
    else
    {
        emit print_message(QString("won't display medium frequency AC level"));
        show_mediumAC = 0;
    }
    return;
}

void dct::set_HAC(const int state)
{
    if (state == Qt::Checked)
    {
        emit print_message(QString("will display high frequency AC level"));
        show_highAC = 1;
    }
    else
    {
        emit print_message(QString("won't display high frequency AC level"));
        show_highAC = 0;
    }
    return;
}
