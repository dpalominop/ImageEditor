#include "myqlineedit.h"

myQLineEdit::myQLineEdit(const QString &infotext, const QString &txt, QWidget *parent) :  QLineEdit(txt, parent)
{
    info = infotext;
    return;
}

void myQLineEdit::focusInEvent(QFocusEvent *e)
{
    emit print_message(info);
    QLineEdit::focusInEvent(e);
    return;
}
