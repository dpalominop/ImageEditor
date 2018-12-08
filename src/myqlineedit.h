#ifndef MYQLINEEDIT_H
#define MYQLINEEDIT_H

#include <QLineEdit>
#include <QString>

class myQLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    myQLineEdit(const QString &infotext, const QString &txt, QWidget *parent = 0);

signals:
    void print_message(const QString&);
public slots:

protected:
    virtual void focusInEvent(QFocusEvent *e);
private:
    QString info;
};

#endif // MYQLINEEDIT_H
