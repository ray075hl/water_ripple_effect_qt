#include "wlabel.h"
#include <QApplication>
#include <QImage>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    wlabel w;
    w.show();

    return a.exec();
}
