#include "wlabel.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    std::string path = argv[1];
    wlabel w(path);

    w.show();

    return a.exec();
}
