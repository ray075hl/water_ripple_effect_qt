#ifndef WIDGET_H
#define WIDGET_H

#include <QLabel>
#include <QImage>
#include <QPushButton>
#include <opencv2/opencv.hpp>
class wlabel : public QLabel
{
    Q_OBJECT

public:
    wlabel(QLabel *parent = 0);
    ~wlabel();

    double** waterCache1;
    double** waterCache2;
    double** waterCacheTemp;
    int height;
    int width;
    unsigned char* imageDataSource;
    unsigned char* imageDataTarget;

    int dropletcounter;
    int framerate;
    double waterDamper;
    double displacementDamper;
    double luminanceDamper;
    int randomDroplets;
    int byte_line;

    QImage image;
    QTimer *timer;

    cv::Mat img;
    cv::Mat img1;

    QPushButton *button;
    void setDroplet(int x, int y);
    void manipulatePixel(int x, int y);


    void mouseMoveEvent(QMouseEvent * event);
    //void paintEvent(QPaintEvent *e);

private slots:
    void tick();

};

#endif // WIDGET_H
