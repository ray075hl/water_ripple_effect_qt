#ifndef WLABEL_H
#define WLABEL_H

#include <QWidget>
#include <QLabel>
#include <QImage>
#include <QPushButton>
#include <opencv2/opencv.hpp>
#include <QMouseEvent>
class wlabel : public QLabel
{
    Q_OBJECT

public:
    explicit wlabel(std::string path, QLabel *parent = 0);
    ~wlabel();


    float* waterCache1;
    float* waterCache2;
    float* waterCacheTemp;

    unsigned char* imageDataSource;
    unsigned char* imageDataTarget;
//--------------------------------------
    float*         d_waterCache1;
    float*         d_waterCache2;
    float*         d_waterCacheTemp;

    unsigned char*  d_imageDataSource;
    unsigned char*  d_imageDataTarget;
//--------------------------------------

    int dropletcounter;
    int framerate;
    double waterDamper;
    double displacementDamper;
    double luminanceDamper;
    int randomDroplets;
    int byte_line;
    int height;
    int width;

    QImage image;
    QTimer *timer;

    cv::Mat img;
    cv::Mat img1;

    QPushButton *button;

    void mouseMoveEvent(QMouseEvent * event);

    int mouseX;
    int mouseY;

    int frameid;
private slots:
    void tick();

};

#endif // WLABEL_H
