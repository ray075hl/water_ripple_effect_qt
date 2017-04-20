#include "wlabel.h"
#include <QTimer>
#include <QMouseEvent>
#include <QPainter>
#include <math.h>
#include <QWidget>
#include <QPushButton>
#include <QGridLayout>
wlabel::wlabel(QLabel *parent)
    : QLabel(parent)
{

    setMouseTracking(true);

    //-----
    framerate = 40;
    waterDamper = 0.99;
    displacementDamper = 0.15;
    luminanceDamper = 0.80;
    randomDroplets = 4;
    dropletcounter = 0;
    //-----

    img1 =  cv::imread("image7.jpg");
    img  =  img1.clone();
    cv::cvtColor(img,img,CV_BGR2RGB);
    cv::cvtColor(img1,img1,CV_BGR2RGB);

    width = img.cols;
    height = img.rows;

    waterCache1 = (double**)malloc(sizeof(double*)*(img.rows+4));
    for(int i=0;i<img.rows+4;i++){
        waterCache1[i] = (double*)malloc(sizeof(double)*(img.cols+4));
    }
    waterCache2 = (double**)malloc(sizeof(double*)*(img.rows+4));
    for(int i=0;i<img.rows+4;i++){
        waterCache2[i] = (double*)malloc(sizeof(double)*(img.cols+4));
    }
    waterCacheTemp = (double**)malloc(sizeof(double*)*(img.rows+4));
    for(int i=0;i<img.rows+4;i++){
        waterCacheTemp[i] = (double*)malloc(sizeof(double)*(img.cols+4));
    }

    for(int i=0;i<img.rows+4;i++)
        for(int j=0;j<img.cols+4;j++){
             waterCache1[i][j] = 0;
             waterCache2[i][j] = 0;
             waterCacheTemp[i][j] = 0;
        }
    imageDataSource = img1.data;
    imageDataTarget = img.data;

    image = QImage(img.data,  //(const unsigned char*)
                 img.cols,img.rows,
                 img.cols*3,   //new add
                 QImage::Format_RGB888);





    setPixmap(QPixmap::fromImage(image));
    timer = new QTimer(this);
    timer->setInterval(1000/framerate);
    connect(timer, SIGNAL(timeout()), this, SLOT(tick()));
    timer->start();

}

wlabel::~wlabel()
{

}



void wlabel::tick(){

    if(randomDroplets) {
        dropletcounter++;
        if(dropletcounter >= randomDroplets) {
            setDroplet(floor(1.0*rand()/(RAND_MAX+1.0)*width)+1, floor(1.0*rand()/(RAND_MAX+1.0)*height) +1);
            dropletcounter = 0;
        }
    }

    for(int x = 2; x < width+2; x++) {
        for(int y = 2; y < height+2; y++) {
            manipulatePixel(x, y);
        }
    }


    waterCacheTemp = waterCache1;
    waterCache1 = waterCache2;
    waterCache2 = waterCacheTemp;



    setPixmap(QPixmap::fromImage(image));


}

void wlabel::mouseMoveEvent(QMouseEvent *event)
{
    int mouseX = event->x() + 2;
    int mouseY = event->y() + 2;
    if(mouseX > 2 && mouseY > 2 && mouseX < (width+1 ) && mouseY < (height+1))
        setDroplet(mouseX, mouseY);

}
void wlabel::setDroplet(int x, int y){


    waterCache1[y][x]   = 127;
    waterCache1[y][x+1] = 127;
    waterCache1[y][x-1] = 127;
    waterCache1[y+1][x] = 127;
    waterCache1[y-1][x] = 127;


}

int clamp(double v){
    if (v>255)
        v = 255;
    else{

    }
    return v;
}

void wlabel::manipulatePixel(int x, int y){

    int posTargetX = 0;
    int posTargetY = 0;
    int posTarget = 0;
    int posSourceX = 0;
    int posSourceY = 0;
    int posSource = 0;
    int luminance = 0;


    waterCache2[y][x] = ((waterCache1[y][x-1]+waterCache1[y][x+1] +
                          waterCache1[y+1][x]+waterCache1[y-1][x] +
                          waterCache1[y-1][x-1]+waterCache1[y+1][x+1] +
                          waterCache1[y+1][x-1]+waterCache1[y-1][x+1] +
                          waterCache1[y][x-2]+waterCache1[y][x+2] +
                          waterCache1[y+2][x]+waterCache1[y-2][x])/6.0 - waterCache2[y][x])*waterDamper;



    posTargetX = x - 2;
    posTargetY = y - 2;
    posSourceX = floor(waterCache2[y][x]*displacementDamper);

    if(posSourceX<0) posSourceX +=1;

    posSourceY = posTargetY + posSourceX;
    posSourceX += posTargetX;

// keep source position in bounds of canvas
    if(posSourceX < 0) posSourceX = 0;
    if(posSourceX > width - 1) posSourceX = width - 1;
    if(posSourceY < 0) posSourceY = 0;
    if(posSourceY > height - 1) posSourceY = height - 1;

// calculate byte positions in imageData caches
    posTarget = (posTargetX + posTargetY * width) * 3;
    posSource = (posSourceX + posSourceY * width) * 3;

// calculate luminance change for this pixel

    luminance =  floor(waterCache2[y][x]*luminanceDamper);

//manipulate target imageData cache
    imageDataTarget[posTarget]     =    clamp((60+luminance)/60.0*imageDataSource[posSource]     );
    imageDataTarget[posTarget + 1] =    clamp((60+luminance)/60.0*imageDataSource[posSource + 1] );
    imageDataTarget[posTarget + 2] =    clamp((60+luminance)/60.0*imageDataSource[posSource + 2] );


}
