#include "wlabel.h"
#include "ui_wlabel.h"
#include <QTimer>
#include <QMouseEvent>
#include <QPainter>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking


struct timeval start,end;

extern void setDroplet(float*, int, int, int);

extern void cuda_process(int width, int height,
                         float* waterCache1, float* waterCache2, float* waterCacheTemp,
                         unsigned char* imageDataSource, unsigned char* imageDataTarget);

wlabel::wlabel(std::string path,QLabel *parent) :
    QLabel(parent)
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

    img1 =  cv::imread(path);
    img  =  img1.clone();
    cv::cvtColor(img,img,CV_BGR2RGB);
    cv::cvtColor(img1,img1,CV_BGR2RGB);

    width = img.cols;
    height = img.rows;


    setFixedSize(width, height);


    waterCache1 = (float*)malloc(sizeof(float)*(img.rows+4)*(img.cols+4));
    waterCache2 = (float*)malloc(sizeof(float)*(img.rows+4)*(img.cols+4));
    waterCacheTemp = (float*)malloc(sizeof(float)*(img.rows+4)*(img.cols+4));

    for(int i=0;i<img.rows+4;i++)
        for(int j=0;j<img.cols+4;j++){
             waterCache1[i*(img.cols+4)+j] = 0;
             waterCache2[i*(img.cols+4)+j] = 0;
             waterCacheTemp[i*(img.cols+4)+j] = 0;
        }
    imageDataSource = img1.data;
    imageDataTarget = img.data;

    frameid = 0;

    checkCudaErrors(cudaMalloc((void**)&d_waterCache1,       (height+4)*(width+4)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_waterCache2,       (height+4)*(width+4)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_waterCacheTemp,    (height+4)*(width+4)*sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_imageDataSource,  (height)*(width)*3*sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_imageDataTarget,  (height)*(width)*3*sizeof(unsigned char)));


    checkCudaErrors(cudaMemcpy(d_waterCache1,    waterCache1,       (height+4)*(width+4)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_waterCache2,    waterCache2,       (height+4)*(width+4)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_waterCacheTemp, waterCacheTemp,    (height+4)*(width+4)*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_imageDataSource, imageDataSource, (height)*(width)*3*sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_imageDataTarget, imageDataTarget, (height)*(width)*3*sizeof(unsigned char), cudaMemcpyHostToDevice));



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

    if(0 == frameid%25)      gettimeofday(&start,0);
    if(randomDroplets) {
        dropletcounter++;
        if(dropletcounter >= randomDroplets) {
            setDroplet(d_waterCache1, floor(1.0*rand()/(RAND_MAX+1.0)*width)+1, floor(1.0*rand()/(RAND_MAX+1.0)*height) +1,width);
            dropletcounter = 0;
        }
    }


    cuda_process(width, height,
                 d_waterCache1,  d_waterCache2, d_waterCacheTemp,
                 d_imageDataSource,  d_imageDataTarget);


    checkCudaErrors(cudaMemcpy(imageDataTarget, d_imageDataTarget, (height)*(width)*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(d_waterCacheTemp, d_waterCache1,    (height+4)*(width+4)*sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_waterCache1,    d_waterCache2,    (height+4)*(width+4)*sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_waterCache2,    d_waterCacheTemp, (height+4)*(width+4)*sizeof(float), cudaMemcpyDeviceToDevice));
    if(0 == frameid%25){
        gettimeofday(&end,0);
        int timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
        char title[256];
        sprintf(title,"Water Ripple Effect Using GPU: %.1f fps", 1000000.0/timeuse);
        setWindowTitle(title);
    }
    frameid++;
    setPixmap(QPixmap::fromImage(image));

}



void wlabel::mouseMoveEvent(QMouseEvent *event)
{
    mouseX = event->x() + 2;
    mouseY = event->y() + 2;
    if(mouseX > 2 && mouseY > 2 && mouseX < (width+1 ) && mouseY < (height+1))
        setDroplet(d_waterCache1, mouseX, mouseY, width);

}
























