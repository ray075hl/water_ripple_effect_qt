#include <iostream>
#include <cuda.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 32
dim3 block( BLOCK_WIDTH, BLOCK_HEIGHT );
dim3 grid( 0, 0 );
__device__ float clamp(float v){

    if(v > 255) v = 255;
    return v;
}



__global__ void manipulatePixel(int width, int height,
                                float* waterCache1, float* waterCache2,
                                unsigned char* imageDataSource, unsigned char* imageDataTarget)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if( x< width && y< height ){
        x +=2;
        y +=2;
        int posTargetX = 0;
        int posTargetY = 0;
        int posTarget = 0;
        int posSourceX = 0;
        int posSourceY = 0;
        int posSource = 0;
        int luminance = 0;
        float waterDamper = 0.99;
        float displacementDamper = 0.15;
        float luminanceDamper = 0.80;

        waterCache2[y*(width+4) + x] =((waterCache1[y*(width+4)+x-1] + waterCache1[y*(width+4)+x+1] +
                                        waterCache1[(y+1)*(width+4)+x] + waterCache1[(y-1)*(width+4)+x] +
                                        waterCache1[(y-1)*(width+4)+x-1] + waterCache1[(y+1)*(width+4)+x+1] +
                                        waterCache1[(y+1)*(width+4)+x-1] + waterCache1[(y-1)*(width+4)+x+1] +
                                        waterCache1[y*(width+4)+x-2] + waterCache1[y*(width+4)+x+2] +
                                        waterCache1[(y+2)*(width+4)+x] + waterCache1[(y-2)*(width+4)+x])/6
                                        - waterCache2[y*(width+4) + x])*waterDamper;

        posTargetX = x - 2;
        posTargetY = y - 2;
        posSourceX = floor(waterCache2[y*(width+4) + x]*displacementDamper);

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

        luminance =  floor(waterCache2[y*(width+4) + x]*luminanceDamper);

        //manipulate target imageData cache
        imageDataTarget[posTarget]     =    clamp((60+luminance)/60.0*imageDataSource[posSource]     );
        imageDataTarget[posTarget + 1] =    clamp((60+luminance)/60.0*imageDataSource[posSource + 1] );
        imageDataTarget[posTarget + 2] =    clamp((60+luminance)/60.0*imageDataSource[posSource + 2] );


    }
}

__global__ void droplet_kernel(float* waterCache1, int x, int y, int width){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if( idx == x &&  idy == y ){
        waterCache1[y*(width+4) + x]   = 127;
        waterCache1[y*(width+4) + x+1] = 127;
        waterCache1[y*(width+4) + x-1] = 127;
        waterCache1[(y+1)*(width+4) + x] = 127;
        waterCache1[(y-1)*(width+4) + x] = 127;
    }
}
void setDroplet(float* waterCache1, int x, int y, int width)
{


    droplet_kernel<<<grid, block>>>(waterCache1, x, y, width);


}



void cuda_process(int width, int height,
                  float* waterCache1, float* waterCache2, float* waterCacheTemp,
                  unsigned char* imageDataSource, unsigned char* imageDataTarget){



    grid.x = ceil( ( float )width / block.x );
    grid.y = ceil( ( float )height / block.y );


    manipulatePixel<<<grid, block>>>(width, height,
                                     waterCache1, waterCache2,
                                     imageDataSource, imageDataTarget);




}
