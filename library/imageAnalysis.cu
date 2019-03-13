#include "imageAnalysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value);

__global__ void convertRGBToGrayscaleLuminance(unsigned char *image, int width, int height, int numPixels, int channels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((.21 * image[row * width + column]) + (.72 * image[row * width + column + numPixels]) + (.07 * image[row * width + column + (2 * numPixels)]));
    }
    return;
}

__global__ void convertRGBToGrayscaleAverage(unsigned char *image, int width, int height, int numPixels, int channels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((image[row * width + column] + image[row * width + column + numPixels] + image[row * width + column + (2 * numPixels)]) / 3);
    }
    return;
}


void convertRGBToGrayscale(RGBImage *d_rgb, Image *d_gray, int method) {
    /*
    don't think you mentioned a grayscale conversion method so I looked it up and used this page as my guide
    https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    */
    int totalPixels = d_rgb->width * d_rgb->height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    switch (method) {
        case 0:
            // luminance method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(d_gray->image), (int) sizeof(unsigned char) * d_rgb->width * d_rgb->height));

            convertRGBToGrayscaleLuminance<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, d_rgb->height, totalPixels, d_rgb->channels, d_gray->image);
            d_gray->width = d_rgb->width;
            d_gray->height = d_rgb->height;
            break;
        case 1:
            // average method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(d_gray->image), (int) sizeof(unsigned char) * d_rgb->width * d_rgb->height));

            convertRGBToGrayscaleAverage<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, d_rgb->height, totalPixels, d_rgb->channels, d_gray->image);
            d_gray->width = d_rgb->width;
            d_gray->height = d_rgb->height;
            break;
        default:
            break;
    }
}

__global__ void calcHistogram(unsigned char *data, int width, int numPixels, int *histogram) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if (tid < numPixels) {
        int val = data[row * width + column];
        if (val != 0)
            atomicAdd(&histogram[val], 1);
    }
    return;
}

void calculateHistogram(Image *image, int *h_histogram, int *d_histogram) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
//    int operationsPerThread = 10;
//    int numOperations = totalPixels / operationsPerThread;
//    int blocksPerGrid = (numOperations + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram, (int) sizeof(int) * 256));
    CUDA_CHECK_RETURN(cudaMemset(d_histogram, 0, 256 * sizeof(int)));

    calcHistogram<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, image->width, totalPixels, d_histogram);

    CUDA_CHECK_RETURN(cudaMemcpy(h_histogram, d_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost));
}

void equalizeHistogram(int* original,int* equalized, int * mappings){

}

void histogramEqualizeImage(){

}

void extractSingleColorChannel(RGBImage *rgb, Image *out, int color) {
    out->width = rgb->width;
    out->height = rgb->height;
    int totalPixels = rgb->width * rgb->height;
    //TODO: Memory leaks right here probably should fix but meh it should work well enough like this...
    switch (color) {
        case 0: // red
            out->image = rgb->image;
            break;
        case 1: // green
            out->image = rgb->image + totalPixels;

            break;
        case 2: // blue
            out->image = rgb->image + (2 * totalPixels);
            break;
        default:
            printf("invalid option\n");
            break;
    }
}

void copyHostImageToDevice(Image *host, Image *device) {
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMalloc((void **) &(device->image), (int) sizeof(unsigned char) * host->width * host->height));
    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, (int) sizeof(unsigned char) * host->width * host->height, cudaMemcpyHostToDevice));
    // copy height and width to device
    device->height = host->height;
    device->width = host->width;
}

void copyDeviceImageToHost(Image *device, Image *host) {
    // copy height and width back to host
    host->height = device->height;
    host->width = device->width;
    host->image = (unsigned char *) malloc(sizeof(unsigned char) * host->height * host->width);
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(host->image, device->image, sizeof(unsigned char) * device->width * device->height, cudaMemcpyDeviceToHost));
//    for (int i = 0; i < (host->height * host->width); i++) {
//        printf("%i\n", host->image[i]);
//    }

}

void copyDeviceRGBImageToHost(RGBImage *device, RGBImage *host) {
    // copy height and width back to host
    host->height = device->height;
    host->width = device->width;
    host->channels = 1;
    host->image = (unsigned char *) malloc(sizeof(unsigned char) * host->height * host->width);
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(host->image, device->image, sizeof(unsigned char) * device->width * device->height, cudaMemcpyDeviceToHost));
//    for (int i = 0; i < (host->height * host->width); i++) {
//        printf("%i\n", host->image[i]);
//    }

}

void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device) {
    // copy actual image data to device from host
//    unsigned char*
    CUDA_CHECK_RETURN(cudaMalloc((void **) &(device->image), sizeof(unsigned char) * host->width * host->height * host->channels));
    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, sizeof(unsigned char) * host->width * host->height * host->channels, cudaMemcpyHostToDevice));
    // copy height and width to device
    device->height = host->height;
    device->width = host->width;
    device->channels = host->channels;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr<<statement<<" returned "<<cudaGetErrorString(err)<<"("<<err<<") at "<<file<<":"<<line<<std::endl;
    exit(1);
}