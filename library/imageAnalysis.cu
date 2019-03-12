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
        output[row * width + column] = (.21 * image[row * width + column]) + (.72 * image[row * width + column + numPixels]) + (.07 * image[row * width + column + (2 * numPixels)]);
    }
    return;
}

__global__ void convertRGBToGrayscaleAverage(unsigned char *image, int width, int height, int numPixels, int channels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (image[row * width + column] + image[row * width + column + numPixels] + image[row * width + column + (2 * numPixels)]) / 3;
    }
    return;
}

void convertRGBToGrayscale(RGBImage *rgb, Image *gray, int method) {
    /*
    don't think you mentioned a grayscale conversion method so I looked it up and used this page as my guide
    https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    */
    int threadsPerBlock = 256;
    int blocksPerGrid = ((rgb->width * rgb->height) + threadsPerBlock - 1) / threadsPerBlock;
    switch (method) {
        case 0:
            // luminance method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(gray->image), (int) sizeof(unsigned char) * rgb->width * rgb->height));

            convertRGBToGrayscaleLuminance<< < threadsPerBlock, blocksPerGrid, 0>> > (rgb->image, rgb->width, rgb->width * rgb->height, rgb->height, rgb->channels, gray->image);
            gray->width = rgb->width;
            gray->height = rgb->height;
            break;
        case 1:
            // average method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(gray->image), (int) sizeof(unsigned char) * rgb->width * rgb->height));

            convertRGBToGrayscaleAverage<< < threadsPerBlock, blocksPerGrid, 0>> > (rgb->image, rgb->width, rgb->width * rgb->height, rgb->height, rgb->channels, gray->image);
            gray->width = rgb->width;
            gray->height = rgb->height;
            break;
        default:
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
    host->image = (unsigned char *) malloc(sizeof(unsigned char) * device->height * device->width);
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(host->image, device->image, sizeof(unsigned char) * device->width * device->height, cudaMemcpyDeviceToHost));
    // copy height and width back to host
    host->height = device->height;
    host->width = device->width;

}

void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device) {
    // copy actual image data to device from host
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