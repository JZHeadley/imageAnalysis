#include "imageAnalysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vectorSize)
        data[idx] = 1.0f / data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size) {
    float *rc = new float[size];
    float *gpuData;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &gpuData, sizeof(float) * size));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float) * size, cudaMemcpyHostToDevice));

    static const int BLOCK_SIZE = 256;
    const int blockCount = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reciprocalKernel<< < blockCount, BLOCK_SIZE>> > (gpuData, size);

    CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(gpuData));
    return rc;
}

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
//    printf("row: %i col: %i\n", row, column);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((image[row * width + column] + image[row * width + column + numPixels] + image[row * width + column + (2 * numPixels)]) / 3);
//        printf("%i\n", output[row * width + column]);
    }
    return;
}

void convertRGBToGrayscale(unsigned char *image, int channels, int width, int height, int method, unsigned char *output) {
    /*
    don't think you mentioned a grayscale conversion method so I looked it up and used this page as my guide
    https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    */
    int totalPixels = width * height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    switch (method) {
        case 0:
            // average method
            convertRGBToGrayscaleAverage<< < threadsPerBlock, blocksPerGrid, 0>> > (image, width, height, totalPixels, channels, output);
            break;
        case 1:
            // luminance method
            break;
        default:
            break;
    }
}

//Image copyHostImageToDevice(Image host){
//    Image device;
//
//    // copy actual image data back to host from device
//    CUDA_CHECK_RETURN(cudaMalloc((void **)&device.image, (int) sizeof(unsigned char)* host.width * host.height));
//    CUDA_CHECK_RETURN(cudaMemcpy(device.image, host.image, (int) sizeof(unsigned char) * host.width * host.height, cudaMemcpyHostToDevice));
//    // copy height and width back
//    CUDA_CHECK_RETURN(cudaMemcpy(&device.height, &host.height, (int) sizeof(int), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMemcpy(&device.width, &host.width, (int) sizeof(int), cudaMemcpyHostToDevice));
//
//    return device;
//}

//Image copyDeviceImageToHost(Image device){
//    Image host;
//    // copy actual image data back to host from device
//    CUDA_CHECK_RETURN(cudaMemcpy(&host.image, device.image, sizeof(unsigned char) * device.width * device.height, cudaMemcpyDeviceToHost));
//    // copy height and width back
//    CUDA_CHECK_RETURN(cudaMemcpy(&host.height, &device.height, sizeof(int), cudaMemcpyDeviceToHost));
//    CUDA_CHECK_RETURN(cudaMemcpy(&host.width, &device.width, sizeof(int), cudaMemcpyDeviceToHost));
//
//    return host;
//}
//unsigned char *image;
//short channels;
//int width;
//int height;

//void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device){
//    // copy actual image data back to host from device
//    printf("%x %i %i %x\n",&host->width,host->height,host->channels,host->image);
////    CUDA_CHECK_RETURN(cudaMalloc(&(device->image), sizeof(unsigned char) * host->width * host->height));
////    CUDA_CHECK_RETURN(cudaMemcpy(&device->image, &host->image, sizeof(unsigned char) * host->width * host->height, cudaMemcpyHostToDevice));
////    CUDA_CHECK_RETURN(cudaMalloc((void**)&(device->image), sizeof(unsigned char) * host->width * host->height * host->channels));
////    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, sizeof(unsigned char) * host->width * host->height * host->channels, cudaMemcpyHostToDevice));
//    // copy height and width back
//    int *d_height;
//    CUDA_CHECK_RETURN(cudaMalloc(&d_height,sizeof(int)));
//    CUDA_CHECK_RETURN(cudaMemcpy(d_height, &(host->height), sizeof(int), cudaMemcpyHostToDevice));
//
//
////    CUDA_CHECK_RETURN(cudaMemcpy(&(device->width), &(host->width), sizeof(int), cudaMemcpyHostToDevice));
////    CUDA_CHECK_RETURN(cudaMemcpy(&(device->channels), &(host->channels), sizeof(int), cudaMemcpyHostToDevice));
//
//}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr<<statement<<" returned "<<cudaGetErrorString(err)<<"("<<err<<") at "<<file<<":"<<line
             <<std::endl;
    exit(1);
}