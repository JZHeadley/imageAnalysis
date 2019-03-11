#include "imageAnalysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < vectorSize)
        data[idx] = 1.0f / data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
    float *rc = new float[size];
    float *gpuData;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

    static const int BLOCK_SIZE = 256;
    const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
    reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

    CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(gpuData));
    return rc;
}

Image convertRGBToGrayscale(RGBImage rgb,int method) {
    Image gray;
    switch (method) {
        case 0:
            // luminance method
            break;
        case 1:
            // average method
            break;
        default:
            break;
    }
    return gray;
}

Image copyHostImageToDevice(Image host){
    Image device;

    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMalloc((void **)&device.image, (int) sizeof(unsigned char)* host.width * host.height));
    CUDA_CHECK_RETURN(cudaMemcpy(device.image, host.image, (int) sizeof(unsigned char) * host.width * host.height, cudaMemcpyHostToDevice));
    // copy height and width back
    CUDA_CHECK_RETURN(cudaMemcpy(&device.height, &host.height, (int) sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&device.width, &host.width, (int) sizeof(int), cudaMemcpyHostToDevice));

    return device;
}

Image copyDeviceImageToHost(Image device){
    Image host;
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(&host.image, device.image, sizeof(unsigned char) * device.width * device.height, cudaMemcpyDeviceToHost));
    // copy height and width back
    CUDA_CHECK_RETURN(cudaMemcpy(&host.height, &device.height, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(&host.width, &device.width, sizeof(int), cudaMemcpyDeviceToHost));

    return host;
}
//unsigned char *image;
//short channels;
//int width;
//int height;
RGBImage *copyHostRGBImageToDevice(RGBImage *host){
    RGBImage *device = new RGBImage;
    // copy actual image data back to host from device
    printf("%i %i %i %x\n",*(&(host->width)),host->height,host->channels,host->image);
//    CUDA_CHECK_RETURN(cudaMalloc(&(device->image), sizeof(unsigned char) * host->width * host->height));
//    CUDA_CHECK_RETURN(cudaMemcpy(&device->image, &host->image, sizeof(unsigned char) * host->width * host->height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&(device->image), sizeof(unsigned char) * host->width * host->height * host->channels));
    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, sizeof(unsigned char) * host->width * host->height * host->channels, cudaMemcpyHostToDevice));
    // copy height and width back
    CUDA_CHECK_RETURN(cudaMemcpy(&device->height, &host->height, sizeof(int), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMemcpy(&(device->width), &(host->width), sizeof(int), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMemcpy(&(device->channels), &(host->channels), sizeof(int), cudaMemcpyHostToDevice));

    return device;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}