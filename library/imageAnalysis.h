#ifndef IMAGEANALYSIS_LIBRARY_H
#define IMAGEANALYSIS_LIBRARY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Declaring my own types so I don't have to link OpenCV to my library even though its probably about the same as theirs just with fewer features
// Should be a little more easier to use whatever library you want to read in with as well since it just needs to be converted to this format
// Also need this because while you can use C++ classes and things with CUDA its a lot harder than straight C...
typedef struct {
    // using 1-D arrays because its not that hard to work with and CUDA works better with them (better memory addressing, cache hits, etc.)
    unsigned char *image;
    short channels;
    int width;
    int height;
} RGBImage;

typedef struct {
    unsigned char *image;
    int width;
    int height;
} Image;

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

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value);

/**
 * Helper method that takes in an rgb image to convert and then makes the correct cuda call and converts the response back
 * @param d_rgb rgb image to convert to grayscale
 * @param d_gray grayscale image resulting from the conversion
 * @param method enum representing which method to use for grayscale conversion
 *      0 for luminance method 1 for average method
 */

void copyHostImageToDevice(Image *host, Image *device);

void copyDeviceRGBImageToHost(RGBImage *device, RGBImage *host);

void convertRGBToGrayscale(RGBImage *d_rgb, Image *d_gray, int method);

void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device);

void copyDeviceImageToHost(Image *device, Image *host);

void calculateHistogram(Image *image, int *h_histogram, int *d_histogram);

void equalizeHistogram(int *original, int *mappings, int numPixels);

void equalizeImageWithHist(Image *image, Image *d_equalizedImage, int *h_mappings);

void linearFilter(Image *image, Image *output, float *kernel, int kWidth, int kHeight);

void averageFilter(Image *image, Image *output, float *kernel, int kWidth, int kHeight);

void medianFilter(Image *image, Image *output, int *kernel, int kWidth, int kHeight);

void sobelFilter(Image *image, Image *output);

void compassFilter(Image *image, Image *output);

void saltAndPepperNoise(Image *image, Image *output, int level);

void addGaussianNoise(Image *image, Image *output, float mean, float stdDev);

void cleanupEdgeDetection();

void setupEdgeDetection();

void cleanupRandomness();

void imageDilation(Image *image, Image *output, int *structuringElement, int kWidth, int kHeight);

void imageErosion(Image *image, Image *output, int *structuringElement, int kWidth, int kHeight);

void thresholdImage(Image *image, Image *output, int threshold);

void otsuThresholdImage(Image *image, Image *output);

void kMeansThresholding(Image *image, Image *output, int k);

void kMeansThresholding(Image *image, Image *output);

/**
 *
 * @param rgb the image to extract a color from
 * @param out the output image
 * @param color a number specifying which channel to extract
 *              0 for red
 *              1 for green
 *              2 for blue
 */
void extractSingleColorChannel(RGBImage *rgb, Image *out, int color);

void setupRandomness(Image *image);

void imageQuantization(Image *image, Image *output, int *levels, int numLevels);

int calcMSQE(Image *d_image, Image *d_tempImage);

#endif