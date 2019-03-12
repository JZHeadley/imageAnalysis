#ifndef IMAGEANALYSIS_LIBRARY_H
#define IMAGEANALYSIS_LIBRARY_H


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
 * Helper method that takes in an rgb image to convert and then makes the correct cuda call and converts the response back
 * @param rgb rgb image to convert to grayscale
 * @param gray grayscale image resulting from the conversion
 * @param method enum representing which method to use for grayscale conversion
 *      0 for luminance method 1 for average method
 */
void convertRGBToGrayscale(RGBImage *rgb, Image *gray, int method);

void copyHostImageToDevice(Image *host, Image *device);

void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device);

void copyDeviceImageToHost(Image *device, Image *host);

void calculateHistogram(Image *image, int *h_histogram);

#endif