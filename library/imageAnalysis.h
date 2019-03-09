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

#endif