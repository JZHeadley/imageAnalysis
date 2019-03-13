#include "imageAnalysis.h"
#include "../library/imageAnalysis.h"

#include <vector>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;
#define DEBUG_GRAYSCALE 0
// I don't write very memory efficient c code and tend to introduce some memory leakage but oh well today isn't the day I figure it out...


void convertMatToRGBImage(Mat mat, RGBImage *output) {
    Mat bgr[3];
    split(mat, bgr);
    output->channels = mat.channels();
    output->height = mat.rows;
    output->width = mat.cols;
//    printf("%i %i %i\n", output.height,output.width,output.channels);
    output->image = ((unsigned char *) malloc(sizeof(unsigned char) * mat.total() * output->channels));
    int numPixels = mat.total();
    // swapping into rgb format here instead of the bgr the OpenCV Mat is in
    int row = 0, col = 0;
    for (int i = 0; i < numPixels; i++) {
        row = i / numPixels;
        col = i - ((i / numPixels) * numPixels);
        output->image[i] = bgr[2].at<uchar>(row, col);
        output->image[i + numPixels] = bgr[1].at<uchar>(row, col);
        output->image[i + numPixels * 2] = bgr[0].at<uchar>(row, col);
    }
}


void convertRGBImageToMat(RGBImage *image, Mat *output) {
    // adapted from https://stackoverflow.com/a/43190162
    int numPixels = image->height * image->width;
    Mat channelR(image->height, image->width, CV_8UC1, image->image);
    Mat channelG(image->height, image->width, CV_8UC1, image->image + numPixels);
    Mat channelB(image->height, image->width, CV_8UC1, image->image + 2 * numPixels);
    std::vector <Mat> channels{channelB, channelG, channelR};

    merge(channels, *output);
}

void convertImageToMat(Image *image, Mat *mat) {
    Mat output(image->height, image->width, CV_8UC1, image->image);
    *mat = output;
//    imshow("grayscaled with cuda", output);
//    waitKey(0);

}


int main(int argc, char *argv[]) {
    Mat mat = imread("/home/jzheadley/Pictures/Lenna.png", CV_LOAD_IMAGE_COLOR);
    RGBImage *h_rgbImage = new RGBImage;
    convertMatToRGBImage(mat, h_rgbImage);
    printf("image pointer: %x width: %i height: %i channels: %i \n", h_rgbImage->image, h_rgbImage->width, h_rgbImage->height, h_rgbImage->channels);
//    imshow("Lenna", mat);

//    Mat *output = new Mat;
//    convertRGBImageToMat(h_rgbImage, output);
//    imshow("Converted back and forth", *output);
//    waitKey(0);

    RGBImage *d_rgbImage = new RGBImage;
    copyHostRGBImageToDevice(h_rgbImage, d_rgbImage);
    printf("image pointer: %x width: %i height: %i channels: %i \n", d_rgbImage->image, d_rgbImage->width, d_rgbImage->height, d_rgbImage->channels);

    Image *d_grayImage = new Image;
    convertRGBToGrayscale(d_rgbImage, d_grayImage, 0);

    if (DEBUG_GRAYSCALE) {
        Image *h_grayImage = new Image;
        copyDeviceImageToHost(d_grayImage, h_grayImage);

        Mat *grayscale = new Mat;
        convertImageToMat(h_grayImage, grayscale);
        imshow("grayscaled with cuda", *grayscale);
//     Loop until escape is pressed
        while (cvWaitKey(1) != '\33') {

        }
    }

    int *h_histogram = (int *) malloc(sizeof(int) * 256);
    int *d_histogram;
    calculateHistogram(d_grayImage, h_histogram, d_histogram);
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        printf("%i\n", h_histogram[i]);
        sum += h_histogram[i];
    }
    printf("total pixels: %i num in histogram: %i\n", d_grayImage->width * d_grayImage->height, sum);
    return 0;
}
