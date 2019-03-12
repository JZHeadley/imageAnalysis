#include "imageAnalysis.h"
#include "../library/imageAnalysis.h"

#include <vector>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// I don't write very memory efficient c code and tend to introduce some memory leakage but oh well today isn't the day I figure it out...


void convertMatToImage(Mat mat, Image *output) {
    Mat bgr[3];
    split(mat, bgr);
//    *output->channels = (mat.channels());
//    *output->height = mat.rows;
//    *output->width = mat.cols;
//    printf("%i %i %i\n", output.height,output.width,output.channels);
    unsigned char *temp = ((unsigned char *) malloc(sizeof(unsigned char) * mat.total() * mat.channels()));
    int numPixels = mat.total();
    // swapping into rgb format here instead of the bgr the OpenCV Mat is in
    int row = 0, col = 0;
    for (int i = 0; i < numPixels; i++) {
        row = i / numPixels;
        col = i - ((i / numPixels) * numPixels);
        temp[i] = bgr[2].at<uchar>(row, col);
        temp[i + numPixels] = bgr[1].at<uchar>(row, col);
        temp[i + numPixels * 2] = bgr[0].at<uchar>(row, col);
    }
    unsigned char *gray = ((unsigned char *) malloc(sizeof(unsigned char) * mat.total()));
    convertRGBToGrayscale(temp, mat.channels(), mat.cols, mat.rows, 0, gray);

//    output->image =;
    delete temp;
//    delete bgr;
    output->height=mat.rows;
    output->width=mat.cols;
    output->image=gray;
}


//void convertRGBImageToMat(RGBImage *image,Mat *output) {
//    // adapted from https://stackoverflow.com/a/43190162
//    int numPixels = *image->height * *image->width;
//    Mat channelR(*image->height, *image->width, CV_8UC1, image->image);
//    Mat channelG(*image->height, *image->width, CV_8UC1, image->image + numPixels);
//    Mat channelB(*image->height, *image->width, CV_8UC1, image->image + 2 * numPixels);
//    std::vector <Mat> channels{channelB, channelG, channelR};
//
//    merge(channels, *output);
//}


//
//Mat convertImageToMat(Image *image) {
//    Mat output(*image->height, *image->width, CV_8UC1, image->image);
//    return output;
//}


int main(int argc, char *argv[]) {
    Mat mat;
    mat = imread("/home/jzheadley/Pictures/Lenna.png", CV_LOAD_IMAGE_COLOR);
    Image *image = new Image;
    convertMatToImage(mat, image);
    printf("width: %i height: %i\n", image->width, image->height);
    imshow("Lenna", mat);

//    Mat *output = new Mat;
//    convertImageToMat(image, output);
//    imshow("Converted back and forth", *output);
    waitKey(0);

//    RGBImage *d_rgbImage =new RGBImage;
//    copyHostRGBImageToDevice(rgbImage,d_rgbImage);
//    Image d_grayImage = convertRGBToGrayscale(d_rgbImage,0);
//    Image h_image = copyDeviceImageToHost(d_grayImage);
//    Mat grayscale = convertImageToMat(h_image);
//    imshow("grayscaled with cuda", grayscale);
//    waitKey(0);
    return 0;
}
