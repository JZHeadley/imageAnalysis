#include "imageAnalysis.h"
#include "../library/imageAnalysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <math.h>
#include <string.h>
# include <assert.h>

#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <json/json.h>

using namespace std;
using namespace cv;

//#define DEBUG_GRAYSCALE true
#define DEBUG_GRAYSCALE false
//#define DEBUG_HIST true
#define DEBUG_HIST false
//#define DEBUG_EQUALIZED true
#define DEBUG_EQUALIZED false
// I don't write very memory efficient c code and tend to introduce some memory leakage but oh well today isn't the day I figure it out...


void convertMatToRGBImage(Mat mat, RGBImage *output) {
    Mat bgr[3];
    split(mat, bgr);
    output->channels = mat.channels();
    output->height = mat.rows;
    output->width = mat.cols;
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

}

/*
 * The following few functions are related to drawing a histogram and I didn't write them myself.
 * I mentioned it to a friend and he wrote them for the fun of it.  They shouldn't make it into whatever I turn in and are not required functions so I figure its fine for testing things out.
 */
int findMax(int *arr, int len) {
    int m = -1;
    for (int i = 0; i < len; i++) {
        if (arr[i] > m) {
            m = arr[i];
        }
    }
    return m;
}

int getTerminalWidth() {
    struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);
    return w.ws_col;
}

void padWithZeroes(char *str, int num, int numDigits) {
    int i = 0;
    if (num == 0) {
        for (int i = 0; i < numDigits; i++) {
            str[i] = '0';
        }
    } else {
        while (num * pow(10, i) < pow(10, numDigits - 1)) {
            str[i] = '0';
            i++;
        }
        sprintf(str + i, "%d", num);
    }
}

void drawHistogram(int *arr, int len) {
    int max = findMax(arr, len);
    int numDigits = 0;
    int width = getTerminalWidth() - 5;
    int dw = max / width;
    while (pow(10, numDigits) < len) {
        numDigits++;
    }
    width = getTerminalWidth() - (numDigits + 2);
    dw = max / width;
    printf("%d, %d, %d\n", max, width, dw);
    for (int i = 0; i < len; i++) {
        char string[numDigits + 2];
        padWithZeroes(string, i, numDigits);
        printf("\n");
        printf("%s|", string);
        for (int j = 0; j < width - numDigits + 2; j++) {
            if (arr[i] > dw * j) {
                printf("#");
            } else {
                break;
            }
        }
    }
    printf("\n");
}

void testing() {
    Mat mat = imread("/home/jzheadley/Pictures/Lenna.png", CV_LOAD_IMAGE_COLOR);
    RGBImage *h_rgbImage = new RGBImage;
    convertMatToRGBImage(mat, h_rgbImage);
    if (DEBUG_GRAYSCALE) {
        imshow("Lenna", mat);
        printf("image pointer: %x width: %i height: %i channels: %i \n", h_rgbImage->image, h_rgbImage->width, h_rgbImage->height, h_rgbImage->channels);
    }
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

    int *h_histogram;
    cudaMallocHost(&h_histogram, sizeof(int) * 256);
    int *d_histogram;
    calculateHistogram(d_grayImage, h_histogram, d_histogram);
    if (DEBUG_HIST) {
        int sum = 0;
        for (int i = 0; i < 256; i++) {
//            printf("%i\n", h_histogram[i]);
            sum += h_histogram[i];
        }
        printf("total pixels: %i num in histogram: %i\n", d_grayImage->width * d_grayImage->height, sum);
        drawHistogram(h_histogram, 256);
    }
    int h_mappings[256];
    equalizeHistogram(h_histogram, h_mappings, d_grayImage->height * d_grayImage->width);
    Image *d_equalizedImage = new Image;
    equalizeImageWithHist(d_grayImage, d_equalizedImage, h_mappings);
    if (DEBUG_EQUALIZED) {
        Image *h_equalizedImage = new Image;
        copyDeviceImageToHost(d_equalizedImage, h_equalizedImage);
        Mat *equalizedMat = new Mat;
        convertImageToMat(h_equalizedImage, equalizedMat);
        imshow("equalized", *equalizedMat);
        while (cvWaitKey(1) != '\33') {

        }

        int *h_histogram2;
        cudaMallocHost(&h_histogram2, sizeof(int) * 256);
        int *d_histogram2;
        calculateHistogram(d_equalizedImage, h_histogram2, d_histogram2);
        drawHistogram(h_histogram2, 256);

    }
}

void readInKernel(Json::Value kernel, int *k, int numValues) {
    const Json::Value &k_vals = kernel["values"];
    printf("numVals %i k_vals size: %i\n", numValues, k_vals.size());
    assert(numValues == k_vals.size());
    for (int i = 0; i < numValues; i++) {
        k[i] = k_vals[i].asInt();
    }
}

void executeOperations(Json::Value json) {
    const Json::Value &operations = json["operations"];
    for (int i = 0; i < operations.size(); i++) {
        string type = operations[i]["type"].asString();
        printf("Operation %i: %s\n", i, type.c_str());
        if (type == "linear-filter") {
            printf("Linear Filter\n");
            Json::Value kernel = operations[i]["kernel"];
            int k_width = kernel["width"].asInt();
            int k_height = kernel["height"].asInt();
            int *k = (int *) malloc(sizeof(int) * k_width * k_height);
            printf("width: %i height: %i\n", k_width, k_height);
            readInKernel(kernel, k, k_width * k_height);



            free(k);
        } else if (type == "median-filter") {
            printf("Median Filter\n");
            Json::Value kernel = operations[i]["kernel"];
            int k_width = kernel["width"].asInt();
            int k_height = kernel["height"].asInt();
            int *k = (int *) malloc(sizeof(int) * k_width * k_height);
            printf("width: %i height: %i\n", k_width, k_height);
            readInKernel(kernel, k, k_width * k_height);




            free(k);
        } else if (type == "gaussian-noise") {
            printf("Gaussian Noise\n");
        } else if (type == "salt-and-pepper") {
            printf("Salt and Pepper Noise\n");
        } else if (type == "histogram-equalization") {
            printf("Histogram Equalization\n");
        } else {
            printf("Unsupported Operation\n");
        }
    }
}

// end not my work
int main(int argc, char *argv[]) {
    Json::Value json;
    std::ifstream config("/home/jzheadley/CLionProjects/imageAnalysis/test/input.json", std::ifstream::binary);
    config>>json;
    string input_image_folder = json["image_folder"].asString();
    string output_image_folder = json["output_dir"].asString();

    bool saveFinalImages = json["saveFinalImages"].asBool();
    bool saveIntermediateImages = json["saveIntermediateImages"].asBool();
    printf("Input: %s\nOutput: %s\nSaving intermediates: %s\nSaving Finals: %s\n",
           input_image_folder.c_str(),
           output_image_folder.c_str(),
           saveIntermediateImages ? "true" : "false",
           saveFinalImages ? "true" : "false");
    executeOperations(json);
    testing();


    return 0;
}

