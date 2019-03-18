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

#include <dirent.h>
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
//#define DEBUG_LINFILTER true
#define DEBUG_LINFILTER false
//#define DEBUG_MEDFILTER true
#define DEBUG_MEDFILTER false
// I don't write very memory efficient c code and tend to introduce some memory leakage but oh well today isn't the day I figure it out...


vector<int> compression_params;

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
// end not my work

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

    int *h_histogram = nullptr;
    cudaMallocHost(&h_histogram, sizeof(int) * 256);
    int *d_histogram = nullptr;
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

        int *h_histogram2 = nullptr;
        cudaMallocHost(&h_histogram2, sizeof(int) * 256);
        int *d_histogram2 = nullptr;
        calculateHistogram(d_equalizedImage, h_histogram2, d_histogram2);
        drawHistogram(h_histogram2, 256);

    }
    Image *d_linFilImage = new Image;
    int kern[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    linearFilter(d_equalizedImage, d_linFilImage, kern, 3, 3);
    if (DEBUG_LINFILTER) {
        Image *h_linFilImage = new Image;
        copyDeviceImageToHost(d_linFilImage, h_linFilImage);
        Mat *linFilMat = new Mat;
        convertImageToMat(h_linFilImage, linFilMat);
        imshow("Linear Filter", *linFilMat);
        while (cvWaitKey(1) != '\33') {

        }
    }
    Image *d_medFilImage = new Image;
    int medKern[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    medianFilter(d_equalizedImage, d_medFilImage, medKern, 3, 3);
    if (DEBUG_MEDFILTER) {
        Image *h_medFilImage = new Image;
        copyDeviceImageToHost(d_medFilImage, h_medFilImage);
        Mat *medFilMat = new Mat;
        convertImageToMat(h_medFilImage, medFilMat);
        imshow("Median Filter", *medFilMat);
        while (cvWaitKey(1) != '\33') {

        }
    }

}

void readInKernel(Json::Value kernel, int *k, int numValues) {
    const Json::Value &k_vals = kernel["values"];
    assert(numValues == k_vals.size());
    for (int i = 0; i < numValues; i++) {
        k[i] = k_vals[i].asInt();
    }
}

vector <string> getFileNames(string input_image_folder) {
    // adapted from this https://stackoverflow.com/a/612176
    vector <string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_image_folder.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (strncmp(ent->d_name, ".", 1)) {
                files.push_back(ent->d_name);
//                printf("%s\n", ent->d_name);
            }
        }
        closedir(dir);
    } else {
        perror("");
    }
    return files;

}

void saveImage(string output_image_folder, Image *d_image, Image *h_image, Mat *outputMat, string type, string fileName) {
    copyDeviceImageToHost(d_image, h_image);
    convertImageToMat(h_image, outputMat);
    string outPath = output_image_folder + "/" + fileName;
    if (type.length() > 0) {
        outPath = output_image_folder + "/" + type + "-" + fileName;
    }
    printf("writing to %s\n", outPath.c_str());
    imwrite(outPath, *outputMat);//, compression_params);
}

void executeOperations(Json::Value json, string input_image_folder, string output_image_folder, bool saveFinalImages, bool saveIntermediateImages, string extract_channel) {
    vector <string> files = getFileNames(input_image_folder);
    const Json::Value &operations = json["operations"];
    int numOperations = operations.size();
    string curFilePath;
    int k_width;
    int k_height;
    int *kern;
    Mat mat;
    int *h_histogram = nullptr;
    int *d_histogram = nullptr;
    int h_mappings[256];
    cudaMallocHost(&h_histogram, sizeof(int) * 256);
    RGBImage *h_rgbImage = new RGBImage;
    RGBImage *d_rgbImage = new RGBImage;
    Image *d_image = new Image;
    Image *d_tempImage = new Image;
    Image *h_image = new Image;
    Mat *outputMat = new Mat;
//    cudaStream_t *stream = new cudaStream_t;

    Image *h_equalizedImage = new Image;
    bool randomnessSet = false;
    for (int k = 0; k < files.size(); k++) { // iterate through all the images in the folder

        curFilePath = files[k];
        printf("Working on image %s\n", curFilePath.c_str());
        mat = imread(input_image_folder + "/" + curFilePath, CV_LOAD_IMAGE_COLOR);

        convertMatToRGBImage(mat, h_rgbImage);

        // convert image to a single color spectrum
        if (extract_channel == "grey") {
            copyHostRGBImageToDevice(h_rgbImage, d_rgbImage);
            convertRGBToGrayscale(d_rgbImage, d_image, 0);
        } else if (extract_channel == "red") {
            extractSingleColorChannel(h_rgbImage, h_image, 0);
            copyHostImageToDevice(h_image, d_image);
        } else if (extract_channel == "green") {
            extractSingleColorChannel(h_rgbImage, h_image, 1);
            copyHostImageToDevice(h_image, d_image);
        } else if (extract_channel == "blue") {
            extractSingleColorChannel(h_rgbImage, h_image, 2);
            copyHostImageToDevice(h_image, d_image);
        } else {
            printf("Unsupported color option: %s\n", extract_channel.c_str());
            exit(-10);
        }
        if (saveIntermediateImages) {
            saveImage(output_image_folder, d_image, h_image, outputMat, extract_channel, curFilePath);
        }
        if (!randomnessSet) {
            setupRandomness(d_image);
            randomnessSet = true;
        }

        for (int i = 0; i < numOperations; i++) { // perform the operations on each image
            bool supported = true;

            string type = operations[i]["type"].asString();
            if (type == "linear-filter") {
                Json::Value kernel = operations[i]["kernel"];
                k_width = kernel["width"].asInt();
                k_height = kernel["height"].asInt();
                kern = (int *) malloc(sizeof(int) * k_width * k_height);
                readInKernel(kernel, kern, k_width * k_height);
                linearFilter(d_image, d_tempImage, kern, k_width, k_height);
                d_image->image = d_tempImage->image;

                free(kern);
            } else if (type == "median-filter") {
                Json::Value kernel = operations[i]["kernel"];
                k_width = kernel["width"].asInt();
                k_height = kernel["height"].asInt();
                kern = (int *) malloc(sizeof(int) * k_width * k_height);
                readInKernel(kernel, kern, k_width * k_height);
                medianFilter(d_image, d_tempImage, kern, k_width, k_height);
                d_image->image = d_tempImage->image;

                free(kern);
//            } else if (type == "gaussian-noise") {
////                printf("Gaussian Noise\n");
            } else if (type == "salt-and-pepper") {
                int level = operations[i]["intensity"].asInt();
//                cudaStreamSynchronize(*stream);
                saltAndPepperNoise(d_image, d_tempImage, level);
                d_image->image = d_tempImage->image;

            } else if (type == "histogram-equalization") {
                calculateHistogram(d_image, h_histogram, d_histogram);
                equalizeHistogram(h_histogram, h_mappings, d_image->height * d_image->width);
                equalizeImageWithHist(d_image, d_tempImage, h_mappings);
                d_image->image = d_tempImage->image;
            } else if (type == "quantization") {
                const Json::Value &levelsJson = operations[i]["levels"];
                int numLevels = levelsJson.size();
                int *levels = (int *) malloc(sizeof(int) * 3 * numLevels);
                for (int v = 0; v < numLevels; v++) {
//                    k[i] = k_vals[i].asInt();
                    Json::Value levelJson = levelsJson[v];
                    levels[v * 3] = levelJson["min"].asInt();
                    levels[v * 3 + 1] = levelJson["max"].asInt();
                    levels[v * 3 + 2] = levelJson["val"].asInt();
                }
                imageQuantization(d_image, d_tempImage, levels, numLevels);
                d_image->image = d_tempImage->image;
                free(levels);
            } else {
                printf("Unsupported Operation\n");
                supported = false;
            }
            // copy images back to host and save intermediates if configured to do so...
            if (saveIntermediateImages && supported) {
                saveImage(output_image_folder, d_image, h_image, outputMat, type, curFilePath);
            }
            supported = true;
        }
        // copy device image back to host and save it if configured to do so...
        if (saveFinalImages) {
            saveImage(output_image_folder, d_image, h_image, outputMat, "", curFilePath);
        }
    }
}


int main(int argc, char *argv[]) {
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    Json::Value json;
    std::ifstream config("/home/jzheadley/CLionProjects/imageAnalysis/test/input.json", std::ifstream::binary);
    config>>json;
    string input_image_folder = json["image_folder"].asString();
    string output_image_folder = json["output_dir"].asString();
    string extract_channel = json["extract_channel"].asString();

    bool saveFinalImages = json["saveFinalImages"].asBool();
    bool saveIntermediateImages = json["saveIntermediateImages"].asBool();
    printf("Input: %s\nOutput: %s\nSaving intermediates: %s\nSaving Finals: %s\n",
           input_image_folder.c_str(),
           output_image_folder.c_str(),
           saveIntermediateImages ? "true" : "false",
           saveFinalImages ? "true" : "false");
//    testing();
    executeOperations(json, input_image_folder, output_image_folder, saveFinalImages, saveIntermediateImages, extract_channel);


    return 0;
}

