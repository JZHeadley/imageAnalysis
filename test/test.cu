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
#include <regex>

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>

using namespace std;
using namespace cv;
#define LOGLEVEL 5
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

void readInKernel(Json::Value kernel, float *k, int numValues) {
    const Json::Value &k_vals = kernel["values"];
    assert(numValues == k_vals.size());
    for (int i = 0; i < numValues; i++) {
        k[i] = k_vals[i].asFloat();
    }
}

void readInKernel(Json::Value kernel, int *k, int numValues) {
    const Json::Value &k_vals = kernel["values"];
    assert(numValues == k_vals.size());
    for (int i = 0; i < numValues; i++) {
        k[i] = k_vals[i].asInt();
    }
}

vector <string> getFileNames(string input_image_folder, regex filter) {
    // adapted from this https://stackoverflow.com/a/612176
    vector <string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_image_folder.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (strncmp(ent->d_name, ".", 1)) {
                if (regex_search(ent->d_name, filter))
                    files.push_back(ent->d_name);
//                printf("%s\n", ent->d_name);
            }
        }
        closedir(dir);
    } else {
        perror("");
    }

    std::sort(files.begin(), files.end());
    return files;

}

void saveImage(string output_image_folder, Image *d_image, Image *h_image, Mat *outputMat, string type, string fileName) {
    copyDeviceImageToHost(d_image, h_image);
    convertImageToMat(h_image, outputMat);
    string outPath = output_image_folder + "/" + fileName;
    if (type.length() > 0) {
        outPath = output_image_folder + "/" + type + "-" + fileName;
    }

    if (LOGLEVEL >= 5)
        printf("writing to %s\n", outPath.c_str());
    imwrite(outPath, *outputMat);//, compression_params);
}

void executeOperations(Json::Value json, string input_image_folder, string output_image_folder, bool saveFinalImages, bool saveIntermediateImages, string extract_channel, regex fileFilter,
                       bool calcMSQEConfig) {
    vector <string> files = getFileNames(input_image_folder, fileFilter);
    const Json::Value &operations = json["operations"];
    int numOperations = operations.size();
    string curFilePath;
    int k_width;
    int k_height;
    float *kern;
    int *medKern;
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
//    types of cells: cyl inter let mod para super svar


//    cudaStream_t *stream = new cudaStream_t;
    cudaEvent_t operationStart, operationStop, batchStart, batchStop;
    cudaEventCreate(&operationStart);
    cudaEventCreate(&operationStop);
    cudaEventCreate(&batchStart);
    cudaEventCreate(&batchStop);
    float milliseconds = 0;
    float totalBatchTime = 0,
            totalGrayscaleTime = 0,
            totalSingleChannelConvertTime = 0,
            totalGaussianNoiseTime = 0,
            totalSaltAndPepperNoiseTime = 0,
            totalHistEqualizationTime = 0,
            totalQuantizationTime = 0,
            totalLinearFilterTime = 0,
            totalAverageFilterTime = 0,
            totalMedianFilterTime = 0;
    float totalMSQE = 0;
    float numImages = files.size();
    Image *h_equalizedImage = new Image;
    bool randomnessSet = false;
    cudaEventRecord(batchStart);

    for (int k = 0; k < files.size(); k++) { // iterate through all the images in the folder
        curFilePath = files[k];
        if (LOGLEVEL >= 4)
            printf("Working on image %s\n", curFilePath.c_str());
        try {
            mat = imread(input_image_folder + "/" + curFilePath, CV_LOAD_IMAGE_COLOR);
            convertMatToRGBImage(mat, h_rgbImage);
            // convert image to a single color spectrum
            if (extract_channel == "grey") {
                cudaEventRecord(operationStart);

                copyHostRGBImageToDevice(h_rgbImage, d_rgbImage);
                convertRGBToGrayscale(d_rgbImage, d_image, 0);

                cudaEventRecord(operationStop);
                cudaEventSynchronize(operationStop);
                cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                totalGrayscaleTime += milliseconds;
            } else if (extract_channel == "red") {
                cudaEventRecord(operationStart);

                extractSingleColorChannel(h_rgbImage, h_image, 0);
                copyHostImageToDevice(h_image, d_image);

                cudaEventRecord(operationStop);
                cudaEventSynchronize(operationStop);
                cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                totalSingleChannelConvertTime += milliseconds;
            } else if (extract_channel == "green") {
                cudaEventRecord(operationStart);

                extractSingleColorChannel(h_rgbImage, h_image, 1);
                copyHostImageToDevice(h_image, d_image);

                cudaEventRecord(operationStop);
                cudaEventSynchronize(operationStop);
                cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                totalSingleChannelConvertTime += milliseconds;
            } else if (extract_channel == "blue") {
                cudaEventRecord(operationStart);

                extractSingleColorChannel(h_rgbImage, h_image, 2);
                copyHostImageToDevice(h_image, d_image);

                cudaEventRecord(operationStop);
                cudaEventSynchronize(operationStop);
                cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                totalSingleChannelConvertTime += milliseconds;
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
                    kern = (float *) malloc(sizeof(float) * k_width * k_height);
                    readInKernel(kernel, kern, k_width * k_height);
                    cudaEventRecord(operationStart);
                    linearFilter(d_image, d_tempImage, kern, k_width, k_height);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalLinearFilterTime += milliseconds;
                    free(kern);
                } else if (type == "average-filter") {
                    Json::Value kernel = operations[i]["kernel"];
                    k_width = kernel["width"].asInt();
                    k_height = kernel["height"].asInt();
                    kern = (float *) malloc(sizeof(float) * k_width * k_height);
                    readInKernel(kernel, kern, k_width * k_height);
                    cudaEventRecord(operationStart);
                    averageFilter(d_image, d_tempImage, kern, k_width, k_height);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalAverageFilterTime += milliseconds;
                    free(kern);
                } else if (type == "median-filter") {
                    Json::Value kernel = operations[i]["kernel"];
                    k_width = kernel["width"].asInt();
                    k_height = kernel["height"].asInt();
                    medKern = (int *) malloc(sizeof(int) * k_width * k_height);
                    readInKernel(kernel, medKern, k_width * k_height);
                    cudaEventRecord(operationStart);
                    medianFilter(d_image, d_tempImage, medKern, k_width, k_height);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalMedianFilterTime += milliseconds;
                    free(medKern);
                } else if (type == "gaussian-noise") {
                    cudaEventRecord(operationStart);
                    float stdDev = operations[i]["std_dev"].asFloat();
                    float mean = operations[i]["mean"].asFloat();
                    addGaussianNoise(d_image, d_tempImage, mean, stdDev);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalGaussianNoiseTime += milliseconds;
                } else if (type == "salt-and-pepper") {
                    int level = operations[i]["intensity"].asInt();
                    cudaEventRecord(operationStart);
                    saltAndPepperNoise(d_image, d_tempImage, level);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalSaltAndPepperNoiseTime += milliseconds;
                } else if (type == "histogram-equalization") {
                    cudaEventRecord(operationStart);
                    calculateHistogram(d_image, h_histogram, d_histogram);
                    equalizeHistogram(h_histogram, h_mappings, d_image->height * d_image->width);
                    equalizeImageWithHist(d_image, d_tempImage, h_mappings);
                    CUDA_CHECK_RETURN(cudaFree(d_image->image));
                    d_image->image = d_tempImage->image;
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalHistEqualizationTime += milliseconds;
                } else if (type == "quantization") {
                    const Json::Value &levelsJson = operations[i]["levels"];
                    int numLevels = levelsJson.size();
                    int *levels = (int *) malloc(sizeof(int) * 3 * numLevels);
                    for (int v = 0; v < numLevels; v++) {
                        Json::Value levelJson = levelsJson[v];
                        levels[v * 3] = levelJson["min"].asInt();
                        levels[v * 3 + 1] = levelJson["max"].asInt();
                        levels[v * 3 + 2] = levelJson["val"].asInt();
                    }
                    cudaEventRecord(operationStart);
                    imageQuantization(d_image, d_tempImage, levels, numLevels);
                    if (calcMSQEConfig) {
                        int MSQE = calcMSQE(d_image, d_tempImage);
                        printf("MSQE of imageQuantization is %i\n", MSQE);
                        totalMSQE += MSQE;
                    }
                    d_image->image = d_tempImage->image;
                    free(levels);
                    cudaEventRecord(operationStop);
                    cudaEventSynchronize(operationStop);
                    cudaEventElapsedTime(&milliseconds, operationStart, operationStop);
                    totalQuantizationTime += milliseconds;
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
            cudaEventRecord(batchStop);
            cudaEventSynchronize(batchStop);
            cudaEventElapsedTime(&milliseconds, batchStart, batchStop);
            totalBatchTime += milliseconds;
        } catch (const std::exception &e) {
            printf("Some sort of issue processing image %s\n", curFilePath.c_str());
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
            }
            numImages--;
            continue;
        }
    }
    printf("\n\nTotal time spent on the entire batch: %0.4f ms average of %0.4f ms for each image\n", totalBatchTime, totalBatchTime / numImages);
    if (extract_channel == "grey") {
        printf("Total time spent converting to grayscale: %0.4f ms average of: %0.4f ms per image\n", totalGrayscaleTime, totalGrayscaleTime / numImages);
    } else {
        printf("Total time spent converting to a single channel: %0.4f ms average of: %0.4f ms per image\n", totalSingleChannelConvertTime, totalSingleChannelConvertTime / numImages);
    }
    printf("Total time spent performing histogram equalization: %0.4f ms average of: %0.4f ms per image\n", totalHistEqualizationTime, totalHistEqualizationTime / numImages);
    printf("Total time spent adding gaussian noise: %0.4f ms average of: %0.4f ms per image\n", totalGaussianNoiseTime, totalGaussianNoiseTime / numImages);
    printf("Total time spent adding salt and pepper noise: %0.4f ms average of: %0.4f ms per image\n", totalSaltAndPepperNoiseTime, totalSaltAndPepperNoiseTime / numImages);
    if (calcMSQEConfig) {
        printf("Total time spent performing image quantization: %0.4f ms average of: %0.4f ms per image with an average MSQE of %0.4f\n", totalQuantizationTime, totalQuantizationTime / numImages,
               totalMSQE / numImages);
    } else {
        printf("Total time spent performing image quantization: %0.4f ms average of: %0.4f ms per image\n", totalQuantizationTime, totalQuantizationTime / numImages);
    }
    printf("Total time spent linear filtering image: %0.4f ms average of: %0.4f ms per image\n", totalLinearFilterTime, totalLinearFilterTime / numImages);
    printf("Total time spent average filtering image: %0.4f ms average of: %0.4f ms per image\n", totalAverageFilterTime, totalAverageFilterTime / numImages);
    printf("Total time spent median filtering image: %0.4f ms average of: %0.4f ms per image\n", totalMedianFilterTime, totalMedianFilterTime / numImages);

}


int main(int argc, char *argv[]) {
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    Json::Value json;
    if (argc < 2) {
        printf("please pass a config file path as the argument\n");
        exit(-1);
    }
    std::ifstream config(argv[1], std::ifstream::binary);
    config>>json;
    string input_image_folder = json["image_folder"].asString();
    string output_image_folder = json["output_dir"].asString();
    string extract_channel = json["extract_channel"].asString();
    regex fileFilter = regex(json["input_image_filter"].asString());
    bool saveFinalImages = json["saveFinalImages"].asBool();
    bool saveIntermediateImages = json["saveIntermediateImages"].asBool();
    bool calcMSQEConfig = json["calc_MSQE"].asBool();
    printf("Input: %s\nOutput: %s\nSaving intermediates: %s\nSaving Finals: %s\n",
           input_image_folder.c_str(),
           output_image_folder.c_str(),
           saveIntermediateImages ? "true" : "false",
           saveFinalImages ? "true" : "false");
    executeOperations(json, input_image_folder, output_image_folder, saveFinalImages, saveIntermediateImages, extract_channel, fileFilter, calcMSQEConfig);


    return 0;
}

