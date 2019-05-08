#ifndef IMAGE_MACHINE_LEARNING_LIBRARY_H
#define IMAGE_MACHINE_LEARNING_LIBRARY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../imageAnalysis.h"
#include <vector>

using namespace std;

/*
 * The majority of the knn code comes from Dr. Cano's class.  So yay, thanks for asking for something I have code for.
 * Still needs major modifications to classify more than a single instance at a time but hey I've got a massive start because of Cano
 */
void knn(int numTrain, int numTest, float *train, float *test, int numAttributes, int *h_predictions, int k);

/*
 * This is kinda annoying to do...
 */
float knnTenfoldCrossVal(float *dataset, int numInstances, int numAttributes, int k);

vector<float> featureExtraction(Image *image, Image *tempImage, int *h_histogram, int *d_histogram);

vector<float> extractAllPixelsAsFeatures(Image *image);

#endif