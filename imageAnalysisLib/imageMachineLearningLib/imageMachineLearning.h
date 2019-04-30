#ifndef IMAGE_MACHINE_LEARNING_LIBRARY_H
#define IMAGE_MACHINE_LEARNING_LIBRARY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../imageAnalysis.h"
/*
 * The majority of the knn code comes from Dr. Cano's class.  So yay, thanks for asking for something I have code for.
 * Still needs major modifications to classify more than a single instance at a time but hey I've got a massive start because of Cano
 */
void knn(int numTrain, int numTest, float *train, float *test, int numAttributes, int k);


#endif