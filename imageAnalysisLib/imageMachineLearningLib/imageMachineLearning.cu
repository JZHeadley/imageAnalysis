#include "imageMachineLearning.h"

#include <math.h>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#define DEBUG 0
using namespace std;

__inline__ __device__ void reduceToK(float *distancesTo, int *indexes, int k, int curSize) {
    // we're just going to do a simple bubble sort and pretend the elements past k don't exist
    float tmp;
    unsigned char idx;
    for (int i = 0; i < curSize - 1; i++) {
        for (int j = 0; j < curSize - i - 1; j++) {
            if (distancesTo[j] > distancesTo[j + 1]) {
                tmp = distancesTo[j];
                idx = indexes[j];
                distancesTo[j] = distancesTo[j + 1];
                indexes[j] = indexes[j + 1];
                distancesTo[j + 1] = tmp;
                indexes[j + 1] = idx;
            }
        }
    }
}

__global__ void computeDistances(int numTrain, int numTest, int numAttributes, float *train, float *test, float *distances) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / numTrain; // instance1Index
    int column = tid - ((tid / numTrain) * numTrain); //instance2Index
    if ((tid < (numTrain * numTest)/* * numInstances*/)) {
        float sum = 0;
        int instance1 = row * numAttributes;
        int instance2 = column * numAttributes;
        for (int atIdx = 0; atIdx < numAttributes - 1; atIdx++) // numAttributes -1 since we don't want to compare class in the distance because that doesn't make sense
        {
            sum += ((test[instance1 + atIdx] - train[instance2 + atIdx]) * (test[instance1 + atIdx] - train[instance2 + atIdx]));
        }
        distances[row * numTrain + column] = (float) sqrt(sum);
    }
}

void printMatrix(float *matrix, int numX, int numY) {
    for (int i = 0; i < numY; i++) {
        for (int j = 0; j < numX; j++) {
            printf("%f ", matrix[i * numX + j]);
        }
        printf("\n");
    }
}


__inline__ __device__ int vote(float *distancesTo, int *indexes, float *dataset, int k, int numAttributes) {
    int classVotes[32]; // can technically parallelize this reading in the class num and probably should come back and do that
    bool duplicate = false;
    int finalClass;
    int mostVotes = 0;
    for (int i = 0; i < 32; i++)
        classVotes[i] = 0;
    for (int i = 0; i < k; i++) {
        int classNum = dataset[indexes[i] * numAttributes + numAttributes - 1];
        classVotes[classNum] += 1;
    }
    for (int i = 0; i < 32; i++) // have to find highest count first
    {
        if (classVotes[i] > mostVotes) {
            finalClass = i;
            mostVotes = classVotes[i];
        }
    }
    for (int i = 0; i < 32; i++) // then compare to that to ensure we don't have duplicates
    {
        if (classVotes[i] == mostVotes && classVotes[i] > 0)
            duplicate = true;
    }
    if (duplicate) {
        if ((k - 1) > 0) // I'm not quite sure why I'm detecting dupes when k=1 but I am soo... this takes care of that and makes everything correct again...
        {
            return vote(distancesTo, indexes, dataset, k - 1, numAttributes);
        }
    }
    return finalClass;
}

__device__ __inline__ void printDistanceInfo(int *indexes, float *distances, int size, int threadId, int blockId) {
    if (DEBUG) {
        if (blockId == 0 && threadId == 0) {
            for (int i = 0; i < size; i++) {
                printf("%i ", indexes[i]);
            }
            printf("\n");
            for (int i = 0; i < size; i++) {
                printf("%f ", distances[i]);
            }
            printf("\n");
        }
    }
}

__global__ void knn(int numTrain, int numAttributes, float *distances, int *predictions, float *train, int k) {
    __shared__ int indexes[256];
    __shared__ float distancesTo[256];
    if (numTrain > blockDim.x) { // need an initial reduction so we can fit the best instances into our shared memory to use to vote
        int instancesPerThread = (numTrain + blockDim.x - 1) / blockDim.x;
        int bestInstanceId = -1;
        float bestDistance = INT_MAX;
        int offset = threadIdx.x * instancesPerThread;
        for (int i = offset; i < min(blockDim.x, offset + instancesPerThread); i++) {

            if (distances[blockIdx.x * blockDim.x + offset] < bestDistance) {
                bestDistance = distances[blockIdx.x * blockDim.x + offset];
                bestInstanceId = offset;
            }
        }
        indexes[threadIdx.x] = bestInstanceId;
        distancesTo[threadIdx.x] = bestDistance;

    } else { // numTrain <= blockDim.x
        indexes[threadIdx.x] = threadIdx.x;
        distancesTo[threadIdx.x] = distances[blockDim.x * blockIdx.x + threadIdx.x];
    }
    __syncthreads(); // get all the threads of the block together again after the initial reduction
    printDistanceInfo(indexes, distancesTo, blockDim.x, threadIdx.x, blockIdx.x);
    if (threadIdx.x < blockDim.x / 2) // only need the first half (128) of the threads to work on the 256 length shared mem arrays
    {
        int s;
        // this for should probably have the conditional of (s>>1) > k but if I do that I don't reduce enough sooo...
        // we're going with this until I find that error and just upping s back up after this for
        for (s = blockDim.x / 2; (s) > k; s >>= 1) {
            if (threadIdx.x < s) {
                if (distancesTo[threadIdx.x + s] < distancesTo[threadIdx.x]) {
                    distancesTo[threadIdx.x] = distancesTo[threadIdx.x + s];
                    indexes[threadIdx.x] = indexes[threadIdx.x + s];
                }
                __syncthreads();
            }
        }
        s *= 2;

        __syncthreads();
        printDistanceInfo(indexes, distancesTo, blockDim.x, threadIdx.x, blockIdx.x);
        if (s > k && threadIdx.x == 1) { // we need to reduce it just a little more
            // remember to change both the indexes and distancesTo arrays
            reduceToK(distancesTo, indexes, k, s);
        }
        __syncthreads();


        printDistanceInfo(indexes, distancesTo, blockDim.x, threadIdx.x, blockIdx.x);

        if (threadIdx.x == 1)
            predictions[blockIdx.x] = vote(distancesTo, indexes, train, k, numAttributes);
        if (DEBUG) {
            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                for (int i = 0; i < gridDim.x; i++) {
                    printf("%i ", predictions[i]);
                }
                printf("\n");
            }
        }
    }
}

void knn(int numTrain, int numTest, float *h_train, float *h_test, int numAttributes, int *h_predictions, int k) {
    int NUM_STREAMS = 2;
    cudaStream_t *streams = (cudaStream_t *) malloc(NUM_STREAMS * sizeof(cudaStream_t));
    for (int i = 0; i < NUM_STREAMS; i++) // multiple streams
        cudaStreamCreate(&streams[i]);

    float *h_distances;
    float *d_train, *d_test, *d_distances;
    int *d_predictions;

    int threadsPerBlock = min(numTrain * numTest, 256);
    int blocksPerGrid = ((numTrain * numTest) + threadsPerBlock - 1) / threadsPerBlock;
    cudaMallocHost(&h_predictions, sizeof(int) * numTrain);
//    cudaMallocHost(&h_distances, sizeof(float) * numTrain * numTest);

//    cudaMallocHost(&d_predictions, sizeof(int) * numTest);
//    cudaMallocHost(&d_train, sizeof(float) * numTrain * numAttributes);
//    cudaMallocHost(&d_test, sizeof(float) * numTest * numAttributes);
//    cudaMallocHost(&d_distances, sizeof(float) * numTrain * numTest);
//TODO: figure out why the hell I mallocd on the host here...

    cudaMalloc(&d_predictions, sizeof(int) * numTest);
    cudaMalloc(&d_train, sizeof(float) * numTrain * numAttributes);
    cudaMalloc(&d_test, sizeof(float) * numTest * numAttributes);
    cudaMalloc(&d_distances, sizeof(float) * numTrain * numTest);
    cudaMemcpy(d_train, h_train, numTrain * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, numTest * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpyAsync(d_distances, h_distances, numTriangularSpaces * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    computeDistances<< < blocksPerGrid, threadsPerBlock, 0, streams[0]>> > (numTrain, numTest, numAttributes, d_train, d_test, d_distances);
//    cudaMemcpyAsync(h_distances, d_distances, numTest * numTrain * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
//    printMatrix(h_distances, numTrain, numTest);
    knn<< < numTest, min(numTrain, 256), 0, streams[0]>> > (numTrain, numAttributes, d_distances, d_predictions, d_train, k);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(h_predictions, d_predictions, numTest * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
    if (DEBUG) {
        printf("host predictions:\t");
        for (int i = 0; i < numTest; i++) {
            printf("%i ", h_predictions[i]);
        }
        printf("\n");
    }
    cudaFree(d_predictions);
    cudaFree(d_train);
    cudaFree(d_test);
    cudaFree(d_distances);
//    cudaFreeHost(d_predictions);
//    cudaFreeHost(d_train);
//    cudaFreeHost(d_test);
//    cudaFreeHost(d_distances);
//    cudaFreeHost(h_distances);
//    cudaFreeHost(h_predictions);
    free(streams);
}

typedef struct {
    int threadId;
    int k;
    float *dataset;
    int numInstances;
    int numAttributes;
} ValidationArgs;

double computeAccuracy(int *predictions, float *test, int numInstances, int numAttributes) {
    int numCorrect = 0;
    int numIncorrect = 0;
    for (int i = 0; i < numInstances; i++) {
        if (predictions[i] == (int) test[i * numAttributes + numAttributes - 1]) {
            numCorrect++;
        } else {
//            printf("%f is not %i\n", test[i * numAttributes + numAttributes-1], predictions[i]);
            numIncorrect++;
        }
    }
    printf("numCorrect: %i numIncorrect: %i\n", numCorrect, numIncorrect);
    printf("accuracy was %f\n", (numCorrect / (double) numInstances));
    return (numCorrect / (double) numInstances);
}

void *knnThreadValidation(void *args) {
    ValidationArgs *valArgs = (ValidationArgs *) args;
    int threadId = valArgs->threadId;
    int k = valArgs->k;
    float *originalDataset = valArgs->dataset;
    int numInstances = valArgs->numInstances;
    int numAttributes = valArgs->numAttributes;

//    int instancesPerTask = ceil(numInstances / 10);
//    printf("thread %i numInstances %i instancesPerThread %i\n", threadId, numInstances, instancesPerTask);
    vector<float> datasetVec(originalDataset, originalDataset + numInstances * numAttributes);
//    printf("datasetSize =%i, vecSize is %i\n", numInstances * numAttributes, datasetVec.size());
    int numToRotate = (numInstances * .1);
//    printf("thread %i rotating by %i\n", threadId, (numToRotate * numAttributes * threadId));

    rotate(datasetVec.begin(), datasetVec.begin() + (numToRotate * numAttributes * threadId), datasetVec.end());
    double *result = (double *) malloc(sizeof(double));
    float *dataset = &datasetVec[0];
    int trainSize = (datasetVec.size() - (numAttributes * numToRotate)) / numAttributes;
    int *predictions = (int *) malloc(sizeof(int) * numToRotate);
//    printf("train size is %i testSize is %i\n", trainSize, numToRotate);
    knn(trainSize, numToRotate, dataset, dataset + (trainSize), numAttributes, predictions, k);

    double accuracy = computeAccuracy(predictions, dataset + (trainSize), numToRotate, numAttributes);
    *result = accuracy;
    return result;
}

float knnTenfoldCrossVal(float *dataset, int numInstances, int numAttributes, int k) {
/*
 * Should be able to do each validation in parallel... which would be wonderful because cuda + more parallel = awesome
 */
    int NUM_THREADS = 10;
    pthread_t *threads = (pthread_t *) malloc(NUM_THREADS * sizeof(pthread_t));
    int *threadIds = (int *) malloc(NUM_THREADS * sizeof(int));
    // shuffling the dataset so we don't cut off huge chunks of classes from the rotation below
    vector<float> datasetVec(dataset, dataset + numInstances * numAttributes);
//    random_shuffle(datasetVec.begin(), datasetVec.end());
//    dataset = &datasetVec[0];

    for (int i = 0; i < NUM_THREADS; i++)
        threadIds[i] = i;

    for (int i = 0; i < NUM_THREADS; i++) {
        ValidationArgs *args = new ValidationArgs;
        args->threadId = threadIds[i];
        args->k = k;
        args->dataset = dataset;
        args->numInstances = numInstances;
        args->numAttributes = numAttributes;
        int status = pthread_create(&threads[i], NULL, knnThreadValidation, (void *) args);

    }
    double totalAccuracy = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        void *thread_result;
        int err = pthread_join(threads[i], &thread_result);

        double threadAccuracy = *(double *) thread_result;
        printf("thread %i had accuracy of %f\n", i, threadAccuracy);
        free(thread_result);
        totalAccuracy += threadAccuracy;
    }
    printf("Average accuracy was %f\n", totalAccuracy / 10.0);
    return totalAccuracy / 10.0;
}

void calcMode(int *d_histogram, int *mode) {
    thrust::device_ptr<int> dp = thrust::device_pointer_cast(d_histogram);
    thrust::device_vector<int> thrust_hist(dp, dp + 255);
    thrust::device_vector<int>::iterator iter =
            thrust::max_element(thrust_hist.begin(), thrust_hist.end());
    int position = iter - thrust_hist.begin();
    *mode = position;
}

int calculateCellArea(int *histogram, int numPixels) {
    int area = 0;
    // I make the assumption that the foreground is always the minority here
    // because sometimes I get a white foreground and black background others
    // I get a black foreground and white background.  Not quite sure why.
    area = min(histogram[255], numPixels - histogram[255]);
    return area;
}

int calculateCellPerimeter(int *histogram) {
    int perimeter = 0;
    perimeter = histogram[255];
    return perimeter;
}

vector<float> featureExtraction(Image *image, Image *tempImage, int *h_histogram, int *d_histogram) {
    vector<float> features;
    float mean, stdDev;
    calculateMeanAndStdDev(image, &mean, &stdDev);
    Image *tempImage2 = new Image;
    otsuThresholdImage(image, tempImage);
    calculateHistogram(tempImage, h_histogram, d_histogram);
    int totalCellArea = calculateCellArea(h_histogram, image->height * image->width);
    compassFilter(tempImage, tempImage2);
    calculateHistogram(tempImage2, h_histogram, d_histogram);
    int totalCellPerimeter = calculateCellPerimeter(h_histogram);


    //    int mode;
//    calcMode(d_histogram, &mode);
    features.push_back(mean);
    features.push_back(stdDev);
    features.push_back(totalCellArea);
    features.push_back(totalCellPerimeter);
//    features.push_back(mode);

    CUDA_CHECK_RETURN(cudaFree(tempImage2->image))
    CUDA_CHECK_RETURN(cudaFree(tempImage->image))
    return features;
}