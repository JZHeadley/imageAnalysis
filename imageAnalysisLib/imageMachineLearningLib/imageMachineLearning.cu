#include "imageMachineLearning.h"

#include <math.h>

//TODO: need to convert this to having a train and test set.
//__global__ void computeDistances(int numInstances, int numAttributes, float *dataset, float *distances) {
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    int row = tid / numInstances; // instance1Index
//    int column = tid - ((tid / numInstances) * numInstances); //instance2Index
//    if ((tid < numInstances /* * numInstances*/)) {
//        float sum = 0;
//        int instance1 = row * numAttributes;
//        int instance2 = column * numAttributes;
//        for (int atIdx = 0; atIdx < numAttributes - 1; atIdx++) // numAttributes -1 since we don't want to compare class in the distance because that doesn't make sense
//        {
//            sum += ((dataset[instance1 + atIdx] - dataset[instance2 + atIdx]) * (dataset[instance1 + atIdx] - dataset[instance2 + atIdx]));
//        }
//        distances[row * numInstances + column] = (float) sqrt(sum);
//        distances[column * numInstances + row] = distances[row * numInstances + column]; //set the distance for the other half of the pair we just computed
//    }
//}
//
//__inline__ __device__ void reduceToK(float *distancesTo, int *indexes, int k, int curSize) {
//    // we're just going to do a simple bubble sort and pretend the elements past k don't exist
//    float tmp;
//    unsigned char idx;
//    for (int i = 0; i < curSize - 1; i++) {
//        for (int j = 0; j < curSize - i - 1; j++) {
//            if (distancesTo[j] > distancesTo[j + 1]) {
//                tmp = distancesTo[j];
//                idx = indexes[j];
//                distancesTo[j] = distancesTo[j + 1];
//                indexes[j] = indexes[j + 1];
//                distancesTo[j + 1] = tmp;
//                indexes[j + 1] = idx;
//            }
//        }
//    }
//}
//
//__inline__ __device__ int vote(float *distancesTo, int *indexes, float *dataset, int k, int numAttributes) {
//    int classVotes[32]; // can technically parallelize this reading in the class num and probably should come back and do that
//    bool duplicate = false;
//    int finalClass;
//    int mostVotes = 0;
//    for (int i = 0; i < 32; i++)
//        classVotes[i] = 0;
//    for (int i = 0; i < k; i++) {
//        int classNum = dataset[indexes[i] * numAttributes + numAttributes - 1];
//        classVotes[classNum] += 1;
//    }
//    for (int i = 0; i < 32; i++) // have to find highest count first
//    {
//        if (classVotes[i] > mostVotes) {
//            finalClass = i;
//            mostVotes = classVotes[i];
//        }
//    }
//    for (int i = 0; i < 32; i++) // then compare to that to ensure we don't have duplicates
//    {
//        if (classVotes[i] == mostVotes && classVotes[i] > 0)
//            duplicate = true;
//    }
//    if (duplicate) {
//        if ((k - 1) > 0) // I'm not quite sure why I'm detecting dupes when k=1 but I am soo... this takes care of that and makes everything correct again...
//        {
//            return vote(distancesTo, indexes, dataset, k - 1, numAttributes);
//        }
//    }
//    return finalClass;
//}
//
//__global__ void knn(int *predictions, float *distances, float *dataset, int numAttributes, int k) {
//    __shared__ int indexes[256];
//    __shared__ float distancesTo[256];
//    // gridDim.x is numInstances
//    int bestInstanceId;
//    float bestDistance = INT_MAX;
//    int instanceFrom = blockIdx.x * gridDim.x;
//    int distancePos;
//    int rowBoundary = instanceFrom + gridDim.x - 1;
//    if (blockDim.x < gridDim.x) { //If we have more elements than threads we need to do an inital reduction to fit into our shared mem
//        if (threadIdx.x < blockDim.x) // only want 256 threads to come into this otherwise we will go out of bounds of our shared mem
//        {
//            for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) // will try to make this more coalesced later
//            {
//                if (i == blockIdx.x) // don't need to include the diagonal
//                    continue;
//
//                distancePos = instanceFrom + i;
//                if (distancePos > rowBoundary) { // should take care of the final elements
//                    break;
//                }
//                if (distances[distancePos] < bestDistance) {
//                    bestDistance = distances[distancePos];
//                    bestInstanceId = i;
//                }
//            }
//            indexes[threadIdx.x] = bestInstanceId;
//            distancesTo[threadIdx.x] = bestDistance;
//        }
//        __syncthreads();
//
//        if (threadIdx.x < blockDim.x / 2) // only need the first half(128) of the threads to work on the 256 length shared mem arrays
//        {
//            int s;
//            // this for should probably have the conditional of (s>>1) > k but if I do that I don't reduce enough sooo...
//            // we're going with this until I find that error and just upping s back up after this for
//            for (s = blockDim.x / 2; (s) > k; s >>= 1) {
//                if (threadIdx.x < s) {
//                    if (distancesTo[threadIdx.x + s] < distancesTo[threadIdx.x]) {
//                        distancesTo[threadIdx.x] = distancesTo[threadIdx.x + s];
//                        indexes[threadIdx.x] = indexes[threadIdx.x + s];
//                    }
//                    __syncthreads();
//                }
//            }
//            s *= 2;
//            __syncthreads();
//            if (s > k && threadIdx.x == 1) { // we need to reduce it just a little more
//                // remember to change both the indexes and distancesTo arrays
//                reduceToK(distancesTo, indexes, k, s);
//            }
//            __syncthreads();
//            if (threadIdx.x == 1)
//                predictions[blockIdx.x] = vote(distancesTo, indexes, dataset, k, numAttributes);
//        }
//    }
//}

__global__ void computeDistances(int numTrain, int numTest, int numAttributes, float *train, float *test, float *distances) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / numTrain; // instance1Index
    int column = tid - ((tid / numTest) * numTest); //instance2Index
    if ((tid < (numTrain * numTest)/* * numInstances*/)) {
        float sum = 0;
        int instance1 = row * numAttributes;
        int instance2 = column * numAttributes;
        for (int atIdx = 0; atIdx < numAttributes - 1; atIdx++) // numAttributes -1 since we don't want to compare class in the distance because that doesn't make sense
        {
            sum += ((train[instance1 + atIdx] - test[instance2 + atIdx]) * (train[instance1 + atIdx] - test[instance2 + atIdx]));
        }
        distances[row * numTrain + column] = (float) sqrt(sum);
        distances[column * numTest + row] = distances[row * numTrain + column]; //set the distance for the other half of the pair we just computed
    }
}

void printMatrix(float *matrix, int numX, int numY) {

    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            printf("%f ", matrix[i * numX + j]);
        }
        printf("\n");
    }
}

void knn(int numTrain, int numTest, float *h_train, float *h_test, int numAttributes, int k) {
    int NUM_STREAMS = numTest;
    cudaStream_t *streams = (cudaStream_t *) malloc(NUM_STREAMS * sizeof(cudaStream_t));
    for (int i = 0; i < NUM_STREAMS; i++) // multiple streams
        cudaStreamCreate(&streams[i]);

    int *h_predictions;
    float /* *h_train, *h_test,*/ *h_distances;
    float *d_train, *d_test, *d_distances;
    int *d_predictions;

    int threadsPerBlock = 256;
    int blocksPerGrid = ((numTrain * numTest) + threadsPerBlock - 1) / threadsPerBlock;

    cudaMallocHost(&h_predictions, sizeof(int) * numTrain);
//    cudaMallocHost(&h_train, sizeof(float) * numTrain * numAttributes);
//    cudaMallocHost(&h_test, sizeof(float) * numTest * numAttributes);
    cudaMallocHost(&h_distances, sizeof(float) * numTrain * numTest);

    cudaMallocHost(&d_predictions, sizeof(int) * numTrain);
    cudaMallocHost(&d_train, sizeof(float) * numTrain * numAttributes);
    cudaMallocHost(&d_test, sizeof(float) * numTest * numAttributes);
    cudaMallocHost(&d_distances, sizeof(float) * numTrain * numTest);
    cudaMemcpy(d_train, h_train, numTrain * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, numTest * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpyAsync(d_distances, h_distances, numTriangularSpaces * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    computeDistances<< < blocksPerGrid, threadsPerBlock, 0, streams[0]>> > (numTrain, numTest, numAttributes, d_train, d_test, d_distances);
    cudaMemcpyAsync(h_distances, d_distances, numTest * numTrain * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    printMatrix(h_distances, numTrain, numTest);
//    knn();
}