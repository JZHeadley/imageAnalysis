#include "imageMachineLearning.h"

#include <math.h>

#define DEBUG 1

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
        printf("more train than 256");
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
//        if (DEBUG) {
//            __syncthreads();
//            if (threadIdx.x == 0 && blockIdx.x == 0) {
//                for (int i = 0; i < gridDim.x; i++) {
//                    printf("%i\n", predictions[i]);
//                }
//            }
//        }
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
    cudaMallocHost(&h_distances, sizeof(float) * numTrain * numTest);

    cudaMallocHost(&d_predictions, sizeof(int) * numTest);
    cudaMallocHost(&d_train, sizeof(float) * numTrain * numAttributes);
    cudaMallocHost(&d_test, sizeof(float) * numTest * numAttributes);
    cudaMallocHost(&d_distances, sizeof(float) * numTrain * numTest);
    cudaMemcpy(d_train, h_train, numTrain * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, numTest * numAttributes * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpyAsync(d_distances, h_distances, numTriangularSpaces * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    computeDistances<< < blocksPerGrid, threadsPerBlock, 0, streams[0]>> > (numTrain, numTest, numAttributes, d_train, d_test, d_distances);
    cudaMemcpyAsync(h_distances, d_distances, numTest * numTrain * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
//    printMatrix(h_distances, numTrain, numTest);
    knn<< < numTest, min(numTrain, 256), 0, streams[0]>> > (numTrain, numAttributes, d_distances, d_predictions, d_train, k);

    cudaMemcpyAsync(h_predictions, d_predictions, numTest * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

}

typedef struct {
    int threadId;
    int k;
    float *dataset;
    int numInstances;
    int numAttributes;
} ValidationArgs;

void *knnThreadValidation(void *args) {
    ValidationArgs *valArgs = (ValidationArgs *) args;
    int threadId = valArgs->threadId;
    int k = valArgs->k;
    float *dataset = valArgs->dataset;
    int numInstances = valArgs->numInstances;
    int numAttributes = valArgs->numAttributes;
    double *result = (double *) malloc(sizeof(double));
    printf("Thread id %i is here\n", threadId);

    *result = 42.0;

    return result;
}

void knnTenfoldCrossVal(float *dataset, int numInstances, int numAttributes, int k) {
/*
 * Should be able to do each validation in parallel... which would be wonderful because cuda + more parallel = awesome
 */
    int NUM_THREADS = 10;
    pthread_t *threads = (pthread_t *) malloc(NUM_THREADS * sizeof(pthread_t));
    int *threadIds = (int *) malloc(NUM_THREADS * sizeof(int));

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
        free(thread_result);
        totalAccuracy += threadAccuracy;
    }
    printf("Average accuracy was %f\n", totalAccuracy / 10);
}