#include "imageAnalysis.h"


#include <math.h>

#include <omp.h>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>


curandState *d_states;

__global__ void convertRGBToGrayscaleLuminance(unsigned char *image, int width, int numPixels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((.21 * image[row * width + column]) + (.72 * image[row * width + column + numPixels]) + (.07 * image[row * width + column + (2 * numPixels)]));
//        printf("%i %i\n", row, column);
    }
    return;
}

__global__ void convertRGBToGrayscaleAverage(unsigned char *image, int width, int numPixels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((image[row * width + column] + image[row * width + column + numPixels] + image[row * width + column + (2 * numPixels)]) / 3);
//        printf("%i %i\n", row, column);
    }
    return;
}

int random(int min, int max) {
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void convertRGBToGrayscale(RGBImage *d_rgb, Image *d_gray, int method) {
    /*
    don't think you mentioned a grayscale conversion method so I looked it up and used this page as my guide
    https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    */
    int totalPixels = d_rgb->width * d_rgb->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &(d_gray->image), (int) sizeof(unsigned char) * d_rgb->width * d_rgb->height));
    switch (method) {
        case 0:
            // luminance method
            convertRGBToGrayscaleLuminance<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, totalPixels, d_gray->image);
            d_gray->width = d_rgb->width;
            d_gray->height = d_rgb->height;
            break;
        case 1:
            // average method

            convertRGBToGrayscaleAverage<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, totalPixels, d_gray->image);
            d_gray->width = d_rgb->width;
            d_gray->height = d_rgb->height;
            break;
        default:
            printf("WTF why are we defaulting?\n");
            break;
    }
}

void extractSingleColorChannel(RGBImage *rgb, Image *out, int color) {
    out->width = rgb->width;
    out->height = rgb->height;
    int totalPixels = rgb->width * rgb->height;
    //TODO: Memory leaks right here probably should fix but meh it should work well enough like this...
    switch (color) {
        case 0: // red
            out->image = rgb->image;
            break;
        case 1: // green
            out->image = rgb->image + totalPixels;

            break;
        case 2: // blue
            out->image = rgb->image + (2 * totalPixels);
            break;
        default:
            printf("invalid option\n");
            break;
    }
}

// this method of calculating mean and variance with thrust came from here https://stackoverflow.com/a/41862431
// I tried to write a summation reduction myself (see the function above) and something just wouldn't work so I just decided to use this
struct varianceshifteop : std::unary_function<float, float> {
    varianceshifteop(float m) : mean(m) { /* no-op */ }

    const float mean;

    __device__ float operator()(float data) const {
        return ::pow(data - mean, 2.0f);
    }
};

void calculateMeanAndStdDev(Image *image, float *mean, float *stdDev) {
    int totalPixels = image->width * image->height;
//    printf("%i %i %i %i %i\n", image->width, image->height, totalPixels, threadsPerBlock, blocksPerGrid);
    thrust::device_ptr<unsigned char> dp = thrust::device_pointer_cast(image->image);
    thrust::device_vector<unsigned char> thrust_image_d(dp, dp + totalPixels);
    int sum = thrust::reduce(
            thrust_image_d.cbegin(),
            thrust_image_d.cend(),
            0.0f,
            thrust::plus<float>());
    float meanCalc = sum / (float) totalPixels;

    float variance = thrust::transform_reduce(
            thrust_image_d.cbegin(),
            thrust_image_d.cend(),
            varianceshifteop(meanCalc),
            0.0f,
            thrust::plus<float>()) / (thrust_image_d.size() - 1);
    float stdv = sqrt(variance);
//    printf("%i is the sum of the pixels and %f is the mean variance is %f and stdDev is %f  \n", sum, meanCalc, variance, stdv);
    *mean = meanCalc;
    *stdDev = stdv;
}

__global__ void calcHistogram(unsigned char *data, int width, int numPixels, int *histogram) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if (tid < numPixels) {
        int val = data[row * width + column];
        if (val != 0)
            atomicAdd(&histogram[val], 1);
    }
    return;
}

void calculateHistogram(Image *image, int *h_histogram, int *d_histogram) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram, (int) sizeof(int) * 256));
    CUDA_CHECK_RETURN(cudaMemset(d_histogram, 0, 256 * sizeof(int)));

    calcHistogram<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, image->width, totalPixels, d_histogram);

    CUDA_CHECK_RETURN(cudaMemcpy(h_histogram, d_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost));
}


void equalizeHistogram(int *original, int *mappings, int numPixels) {
//#pragma omp parallel num_threads(2)
    int numColors = 256;

    float pdf[256];
    float cdf[256];
    // tried to use openmp and speed this up more but something is weird with openmp + cuda + cmake and it only ever ran on 1 thread for me
//#pragma omp parallel for default (none) shared(numColors, original, pdf, cdf)
    for (int i = 0; i < numColors; i++) {
//        int threadId = omp_get_thread_num();
//        printf("Thread %i reporting for %i\n", omp_get_thread_num(), i);
        pdf[i] = original[i] / (float) numPixels;
        cdf[i] = pdf[i];
        if (i > 0) {
            cdf[i] = cdf[i] + cdf[i - 1];
            mappings[i] = (int) (cdf[i] * 255);
        } else {
            mappings[i] = (int) (cdf[i] * 255);
        }
    }
//#pragma omp parallel end
}

__global__ void equalizeImage(unsigned char *image, int width, int numPixels, int *mappings, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) (mappings[image[row * width + column]]);
    }
    return;
}

void equalizeImageWithHist(Image *image, Image *d_equalizedImage, int *h_mappings) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    d_equalizedImage->width = image->width;
    d_equalizedImage->height = image->height;
    int *d_mappings;
    CUDA_CHECK_RETURN(cudaMalloc(&d_mappings, (int) sizeof(int) * 256));
    CUDA_CHECK_RETURN(cudaMemcpy(d_mappings, h_mappings, sizeof(int) * 256, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc(&(d_equalizedImage->image), sizeof(unsigned char) * image->width * image->height));
    equalizeImage<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, image->width, totalPixels, d_mappings, d_equalizedImage->image);
    CUDA_CHECK_RETURN(cudaFree(d_mappings));


}

__global__ void dilateImage(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, float *structuringElement, int kWidth, int kHeight) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < totalPixels)) {
        int k = 0;
        int aboveBelow = (kHeight - 1) / 2;
        int sideToSide = (kWidth - 1) / 2;
        if (row < aboveBelow || row > (height - aboveBelow) || column < sideToSide || column > (width - sideToSide)) {
            output[row * width + column] = 0; // image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else if (image[tid] > 0) {
            for (int i = row - aboveBelow; i <= row + aboveBelow; i++) {
                for (int j = column - sideToSide; j <= column + sideToSide; j++) {
                    output[i * width + j] = (unsigned char) (structuringElement[k] ? 255 : 0);
                    k++;
                }
            }
        } else {
            output[tid] = image[tid];
        }

    }
    return;
}

/**
 * Returns an image with all pixels above the threshold as white and everything else as black
 */
__global__ void thresholding(unsigned char *image, unsigned char *out, int numPixels, int threshold) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numPixels) {
        if (image[tid] > threshold)
            out[tid] = 255;
        else
            out[tid] = 0;
    }
    return;
}

void thresholdImage(Image *image, Image *output, int threshold) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    thresholding<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, totalPixels, threshold);
}

/**
 * adapted from code and concepts found here...
 * http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
 * I'm not trying to plagiarize ... please don't try and get me for it...
 * I used large pieces of their code because I found no better way to do this in a speedy way
 * I tried to come up with a much better cuda implementation but couldn't come up with one
 * that wouldn't take a week to implement and debug so I fell back to calculating sum with thrush
 * and using their general way of finding the threshold because its just so efficient
 */
void otsuThresholdImage(Image *image, Image *output) {
    int totalPixels = image->width * image->height;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))

    int h_histogram[256];
    int *d_histogram;
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram, sizeof(int) * 256))
    calculateHistogram(image, h_histogram, d_histogram);

    thrust::device_ptr<unsigned char> dp = thrust::device_pointer_cast(image->image);
    thrust::device_vector<unsigned char> thrust_image_d(dp, dp + totalPixels);
    int sum = thrust::reduce(
            thrust_image_d.cbegin(),
            thrust_image_d.cend(),
            0.0f,
            thrust::plus<float>());
    float varMax = 0;
    int threshold = 0;
    int backgroundSum = 0;
    int numForegroundPixels = 0;
    int numBackgroundPixels = 0;
    for (int i = 0; i < 256; i++) {
        numBackgroundPixels += h_histogram[i];
        if (numBackgroundPixels == 0)
            continue;
        numForegroundPixels = totalPixels - numBackgroundPixels;
        if (numForegroundPixels == 0)
            continue;

        backgroundSum += i * h_histogram[i];
        float backgroundMean = backgroundSum / numBackgroundPixels;
        float foregroundMean = (sum - backgroundSum) / numForegroundPixels;
        float betweenClassVariance = (float) numBackgroundPixels * (float) numForegroundPixels * (backgroundMean - foregroundMean) * (backgroundMean - foregroundMean);

        if (betweenClassVariance > varMax) {
            varMax = betweenClassVariance;
            threshold = i;
        }
    }
    thresholdImage(image, output, threshold);
}

bool centroidsHaveChanged(unsigned char *centroids, int *count, int k) {
    bool changed = false;
    for (int i = 0; i < k; i++) {
        if ((centroids[i] ^ centroids[i + k]) != 0) {
            printf("Centroids changed on iteration %i from %i to %i\n", *count, centroids[i], centroids[i] + k);
            centroids[i] = centroids[i + k];
            changed = true;

        }
    }
    return changed;
}

__global__ void kMeans(unsigned char *image, unsigned char *labels, unsigned char *centroids, int totalPixels, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < totalPixels) {
        unsigned char bestCentroid = 0;
        int bestDist = abs(centroids[0] - image[tid]);
//        for (int i = 1; i < k; i++) {
//            int dist = abs(centroids[i] - image[tid]);
//            if (dist < bestDist) {
//                bestDist = dist;
//                bestCentroid = i;
//            }
//        }
//        labels[tid] = bestCentroid;

    }
}

void calculateCentroidPositions(unsigned char *h_centroids, unsigned char *d_labels, unsigned char *d_image, int k, int totalPixels) {
    thrust::device_ptr<unsigned char> dp_image = thrust::device_pointer_cast(d_image);
    thrust::device_ptr<unsigned char> dp_labels = thrust::device_pointer_cast(d_labels);
    thrust::sort_by_key(dp_labels, dp_labels + totalPixels, dp_image);
    unsigned char *keysOut;
    CUDA_CHECK_RETURN(cudaMalloc(&keysOut, sizeof(unsigned char) * totalPixels))
    unsigned char *valsOut;
    CUDA_CHECK_RETURN(cudaMalloc(&valsOut, sizeof(unsigned char) * totalPixels))
    int offset = 0;
    for (int i = 0; i < k; ++i) {
        int numClass = thrust::count(dp_labels, dp_labels + totalPixels, i);
        int sum = thrust::reduce(dp_image, dp_image + offset + numClass);
        float average = sum / (float) numClass;
        h_centroids[i + k] = (unsigned char) floor(average);
        offset += numClass;
    }

    CUDA_CHECK_RETURN(cudaFree(keysOut))
    CUDA_CHECK_RETURN(cudaFree(valsOut))
}

__global__ void applyLabels(unsigned char *image, unsigned char *labels, int totalPixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < totalPixels) {
        if (labels[tid] == 0)
            image[tid] = 0;
        else
            image[tid] = 255;
    }
}

void kMeansThresholding(Image *image, Image *output) {
    int totalPixels = image->width * image->height;
    int k = 2;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    // going to use 1 array to represent old and new values of the centroid so I don't have to flip back and forth between variables somehow
    unsigned char *h_centroids = (unsigned char *) malloc(sizeof(unsigned char) * k * 2);
    for (int i = 0; i < k; i++) {
        h_centroids[i] = random(0, 256);
    }
    unsigned char *d_centroids;
    CUDA_CHECK_RETURN(cudaMalloc(&d_centroids, sizeof(unsigned char) * k * 2))
    unsigned char *d_labels;
    CUDA_CHECK_RETURN(cudaMalloc(&d_labels, sizeof(unsigned char) * totalPixels))
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    int count = 0;
    do {
        CUDA_CHECK_RETURN(cudaMemcpy(d_centroids, h_centroids, sizeof(unsigned char) * k * 2, cudaMemcpyHostToDevice));
        kMeans<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, d_labels, d_centroids, totalPixels, k);

        calculateCentroidPositions(h_centroids, d_labels, image->image, k, totalPixels);
        count++;
    } while (centroidsHaveChanged(h_centroids, &count, k) && count < 1000);
    applyLabels<< < threadsPerBlock, blocksPerGrid, 0>> > (output->image, d_labels, totalPixels);
    CUDA_CHECK_RETURN(cudaFree(d_centroids))
}

void imageDilation(Image *image, Image *output, int *structuringElement, int kWidth, int kHeight) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    float *d_structuringElement;
    CUDA_CHECK_RETURN(cudaMalloc(&d_structuringElement, sizeof(float) * kWidth * kHeight))
    CUDA_CHECK_RETURN(cudaMemcpy(d_structuringElement, structuringElement, sizeof(float) * kWidth * kHeight, cudaMemcpyHostToDevice))
    dilateImage<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_structuringElement, kWidth, kHeight);
    CUDA_CHECK_RETURN(cudaFree(d_structuringElement))

    return;
}

__global__ void erodeImage(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, float *structuringElement, int kWidth, int kHeight) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < totalPixels)) {
        int k = 0;
        int aboveBelow = (kHeight - 1) / 2;
        int sideToSide = (kWidth - 1) / 2;
        if (row < aboveBelow || row > (height - aboveBelow) || column < sideToSide || column > (width - sideToSide)) {
            output[row * width + column] = 0; // image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else if (image[tid] > 0) {
            int cond = true;
            for (int i = row - aboveBelow; i <= row + aboveBelow; i++) {
                for (int j = column - sideToSide; j <= column + sideToSide; j++) {
                    if (structuringElement[k] != 0) {
                        cond &= image[i * width + j] && structuringElement[k];
                    }
                    k++;
                }
            }
            if (cond)
                output[tid] = image[tid];
            else
                output[tid] = 0;
        } else {
            output[tid] = image[tid];
        }

    }
    return;
}

void imageErosion(Image *image, Image *output, int *structuringElement, int kWidth, int kHeight) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    float *d_structuringElement;
    CUDA_CHECK_RETURN(cudaMalloc(&d_structuringElement, sizeof(float) * kWidth * kHeight))
    CUDA_CHECK_RETURN(cudaMemcpy(d_structuringElement, structuringElement, sizeof(float) * kWidth * kHeight, cudaMemcpyHostToDevice))
    erodeImage<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_structuringElement, kWidth, kHeight);
    CUDA_CHECK_RETURN(cudaFree(d_structuringElement))

    return;
}

__global__ void averageFilter(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, float *kernel, int kWidth, int kHeight) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < totalPixels)) {
        int aboveBelow = (kHeight - 1) / 2;
        int sideToSide = (kWidth - 1) / 2;
        if (row < aboveBelow || row > (height - aboveBelow) || column < sideToSide || column > (width - sideToSide)) {
            output[row * width + column] = 0; // image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else {
            int sum = 0;
            int k = 0;
            for (int i = row - aboveBelow; i <= row + aboveBelow; i++) {
                for (int j = column - sideToSide; j <= column + sideToSide; j++) {
                    sum += image[i * width + j] * kernel[k];
                    k++;
                }
            }
            output[row * width + column] = (unsigned char) (sum / (kWidth * kHeight));
        }
    }
    return;
}

void averageFilter(Image *image, Image *output, float *kernel, int kWidth, int kHeight) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));
    float *d_kernel;
    CUDA_CHECK_RETURN(cudaMalloc(&d_kernel, sizeof(float) * kWidth * kHeight));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, kernel, sizeof(float) * kWidth * kHeight, cudaMemcpyHostToDevice));
    averageFilter<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_kernel, kWidth, kHeight);
    CUDA_CHECK_RETURN(cudaFree(d_kernel));
}

__global__ void linearFilter(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, float *kernel, int kWidth, int kHeight) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < totalPixels)) {
        int aboveBelow = (kHeight - 1) / 2;
        int sideToSide = (kWidth - 1) / 2;
        if (row < aboveBelow || row > (height - aboveBelow) || column < sideToSide || column > (width - sideToSide)) {
            output[row * width + column] = 0; // image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else {
            int sum = 0;
            int k = 0;
            for (int i = row - aboveBelow; i <= row + aboveBelow; i++) {
                for (int j = column - sideToSide; j <= column + sideToSide; j++) {
                    sum += image[i * width + j] * kernel[k];
                    k++;
                }
            }
            if (sum > 255)
                sum = 255;
            else if (sum < 0)
                sum = 0;
            output[row * width + column] = (unsigned char) sum;
        }
    }
    return;
}

void linearFilter(Image *image, Image *output, float *kernel, int kWidth, int kHeight) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));
    float *d_kernel;
    CUDA_CHECK_RETURN(cudaMalloc(&d_kernel, sizeof(float) * kWidth * kHeight));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, kernel, sizeof(float) * kWidth * kHeight, cudaMemcpyHostToDevice));
    linearFilter<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_kernel, kWidth, kHeight);
    CUDA_CHECK_RETURN(cudaFree(d_kernel));
}

__global__ void combineFilters(unsigned char *sobelX, unsigned char *sobelY, unsigned char *output, int totalPixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < totalPixels) {
        output[tid] = sqrtf((sobelX[tid] * sobelX[tid]) + (sobelY[tid] * sobelY[tid]));
    }
    return;
}

float *d_sobelX, *d_sobelY;

void setupEdgeDetection() {
    float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelX, sizeof(float) * 9))
    CUDA_CHECK_RETURN(cudaMemcpy(d_sobelX, sobelX, sizeof(float) * 9, cudaMemcpyHostToDevice))

    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelY, sizeof(float) * 9))
    CUDA_CHECK_RETURN(cudaMemcpy(d_sobelY, sobelY, sizeof(float) * 9, cudaMemcpyHostToDevice))
}

void cleanupEdgeDetection() {
    CUDA_CHECK_RETURN(cudaFree(d_sobelX))
    CUDA_CHECK_RETURN(cudaFree(d_sobelY))
}

void sobelFilter(Image *image, Image *output) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    unsigned char *d_sobelXImage, *d_sobelYImage;
    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelXImage, sizeof(unsigned char) * image->width * image->height))
    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelYImage, sizeof(unsigned char) * image->width * image->height))

    linearFilter<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, d_sobelXImage, image->width, image->height, totalPixels, d_sobelX, 3, 3);
    linearFilter<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, d_sobelYImage, image->width, image->height, totalPixels, d_sobelY, 3, 3);
    combineFilters<< < threadsPerBlock, blocksPerGrid, 0>> > (d_sobelXImage, d_sobelYImage, output->image, totalPixels);
    CUDA_CHECK_RETURN(cudaFree(d_sobelXImage))
    CUDA_CHECK_RETURN(cudaFree(d_sobelYImage))
}

__global__ void compassFilterKern(unsigned char *image, unsigned char *output, int width, int height, int totalPixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);

    if ((tid < totalPixels)) {

        float sobel0[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        float sobel1[9] = {-2, -1, 0, -1, 0, 1, 0, 1, 2};
        float sobel2[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        float sobel3[9] = {0, -1, -2, 1, 0, -1, 2, 1, 0};
        float *filters[4] = {sobel0, sobel1, sobel2, sobel3};
        if (row < 1 || row > (height - 1) || column < 1 || column > (width - 1)) {
            output[row * width + column] = 0; // image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else {
            int sum = 0;
            int maxSum = 0;
            int k = 0;
            for (int f = 0; f < 4; f++) {
                for (int i = row - 1; i <= row + 1; i++) {
                    for (int j = column - 1; j <= column + 1; j++) {
                        sum += image[i * width + j] * (filters[f][k]);
                        k++;
                    }
                }
                sum = max(sum, sum * -1);
                maxSum = max(sum, maxSum);
            }
//            if (maxSum > 255)
//                maxSum = 255;
//            else if (maxSum < 0)
//                maxSum = 0;
            output[row * width + column] = (unsigned char) maxSum;
        }
    }
    return;
}


void compassFilter(Image *image, Image *output) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height))
    compassFilterKern<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels);
}

__global__ void medianFilter(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, int *kernel, int kWidth, int kHeight, int *filteredVals, int kernSum) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    int kernLen = kWidth * kHeight;
    if ((tid < totalPixels)) {
        int aboveBelow = (kHeight - 1) / 2;
        int sideToSide = (kWidth - 1) / 2;
        if (row < aboveBelow || row > (height - aboveBelow) || column < sideToSide || column > (width - sideToSide)) {
            output[row * width + column] = 0;//image[row * width + column]; // handles when our filter would go outside the edge of the image
        } else {
            int k = 0;
            for (int i = row - aboveBelow; i <= row + aboveBelow; i++) {
                for (int j = column - sideToSide; j <= column + sideToSide; j++) {
                    // in theory this should handle weighted kernels... assuming I got all my indexing correct which I think I did...
                    for (int z = 0; z < kernel[k]; z++) {
                        filteredVals[(row * column * kernLen) + k + z] = image[i * width + j];
                    }
                    k++;
                }
            }

            // to find the median of the filteredValues I'm just going to sort it with an O(n^2) sort because at this level of parellelism O(n^2) on at max a few hundred items is the least of my worries.
            // could be sped up with a quicksort or something but thats a lot harder...
            int base = (row * column * kernSum);
            int j, key;
            for (int i = 0; i < kernSum; i++) {
                j = i - 1;
                key = filteredVals[base + i];
                while (j >= 0 && filteredVals[base + j] > key) {
                    filteredVals[base + j + 1] = filteredVals[base + j];
                    j--;
                }
                filteredVals[base + j + 1] = key;
            }
            output[row * width + column] = (unsigned char) filteredVals[base + kernSum / 2];
        }
    }
    return;
}

int arraySum(int *array, int arrayLen) {
    int sum = 0;
    for (int i = 0; i < arrayLen; i++) {
        sum += array[i];
    }
    return sum;
}

void medianFilter(Image *image, Image *output, int *kernel, int kWidth, int kHeight) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));
    int *d_kernel;
    CUDA_CHECK_RETURN(cudaMalloc(&d_kernel, sizeof(int) * kWidth * kHeight));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, kernel, sizeof(int) * kWidth * kHeight, cudaMemcpyHostToDevice));
    int *d_filteredVals;
    int kernSum = arraySum(kernel, kWidth * kHeight);
    CUDA_CHECK_RETURN(cudaMalloc(&d_filteredVals, sizeof(int) * kernSum * totalPixels));
    medianFilter<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_kernel, kWidth, kHeight, d_filteredVals, kernSum);
    CUDA_CHECK_RETURN(cudaFree(d_kernel));
    CUDA_CHECK_RETURN(cudaFree(d_filteredVals));
}


__global__ void setup_kernel(curandState *state) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, tid, 0, &state[tid]);
    return;
}

__global__ void generateSaltAndPepper(unsigned char *image, unsigned char *output, int width, int totalPixels, curandState *states, int level) {
    // VERY loosely based off https://www.projectrhea.org/rhea/index.php/How_to_Create_Salt_and_Pepper_Noise_in_an_Image
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if (tid < totalPixels) {
        curandState localState = states[tid];
        int randVal = curand_uniform(&localState) * (level - 0 + .999999);
        if (randVal == level) {
            float sop = curand_uniform(&localState);
            if (sop > .5) {
                output[row * width + column] = 255;
            } else {
                output[row * width + column] = 0;
            }
        } else {
            output[row * width + column] = image[row * width + column];
        }
    }
    return;
}


void setupRandomness(Image *image) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&d_states, sizeof(curandState) * totalPixels); // need a random state for each thread
    setup_kernel<< < threadsPerBlock, blocksPerGrid>> > (d_states);
}

void cleanupRandomness() {
    CUDA_CHECK_RETURN(cudaFree(d_states));

}

void saltAndPepperNoise(Image *image, Image *output, int level) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));

    generateSaltAndPepper<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, totalPixels, d_states, level);
}

__global__ void imageQuantizationKernel(unsigned char *image, unsigned char *output, int totalPixels, int *levels, int numLevels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < totalPixels) {
        for (int i = 0; i < numLevels; i++) {
            if (image[tid] >= levels[i * 3] && image[tid] < levels[i * 3 + 1]) {
//                printf("value %i is between %i and %i and was set to %i\n", image[tid], levels[i * 3], levels[i * 3 + 1], levels[i * 3 + 2]);
                output[tid] = (unsigned char) levels[i * 3 + 2];
                break;
            } else {
                output[tid] = image[tid];
            }
        }
    }
    return;
}

__global__ void arrayDifference(unsigned char *image, unsigned char *output, unsigned char *difference, int *histogram, int totalPixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < totalPixels) {
        difference[tid] = (unsigned char) (round((image[tid] - output[tid]) * (image[tid] - output[tid]) * (histogram[image[tid]] / 255.0)));
    }
    return;
}

int calcMSQE(Image *image, Image *output) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    unsigned char *d_difference;
    CUDA_CHECK_RETURN(cudaMalloc(&d_difference, sizeof(unsigned char) * totalPixels));
    int *h_histogram = (int *) malloc(sizeof(int) * 255);
    int *d_histogram;
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram, sizeof(int) * 255));
    calculateHistogram(image, h_histogram, d_histogram);

    arrayDifference<< < threadsPerBlock, blocksPerGrid>> > (image->image, output->image, d_difference, d_histogram, totalPixels);
    thrust::device_ptr<unsigned char> dp = thrust::device_pointer_cast(d_difference);
    thrust::device_vector<unsigned char> thrust_diff(dp, dp + totalPixels);
    int sum = thrust::reduce(
            thrust_diff.cbegin(),
            thrust_diff.cend(),
            0.0f,
            thrust::plus<unsigned char>());
    CUDA_CHECK_RETURN(cudaFree(d_difference));
    CUDA_CHECK_RETURN(cudaFree(d_histogram));
    free(h_histogram);

    return sum;
}

void imageQuantization(Image *image, Image *output, int *levels, int numLevels) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));
    int *d_levels;
    CUDA_CHECK_RETURN(cudaMalloc(&d_levels, sizeof(int) * numLevels * 3));
    CUDA_CHECK_RETURN(cudaMemcpy(d_levels, levels, sizeof(int) * 3 * numLevels, cudaMemcpyHostToDevice))

    // kernel call here
    imageQuantizationKernel<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, totalPixels, d_levels, numLevels);
    CUDA_CHECK_RETURN(cudaFree(d_levels))
}


__global__ void addGaussianNoiseToImage(unsigned char *image, unsigned char *output, int width, int numPixels, float mean, float variance, curandState *states) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if (tid < numPixels) {
        curandState localState = states[tid];
        int randVal = curand_uniform(&localState) * (255 - 0 + .999999);
//        float noise = round((1 / (sqrt(2 * CUDART_PI_F * variance))) * exp(-1.0f * (((randVal - mean) * (randVal - mean)) / (2 * variance))));
// based off the formula found here for pythons random normal
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
        float noise = round(255 * exp(-1.0f * (((randVal - mean) * (randVal - mean)) / (2 * variance))));

//        printf("noise %i %f %f %f %f\n", randVal, coef, exponent, blah, noise);
        output[row * width + column] = (unsigned char) (image[row * width + column] + noise);
    }
    return;
}

void addGaussianNoise(Image *image, Image *output, float meanPar, float stdDevPar) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));
    float mean = meanPar;
    float stdDev = stdDevPar;
    if (mean == -1 || stdDev == -1) {
        calculateMeanAndStdDev(image, &mean, &stdDev);
    }
    addGaussianNoiseToImage<< < threadsPerBlock, blocksPerGrid>> > (image->image, output->image, image->width, totalPixels, mean, stdDev * stdDev, d_states);

}


void copyHostImageToDevice(Image *host, Image *device) {
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMalloc((void **) &(device->image), (int) sizeof(unsigned char) * host->width * host->height));
    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, (int) sizeof(unsigned char) * host->width * host->height, cudaMemcpyHostToDevice));
    // copy height and width to device
    device->height = host->height;
    device->width = host->width;
}

void copyDeviceImageToHost(Image *device, Image *host) {
    // copy height and width back to host
    host->height = device->height;
    host->width = device->width;
    host->image = (unsigned char *) malloc(sizeof(unsigned char) * host->height * host->width);
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(host->image, device->image, sizeof(unsigned char) * device->width * device->height, cudaMemcpyDeviceToHost));
}

void copyDeviceRGBImageToHost(RGBImage *device, RGBImage *host) {
    // copy height and width back to host
    host->height = device->height;
    host->width = device->width;
    host->channels = 1;
    host->image = (unsigned char *) malloc(sizeof(unsigned char) * host->height * host->width);
    // copy actual image data back to host from device
    CUDA_CHECK_RETURN(cudaMemcpy(host->image, device->image, sizeof(unsigned char) * device->width * device->height, cudaMemcpyDeviceToHost));
//    for (int i = 0; i < (host->height * host->width); i++) {
//        printf("%i\n", host->image[i]);
//    }

}

void copyHostRGBImageToDevice(RGBImage *host, RGBImage *device) {
    // copy actual image data to device from host
//    unsigned char*
    CUDA_CHECK_RETURN(cudaMalloc((void **) &(device->image), sizeof(unsigned char) * host->width * host->height * host->channels));
    CUDA_CHECK_RETURN(cudaMemcpy(device->image, host->image, sizeof(unsigned char) * host->width * host->height * host->channels, cudaMemcpyHostToDevice));
    // copy height and width to device
    device->height = host->height;
    device->width = host->width;
    device->channels = host->channels;
}

