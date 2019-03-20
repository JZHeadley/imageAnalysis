#include "imageAnalysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <omp.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value);

curandState *d_states;

__global__ void convertRGBToGrayscaleLuminance(unsigned char *image, int width, int height, int numPixels, int channels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((.21 * image[row * width + column]) + (.72 * image[row * width + column + numPixels]) + (.07 * image[row * width + column + (2 * numPixels)]));
//        printf("%i %i\n", row, column);
    }
    return;
}

__global__ void convertRGBToGrayscaleAverage(unsigned char *image, int width, int height, int numPixels, int channels, unsigned char *output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if ((tid < numPixels)) {
        output[row * width + column] = (unsigned char) ((image[row * width + column] + image[row * width + column + numPixels] + image[row * width + column + (2 * numPixels)]) / 3);
//        printf("%i %i\n", row, column);
    }
    return;
}


void convertRGBToGrayscale(RGBImage *d_rgb, Image *d_gray, int method) {
    /*
    don't think you mentioned a grayscale conversion method so I looked it up and used this page as my guide
    https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    */
    int totalPixels = d_rgb->width * d_rgb->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    switch (method) {
        case 0:
            // luminance method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(d_gray->image), (int) sizeof(unsigned char) * d_rgb->width * d_rgb->height));
            convertRGBToGrayscaleLuminance<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, d_rgb->height, totalPixels, d_rgb->channels, d_gray->image);
            d_gray->width = d_rgb->width;
            d_gray->height = d_rgb->height;
            break;
        case 1:
            // average method
            CUDA_CHECK_RETURN(cudaMalloc((void **) &(d_gray->image), (int) sizeof(unsigned char) * d_rgb->width * d_rgb->height));

            convertRGBToGrayscaleAverage<< < threadsPerBlock, blocksPerGrid>> > (d_rgb->image, d_rgb->width, d_rgb->height, totalPixels, d_rgb->channels, d_gray->image);
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
//    int operationsPerThread = 10;
//    int numOperations = totalPixels / operationsPerThread;
//    int blocksPerGrid = (numOperations + threadsPerBlock - 1) / threadsPerBlock;
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

//TODO: convert this to not an average filter and do normalization on the result of this instead of averaging.
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
            output[row * width + column] = (unsigned char) (sum);
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
                    filteredVals[(row * column * kernLen) + k] = image[i * width + j] * kernel[k];
                    k++;
                }
            }
            // to find the median of the filteredValues I'm just going to sort it with an O(n^2) sort because at this level of parellelism O(n^2) on at max a few hundred items is the least of my worries.
            // could be sped up with a quicksort or something but thats a lot harder...
            int base = (row * column * kernLen);
            int j, key;
            for (int i = 0; i < kernLen; i++) {
                j = i - 1;
                key = filteredVals[base + i];
                while (j >= 0 && filteredVals[base + j] > key) {
                    filteredVals[base + j + 1] = filteredVals[base + j];
                    j--;
                }
                filteredVals[base + j + 1] = key;
            }
            output[row * width + column] = (unsigned char) filteredVals[base + kernLen / 2];
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
//    int id = threadIdx.x + blockIdx.x * 64;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, tid, 0, &state[tid]);
    return;
}

__global__ void generateSaltAndPepper(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, curandState *states, int level) {
    // VERY loosely based off https://www.projectrhea.org/rhea/index.php/How_to_Create_Salt_and_Pepper_Noise_in_an_Image
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    if (tid < totalPixels) {
        curandState localState = states[tid];
        int randVal = curand_uniform(&localState) * (level - 0 + .999999);
        if (randVal == level) {
            randVal = curand_uniform(&localState);
            if (randVal > .5) {
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

void saltAndPepperNoise(Image *image, Image *output, int level) {
    int totalPixels = image->width * image->height;
    int threadsPerBlock = 512;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    output->width = image->width;
    output->height = image->height;
    CUDA_CHECK_RETURN(cudaMalloc(&(output->image), sizeof(unsigned char) * image->width * image->height));

    generateSaltAndPepper<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, image->width, image->height, totalPixels, d_states, level);
}

__global__ void imageQuantizationKernel(unsigned char *image, unsigned char *output, int width, int height, int totalPixels, int *levels, int numLevels) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / width;
    int column = tid - ((tid / width) * width);
    bool leveled = false;
    if (tid < totalPixels) {
        for (int i = 0; i < numLevels; i++) {
            if (image[row * width + column] <= levels[i * 3] && image[row * width + column] < levels[i * 3 + 1]) {
                output[row * width + column] = (unsigned char) levels[i * 3 + 2];
                leveled = true;
            }
        }
        if (!leveled) {
            output[row * width + column] = image[row * width + column];
        }
    }
    return;
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
    CUDA_CHECK_RETURN(cudaMemcpy(d_levels, levels, sizeof(int) * 3 * numLevels, cudaMemcpyHostToDevice));

    // kernel call here
    imageQuantizationKernel<< < threadsPerBlock, blocksPerGrid, 0>> > (image->image, output->image, output->width, output->height, totalPixels, d_levels, numLevels);
//    CUDA_CHECK_RETURN(cudaFree(d_levels));
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
//    for (int i = 0; i < (host->height * host->width); i++) {
//        printf("%i\n", host->image[i]);
//    }

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

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr<<statement<<" returned "<<cudaGetErrorString(err)<<"("<<err<<") at "<<file<<":"<<line<<std::endl;
    exit(1);
}