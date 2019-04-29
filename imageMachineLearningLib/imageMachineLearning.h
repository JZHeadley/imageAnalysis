#ifndef IMAGE_MACHINE_LEARNING_LIBRARY_H
#define IMAGE_MACHINE_LEARNING_LIBRARY_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
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

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value);

#endif