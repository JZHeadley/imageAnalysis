
add_library(imageMachineLearning
        imageMachineLearning.cu
        imageMachineLearning.h
        )

target_compile_features(imageMachineLearning PUBLIC cxx_std_11)
set_target_properties(imageMachineLearning PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
