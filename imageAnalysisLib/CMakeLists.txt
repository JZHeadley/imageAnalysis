
add_library(imageAnalysis
        imageAnalysis.cu
        imageAnalysis.h
        )

target_compile_features(imageAnalysis PUBLIC cxx_std_11)
set_target_properties(imageAnalysis PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
