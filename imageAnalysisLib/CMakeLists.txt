
add_library(imageAnalysis
        imageAnalysis.cu
        imageAnalysis.h
        )

add_subdirectory(imageMachineLearningLib)
include_directories("imageMachineLearningLib")
target_link_libraries(imageAnalysis PRIVATE imageMachineLearning)

target_compile_features(imageAnalysis PUBLIC cxx_std_11)
set_target_properties(imageAnalysis PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
