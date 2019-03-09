add_executable(imageAnalysis_test test.cu)
set_property(TARGET imageAnalysis_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

