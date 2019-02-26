cmake_minimum_required(VERSION 3.13)
project(imageAnalysis C CUDA)

set(CMAKE_C_STANDARD 11)
include_directories("${PROJECT_SOURCE_DIR}/library")

add_subdirectory(library)
add_subdirectory(test)

target_link_libraries(imageAnalysis_test PRIVATE imageAnalysis)

# I don't take you to be an apple guy but just in case...
if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET imageAnalysis_test
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()