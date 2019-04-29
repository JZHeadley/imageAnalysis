cmake_minimum_required(VERSION 3.13)
include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.131.tar.gz"
        SHA1 "0512ea6392065c8ece808296fddd7d8664d30920"
)
project(imageAnalysis CXX CUDA)
hunter_add_package(jsoncpp)
#hunter_add_package(OpenCV)

find_package(OpenCV REQUIRED core imgproc highgui)
find_package(OpenMP)
find_package(jsoncpp CONFIG REQUIRED)

if (OPENMP_FOUND)
    message("Found OpenMP!")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message("Missed OpenMP!")
endif ()


set(CMAKE_C_STANDARD 11)
include_directories("${PROJECT_SOURCE_DIR}/imageAnalysisLib")
include_directories("${PROJECT_SOURCE_DIR}/imageMachineLearningLib")

add_subdirectory(imageAnalysisLib)
add_subdirectory(imageMachineLearningLib)
add_subdirectory(test)

target_link_libraries(imageAnalysis_test PRIVATE imageMachineLearning imageAnalysis ${OpenCV_LIBS} jsoncpp_lib_static)

# I don't take you to be an apple guy but just in case...
if (APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET imageAnalysis_test
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif ()