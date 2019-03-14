cmake_minimum_required(VERSION 3.13)
project(imageAnalysis CXX CUDA)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    message("Found OpenMP!")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message("Missed OpenMP!")
endif ()


set(CMAKE_C_STANDARD 11)
include_directories("${PROJECT_SOURCE_DIR}/library")

add_subdirectory(library)
add_subdirectory(test)

target_link_libraries(imageAnalysis_test PRIVATE imageAnalysis ${OpenCV_LIBS})

# I don't take you to be an apple guy but just in case...
if (APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET imageAnalysis_test
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif ()