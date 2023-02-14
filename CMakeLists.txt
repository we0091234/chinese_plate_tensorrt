cmake_minimum_required(VERSION 3.10)
project(plate_detect_rec)
set(OpenCV_DIR /mnt/Gu/softWare/opencv/opencv455/install/lib/cmake/opencv4)
add_definitions(-std=c++11)
add_definitions(-w)
find_package(CUDA REQUIRED)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

#cuda 
include_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/include)
link_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/lib)

#tensorrt 
include_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/include/)
link_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/lib/)

#opencv 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

#onnx2trt
add_subdirectory(${PROJECT_SOURCE_DIR}/onnx2trt)

# cuda_add_executable(plate_detect_rec detect_rec_plate_cuda.cpp utils.cpp postprocess.cu preprocess.cu)
cuda_add_executable(plate_detect_rec  utils.cpp plate_detect.cpp plate_rec.cpp main.cpp postprocess.cu preprocess.cu)

target_link_libraries(plate_detect_rec nvinfer)
target_link_libraries(plate_detect_rec cudart)
target_link_libraries(plate_detect_rec nvonnxparser)
target_link_libraries(plate_detect_rec ${OpenCV_LIBS})
add_definitions(-O0)