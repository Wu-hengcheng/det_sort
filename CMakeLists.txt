cmake_minimum_required(VERSION 2.8)
project(perception)
set(CMAKE_CXX_STANDARD 14)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Threads)
link_libraries(pthread protobuf)

aux_source_directory(.  DIR_SRCS)
aux_source_directory(deepsort/src  SORT_SRCS)
#aux_source_directory(./object_proto DIR_OBJ_SRCS)
#aux_source_directory(./object_proto/map DIR_MAP_SRCS)
#add_subdirectory(object_proto)


find_package(CUDA REQUIRED)

#find_package(PCL 1.3 REQUIRED COMPONENTS common io) 
#include_directories(${PCL_INCLUDE_DIRS})

include_directories(/usr/local/cuda/include yaml-cpp
                                         deepsort/include
                    /usr/include/eigen3)
#include_directories(. ./object_proto ./object_proto/map)

link_directories(/usr/local/cuda/lib64)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

include_directories(/home/midas/TensorRT-8.0.1.6/include/)
link_directories(/home/midas/TensorRT-8.0.1.6/lib/)

file(GLOB catkin_LIBRARIES "/opt/ros/melodic/lib/lib*.so")
include_directories(/opt/ros/melodic/include;/opt/ros/melodic/xmlrpcpp)

#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} ${DIR_SRCS} ${SORT_SRCS})
#target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${LIBNVONNXPARSER})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${catkin_LIBRARIES} yaml-cpp)
