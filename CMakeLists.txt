cmake_minimum_required(VERSION 3.17)
project(OPENCL_CNN_CONV)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)
link_directories(CL)
add_executable(OPENCL_CNN_CONV main.cpp cnn.h func.h)
# add_executable(OPENCL_CNN_CONV main.cpp )
target_link_libraries(OPENCL_CNN_CONV OpenCL.lib FreeImage.lib)
