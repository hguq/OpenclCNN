cmake_minimum_required(VERSION 3.17)
project(OPENCL_CNN_INTEGER)

set(CMAKE_CXX_STANDARD 11)

add_executable(OPENCL_CNN_INTEGER main.cpp)
target_link_libraries(OPENCL_CNN_INTEGER OpenCL.lib FreeImage.lib)
