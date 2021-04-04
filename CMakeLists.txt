cmake_minimum_required(VERSION 3.17)
project(vector_add)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)
link_directories(CL)
add_executable(vector_add main.cpp)
target_link_libraries(vector_add OpenCL.lib)
