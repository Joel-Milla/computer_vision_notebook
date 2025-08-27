cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(kernel)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Executable for kernels
add_executable (kernel kernel.cpp)
target_link_libraries (kernel ${OpenCV_LIBS} Eigen3::Eigen)