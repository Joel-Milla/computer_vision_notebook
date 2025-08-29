cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(image_gradients)

find_package(OpenCV REQUIRED)


# Executable for sobel_scharr
add_executable (sobel_scharr sobel_scharr.cpp)
target_link_libraries (sobel_scharr ${OpenCV_LIBS})