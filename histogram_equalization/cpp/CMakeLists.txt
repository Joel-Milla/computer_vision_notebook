cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(histograms)

find_package(OpenCV REQUIRED)


# Executables
add_executable (histogram_equalization histogram_equalization.cpp)
target_link_libraries (histogram_equalization ${OpenCV_LIBS})

add_executable (histogram_adaptive_eq histogram_adaptive_eq.cpp)
target_link_libraries (histogram_adaptive_eq ${OpenCV_LIBS})