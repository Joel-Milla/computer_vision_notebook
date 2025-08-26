cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(simple_thresholding)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (simple_thresholding simple_thresholding.cpp)
target_link_libraries (simple_thresholding ${OpenCV_LIBRARIES})

add_executable (otsu_thresholding otsu_thresholding.cpp)
target_link_libraries (otsu_thresholding ${OpenCV_LIBRARIES})


