cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(guess_aruco_type)

find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(cli11_proj)

# Executables
add_executable (guess_aruco_type guess_aruco_type.cpp)
target_link_libraries (guess_aruco_type ${OpenCV_LIBS} CLI11::CLI11)

add_executable (automatically_detecting_aruco_video automatically_detecting_aruco_video.cpp)
target_link_libraries (automatically_detecting_aruco_video ${OpenCV_LIBS} CLI11::CLI11)