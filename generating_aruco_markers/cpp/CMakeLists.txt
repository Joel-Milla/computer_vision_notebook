cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(generating_aruco_markers)

find_package(OpenCV REQUIRED)

# Use FetchContent as recommended in CLI11 docs
# First time doing CMake, it pulls this repository, and the version. Then, it compiles it. This process takes some time the first time. 
# However, after doing make again and again, will use this compiled files and make the linking. Which is faster than having the 10,000+ lines of code and compiling/linking it every time. CMake handles download and compile step the first time. Instead of doing it every time you do 'make'
include(FetchContent)
FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)

FetchContent_MakeAvailable(cli11_proj)

add_executable(generating_aruco generating_aruco.cpp)
target_link_libraries(generating_aruco ${OpenCV_LIBS} CLI11::CLI11)