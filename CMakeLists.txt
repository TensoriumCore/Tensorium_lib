
cmake_minimum_required(VERSION 3.12)
project(morpheus_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-O3 -march=native -mavx2 -mfma)

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/includes)

pybind11_add_module(morpheus ./pybind/bindings.cpp)

target_link_libraries(morpheus PRIVATE OpenMP::OpenMP_CXX)

