
cmake_minimum_required(VERSION 3.12)
project(morpheus_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimizations and SIMD
add_compile_options(-O3 -march=native -mavx2 -mfma)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/includes)

# Create pybind11 module
pybind11_add_module(morpheus ./bindings.cpp)

# Link OpenMP
target_link_libraries(morpheus PRIVATE OpenMP::OpenMP_CXX)

