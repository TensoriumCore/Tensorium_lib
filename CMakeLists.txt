cmake_minimum_required(VERSION 3.12)
project(morpheus_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes)

pybind11_add_module(morpheus ./bindings.cpp)

