cmake_minimum_required(VERSION 3.22)
project(Tensorium LANGUAGES CXX)

option(USE_MPI "Enable MPI support" OFF)
option(USE_KNL "Tune for Intel KNL" OFF)
option(AVX512 "Enable AVX-512" OFF)
option(VERBOSE "Verbose runtime logs" OFF)
option(DEBUG "Build with debug symbols" OFF)
option(BUILD_PLUGINS "Build Clang/LLVM plugins" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_PYBIND "Build Python bindings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only include path
include_directories(${CMAKE_SOURCE_DIR}/Includes)

# Global compiler flags
set(BASE_FLAGS "-O3 -fopenmp -mtune=native -Wno-ignored-attributes -Rpass-analysis=tensorium-align")
set(AVX2_FLAGS "-mfma -mavx2")
set(AVX512_FLAGS "-mfma -mavx512f -mavx512cd")

if(AVX512)
  set(CMAKE_CXX_FLAGS "${BASE_FLAGS} ${AVX512_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "${BASE_FLAGS} ${AVX2_FLAGS}")
endif()

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()

# Subdirectories
if(BUILD_PLUGINS)
  add_subdirectory(Plugins)
endif()

if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()

if(BUILD_PYBIND)
  add_subdirectory(Pybind)
endif()
