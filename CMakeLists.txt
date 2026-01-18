cmake_minimum_required(VERSION 3.22)
project(Tensorium LANGUAGES C CXX)

option(USE_MPI "Enable MPI support" OFF)
option(USE_KNL "Tune for Intel KNL" OFF)
option(AVX512 "Enable AVX-512" OFF)
option(VERBOSE "Verbose runtime logs" OFF)
option(DEBUG "Build with debug symbols" OFF)
option(BUILD_PLUGINS "Build Clang/LLVM plugins" ON)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_PYBIND "Build Python bindings" OFF)
option(USE_CUDA "Enable CUDA kernels (requires nvcc)" OFF)
option(USE_CBLAS "Enable BLAS" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/local" "/opt/local/libexec/llvm-20")
    include_directories("/opt/local/include")
    link_directories("/opt/local/lib")
    link_directories("/opt/local/libexec/llvm-20/lib")
    
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        link_libraries(${ACCELERATE_LIB})
        add_compile_definitions(TENSORIUM_USE_CBLAS)
    endif()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_definitions(TENSORIUM_X86)
    add_compile_options(-mavx2 -mfma)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_compile_definitions(TENSORIUM_ARM)
else()
    add_compile_definitions(TENSORIUM_FALLBACK)
endif()

find_library(NUMA_LIB numa)
if (NUMA_LIB)
    add_compile_definitions(USE_NUMA)
    set(HAVE_NUMA TRUE)
else()
    set(HAVE_NUMA FALSE)
endif()

find_package(CUDAToolkit QUIET)

execute_process(
  COMMAND bash -c "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo none"
  OUTPUT_VARIABLE GPU_NAME OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT GPU_NAME STREQUAL "none")
  add_compile_definitions(TENSORIUM_GPU_PRESENT)
endif()

if (USE_CUDA)
  if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "USE_CUDA=ON but CUDA toolkit not found")
  endif()

  add_compile_definitions(TENSORIUM_CUDA)

  enable_language(CUDA)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")

  set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 90)
  add_compile_definitions(TENSORIUM_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-fPIC --use_fast_math -lineinfo -Wno-deprecated-gpu-targets")

  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR})
else()
  add_compile_definitions(TENSORIUM_NO_CUDA)
endif()

include_directories(${CMAKE_SOURCE_DIR}/Includes)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
  set(BASE_FLAGS "-O3 -mtune=native -Wno-ignored-attributes -Rpass-analysis=tensorium-align")
  set(AVX2_FLAGS "-mfma -mavx2")
  set(AVX512_FLAGS "-mfma -mavx512f -mavx512cd")
  if(AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${AVX512_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${AVX2_FLAGS}")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=native -Wno-ignored-attributes")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
endif()

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()

if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENMP_INCLUDE_PATH "/opt/local/include/libomp")
    set(OPENMP_LIB_PATH "/opt/local/lib/libomp")
    set(OPENMP_LIB "omp")
    
    include_directories(${OPENMP_INCLUDE_PATH})
    link_directories(${OPENMP_LIB_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
      set(OPENMP_LIB "OpenMP::OpenMP_CXX")
    endif()
  endif()
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB "OpenMP::OpenMP_CXX")
  endif()
endif()

if (HAVE_NUMA)
    link_libraries(${NUMA_LIB})
endif()

if(BUILD_PLUGINS)
  add_subdirectory(Plugins)
endif()

if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()


