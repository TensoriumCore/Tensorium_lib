cmake_minimum_required(VERSION 3.22)
project(Tensorium LANGUAGES C CXX)
include(CheckIncludeFileCXX)

option(USE_MPI "Enable MPI support" OFF)
option(USE_KNL "Tune for Intel KNL" OFF)
option(AVX2 "Enable AVX2+FMA codegen" OFF)
option(AVX512 "Enable AVX-512" OFF)
option(VERBOSE "Verbose runtime logs" OFF)
option(DEBUG "Build with debug symbols" OFF)
option(BUILD_PLUGINS "Build Clang/LLVM plugins" OFF)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_PYBIND "Build Python bindings" OFF)
option(USE_CUDA "Enable CUDA kernels (requires nvcc)" OFF)
option(USE_CBLAS "Enable BLAS" ON)
option(USE_HDF5 "Enable HDF5 export support when HDF5 is available" ON)
option(TENSORIUM_BSSN_PROFILE_KERNELS "Enable BSSN RHS timing instrumentation" OFF)
option(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS "Validate on-the-fly Γ̃ symbols" OFF)
option(TENSORIUM_ENABLE_COMPILER_OPT_REPORTS
       "Emit compiler optimization diagnostics for successful and missed optimizations" OFF)
option(TENSORIUM_SAVE_OPTIMIZATION_RECORD
       "Save compiler optimization records to files when the compiler supports it" OFF)
set(TENSORIUM_OPT_REMARK_FILTER ".*" CACHE STRING
    "Regex filter for Clang/IntelLLVM optimization remarks when TENSORIUM_ENABLE_COMPILER_OPT_REPORTS=ON")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        link_libraries(${ACCELERATE_LIB})
        add_compile_definitions(TENSORIUM_USE_CBLAS)
    endif()

    # Ensure the generated binaries run on the host macOS by default.
    # Avoid CMake falling back to the newest SDK version (e.g. 15.x) when
    # the machine is still on 14.x, which would make the executables
    # unlaunchable on this system.
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _macos_ver "${CMAKE_SYSTEM_VERSION}")
        if(NOT _macos_ver)
            execute_process(COMMAND sw_vers -productVersion
                            OUTPUT_VARIABLE _sw_vers
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" _macos_ver "${_sw_vers}")
        endif()
        if(NOT _macos_ver)
            set(_macos_ver "14.0")
        endif()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "${_macos_ver}"
            CACHE STRING "Minimum macOS version to target" FORCE)
        message(STATUS "Setting CMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_definitions(TENSORIUM_X86)
    if(NOT USE_KNL)
      add_compile_options(-mavx)
    endif()
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

set(TENSORIUM_HBW_LIB "")
if(USE_KNL)
  check_include_file_cxx(hbwmalloc.h TENSORIUM_HAVE_HBWMALLOC_H)
  find_library(TENSORIUM_HBW_LIB NAMES hbw)
  add_compile_definitions(TENSORIUM_ARCH_KNL=1)
  if(TENSORIUM_HAVE_HBWMALLOC_H AND TENSORIUM_HBW_LIB)
    add_compile_definitions(USE_KNL)
    message(STATUS "KNL mode: hbwmalloc enabled (${TENSORIUM_HBW_LIB})")
  else()
    message(WARNING "USE_KNL=ON but hbwmalloc not found; falling back to aligned/NUMA allocator.")
  endif()
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

  if (CUDAToolkit_NVCC_EXECUTABLE AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()

  enable_language(CUDA)

  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 90)
  string(JOIN "_" TENSORIUM_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
  add_compile_definitions(TENSORIUM_CUDA_ARCH_LIST="${TENSORIUM_CUDA_ARCH_LIST}")

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-fPIC --use_fast_math -lineinfo -Wno-deprecated-gpu-targets")

  include_directories(${CUDAToolkit_INCLUDE_DIRS})
else()
  add_compile_definitions(TENSORIUM_NO_CUDA)
endif()

include_directories(${CMAKE_SOURCE_DIR}/includes)

set(TENSORIUM_HDF5_FOUND FALSE)
set(TENSORIUM_HDF5_LIBRARIES "")
if(USE_HDF5)
  find_package(HDF5 QUIET COMPONENTS C)
  if(HDF5_FOUND)
    set(TENSORIUM_HDF5_FOUND TRUE)
    set(TENSORIUM_HDF5_LIBRARIES ${HDF5_LIBRARIES})
    include_directories(${HDF5_INCLUDE_DIRS})
    add_compile_definitions(TENSORIUM_USE_HDF5)
    message(STATUS "HDF5 export support enabled")
  else()
    message(STATUS "HDF5 not found; export support disabled")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
  if(USE_KNL)
    set(BASE_FLAGS "-O3 -march=knl -mtune=knl -Wno-ignored-attributes")
    set(KNL_FLAGS "-mfma -mavx512f -mavx512cd")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      add_compile_options(-fopenmp-simd)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${KNL_FLAGS}")
  else()
    set(BASE_FLAGS "-O3 -march=native -mtune=native -Wno-ignored-attributes")
    set(AVX_FLAGS "-mavx")
    set(AVX2_FLAGS "-mfma -mavx2")
    set(AVX512_FLAGS "-mfma -mavx512f -mavx512cd")
    if(AVX512)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${AVX512_FLAGS}")
    elseif(AVX2)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${AVX2_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS} ${AVX_FLAGS}")
    endif()
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=native -Wno-ignored-attributes")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
endif()

set(TENSORIUM_COMPILER_OPT_REPORT_FLAGS "")
if(TENSORIUM_ENABLE_COMPILER_OPT_REPORTS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|IntelLLVM")
    list(APPEND TENSORIUM_COMPILER_OPT_REPORT_FLAGS
         "-Rpass=${TENSORIUM_OPT_REMARK_FILTER}"
         "-Rpass-missed=${TENSORIUM_OPT_REMARK_FILTER}"
         "-Rpass-analysis=${TENSORIUM_OPT_REMARK_FILTER}")
    if(TENSORIUM_SAVE_OPTIMIZATION_RECORD)
      list(APPEND TENSORIUM_COMPILER_OPT_REPORT_FLAGS -fsave-optimization-record)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND TENSORIUM_COMPILER_OPT_REPORT_FLAGS
         -fopt-info-optimized
         -fopt-info-missed)
    if(TENSORIUM_SAVE_OPTIMIZATION_RECORD)
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        list(APPEND TENSORIUM_COMPILER_OPT_REPORT_FLAGS -fsave-optimization-record)
      else()
        message(WARNING
                "TENSORIUM_SAVE_OPTIMIZATION_RECORD=ON requested, but GCC ${CMAKE_CXX_COMPILER_VERSION} "
                "does not reliably support -fsave-optimization-record; enabling stderr reports only.")
      endif()
    endif()
  else()
    message(WARNING
            "TENSORIUM_ENABLE_COMPILER_OPT_REPORTS=ON but compiler ${CMAKE_CXX_COMPILER_ID} "
            "has no configured optimization diagnostics flags.")
  endif()

  if(TENSORIUM_COMPILER_OPT_REPORT_FLAGS)
    add_compile_options(${TENSORIUM_COMPILER_OPT_REPORT_FLAGS})
    string(JOIN " " _tensorium_opt_flags_msg ${TENSORIUM_COMPILER_OPT_REPORT_FLAGS})
    message(STATUS "Compiler optimization diagnostics enabled: ${_tensorium_opt_flags_msg}")
  endif()
endif()

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()
if(TENSORIUM_BSSN_PROFILE_KERNELS)
  add_compile_definitions(TENSORIUM_BSSN_PROFILE_KERNELS)
endif()
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
  add_compile_definitions(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
endif()

set(OPENMP_LIB "")
set(OPENMP_COMPILE_FLAGS "")
set(OPENMP_LIB_DIR "")
if(APPLE)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(OPENMP_LIB "OpenMP::OpenMP_CXX")
        if(TARGET OpenMP::OpenMP_CXX)
            # Discover the actual libomp location so we can add an rpath.
            get_target_property(_omp_loc OpenMP::OpenMP_CXX IMPORTED_LOCATION_RELEASE)
            if(NOT _omp_loc)
                get_target_property(_omp_loc OpenMP::OpenMP_CXX IMPORTED_LOCATION)
            endif()
            if(_omp_loc)
                get_filename_component(OPENMP_LIB_DIR "${_omp_loc}" DIRECTORY)
            endif()
        endif()
    else()
        set(OPENMP_SEARCH_HINTS "")
        if(DEFINED ENV{OpenMP_ROOT} AND NOT "$ENV{OpenMP_ROOT}" STREQUAL "")
            list(APPEND OPENMP_SEARCH_HINTS "$ENV{OpenMP_ROOT}" "$ENV{OpenMP_ROOT}/lib")
        endif()
    if(DEFINED ENV{LIBOMP_ROOT} AND NOT "$ENV{LIBOMP_ROOT}" STREQUAL "")
      list(APPEND OPENMP_SEARCH_HINTS "$ENV{LIBOMP_ROOT}" "$ENV{LIBOMP_ROOT}/lib")
    endif()

    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
      execute_process(
        COMMAND ${BREW_EXECUTABLE} --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(HOMEBREW_LIBOMP_PREFIX)
        list(APPEND OPENMP_SEARCH_HINTS "${HOMEBREW_LIBOMP_PREFIX}" "${HOMEBREW_LIBOMP_PREFIX}/lib")
      endif()
    endif()

    find_library(OPENMP_LIBOMP_LIBRARY
      NAMES omp libomp
      HINTS ${OPENMP_SEARCH_HINTS})
    if(OPENMP_LIBOMP_LIBRARY)
        set(OPENMP_LIB "${OPENMP_LIBOMP_LIBRARY}")
        get_filename_component(OPENMP_LIB_DIR "${OPENMP_LIBOMP_LIBRARY}" DIRECTORY)
        set(OPENMP_COMPILE_FLAGS -Xpreprocessor -fopenmp)
    endif()
  endif()
  if(OPENMP_LIB AND NOT OPENMP_LIB_DIR)
    # Fallback to common Homebrew locations so runtime can locate libomp.
    foreach(_omp_prefix "/usr/local/opt/libomp/lib" "/opt/homebrew/opt/libomp/lib" "/usr/local/lib")
      if(EXISTS "${_omp_prefix}/libomp.dylib")
        set(OPENMP_LIB_DIR "${_omp_prefix}")
        break()
      endif()
    endforeach()
  endif()
else()
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB "OpenMP::OpenMP_CXX")
  endif()
endif()

if (HAVE_NUMA)
    link_libraries(${NUMA_LIB})
endif()
if(USE_KNL AND TENSORIUM_HAVE_HBWMALLOC_H AND TENSORIUM_HBW_LIB)
  link_libraries(${TENSORIUM_HBW_LIB})
endif()

if(BUILD_PLUGINS)
  add_subdirectory(Plugins)
endif()

if(BUILD_PYBIND)
  add_subdirectory(Pybind)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif()
