cmake_minimum_required(VERSION 3.22)
project(Tensorium LANGUAGES C CXX)

# ─────────────────────────────── Options ───────────────────────────────
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

# ─────────────────────────────── Includes ──────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/Includes)

# ─────────────────────────────── CPU architecture flags ───────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
  message(STATUS "Configuring for x86_64: enabling AVX2/FMA or AVX512")
  set(BASE_FLAGS "-O3 -mtune=native -Wno-ignored-attributes -Rpass-analysis=tensorium-align")
  set(AVX2_FLAGS "-mfma -mavx2")
  set(AVX512_FLAGS "-mfma -mavx512f -mavx512cd")
  if(AVX512)
    set(CMAKE_CXX_FLAGS "${BASE_FLAGS} ${AVX512_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${BASE_FLAGS} ${AVX2_FLAGS}")
  endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Configuring for Apple Silicon ARM64: disabling AVX flags")
  # Apple Clang n’a pas de support AVX/FMA → on reste sur les optimisations génériques ARM
  set(CMAKE_CXX_FLAGS "-O3 -mcpu=apple-m1 -Wno-ignored-attributes")
else()
  message(WARNING "Unknown architecture (${CMAKE_SYSTEM_PROCESSOR}); using generic optimization flags.")
  set(CMAKE_CXX_FLAGS "-O3 -mtune=native")
endif()

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()



# ─────────────────────────────── Architecture detection ───────────────────────────────
message(STATUS "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# ─────────────────────────────── OpenMP handling ───────────────────────────────
if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Configuring OpenMP manually for macOS ARM64 (Apple Clang)")
    set(OPENMP_INCLUDE_PATH "/opt/homebrew/opt/libomp/include")
    set(OPENMP_LIB_PATH "/opt/homebrew/opt/libomp/lib")
    set(OPENMP_LIB "omp")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Using OpenMP for macOS x86_64 (Intel)")
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

# ─────────────────────────────── Subdirectories ───────────────────────────────
if(BUILD_PLUGINS)
  add_subdirectory(Plugins)
endif()

if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()

if(BUILD_PYBIND)
  add_subdirectory(Pybind)
endif()
