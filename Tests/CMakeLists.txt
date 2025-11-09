file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Derivatives/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSystems/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Tensor/*.cpp
)

add_executable(TensoriumTests ${TEST_SOURCES})

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  include_directories(${OPENMP_INCLUDE_PATH})
  link_directories(${OPENMP_LIB_PATH})
  target_compile_options(TensoriumTests PRIVATE -Xpreprocessor -fopenmp)
  target_link_libraries(TensoriumTests PRIVATE ${OPENMP_LIB})
elseif(DEFINED OPENMP_LIB)
  target_link_libraries(TensoriumTests PRIVATE ${OPENMP_LIB})
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(TensoriumTests PRIVATE MPI::MPI_CXX)
  target_compile_definitions(TensoriumTests PRIVATE MORPHEUS_USE_MPI)
endif()

if(USE_KNL)
  target_compile_options(TensoriumTests PRIVATE -mtune=knl -mfma -mavx512f -mavx512cd)
  target_compile_definitions(TensoriumTests PRIVATE USE_KNL)
  target_link_libraries(TensoriumTests PRIVATE memkind)
endif()

option(DUMP_ASM "Dump generated assembly (.s files) for inspection" OFF)
option(DUMP_LLVM "Dump LLVM IR (.ll files) for inspection" OFF)

if(DUMP_ASM)
  add_compile_options(-S -masm=intel -fverbose-asm)
endif()

if(DUMP_LLVM)
  add_compile_options(-emit-llvm -S)
endif()

find_library(OPENBLAS_LIB
    NAMES openblasp openblas
    PATHS /opt/OpenBLAS/lib /usr/lib64 /usr/lib /usr/local/lib
)

if(OPENBLAS_LIB)
  target_link_libraries(TensoriumTests PRIVATE ${OPENBLAS_LIB})
  if(EXISTS "/opt/OpenBLAS/include/cblas.h")
    set(BLAS_INCLUDE_DIR "/opt/OpenBLAS/include")
  elseif(EXISTS "/usr/include/openblas/cblas.h")
    set(BLAS_INCLUDE_DIR "/usr/include/openblas")
  elseif(EXISTS "/usr/include/cblas.h")
    set(BLAS_INCLUDE_DIR "/usr/include")
  endif()
  if(DEFINED BLAS_INCLUDE_DIR)
    target_include_directories(TensoriumTests PRIVATE ${BLAS_INCLUDE_DIR})
  endif()
else()
  find_package(BLAS)
  if(BLAS_FOUND)
    target_link_libraries(TensoriumTests PRIVATE ${BLAS_LIBRARIES})
  endif()
endif()
option(GENERATE_BC "Emit LLVM bitcode (.bc) for TensoriumTests" OFF)

if(GENERATE_BC)
  set(BC_OUTPUT_DIR "${CMAKE_BINARY_DIR}/out/bitcode")
  file(MAKE_DIRECTORY ${BC_OUTPUT_DIR})

  if(TARGET TensoriumTests)
    get_target_property(target_sources TensoriumTests SOURCES)
    get_target_property(target_includes TensoriumTests INCLUDE_DIRECTORIES)
    separate_arguments(CLANG_FLAGS UNIX_COMMAND "${CMAKE_CXX_FLAGS}")

    set(INCLUDE_ARGS "")
    if(DEFINED target_includes)
      foreach(inc ${target_includes})
        list(APPEND INCLUDE_ARGS "-I${inc}")
      endforeach()
    endif()
    if(DEFINED BLAS_INCLUDE_DIR)
      list(APPEND INCLUDE_ARGS "-I${BLAS_INCLUDE_DIR}")
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/Includes")
      list(APPEND INCLUDE_ARGS "-I${CMAKE_SOURCE_DIR}/Includes")
    endif()

    set(BC_FILES "")
    foreach(src ${target_sources})
      get_filename_component(src_name ${src} NAME_WE)
      set(bc_file "${BC_OUTPUT_DIR}/${src_name}.bc")
      add_custom_command(
        OUTPUT ${bc_file}
        COMMAND ${CMAKE_CXX_COMPILER}
                -emit-llvm -c ${src}
                -std=c++17
                ${CLANG_FLAGS}
                ${INCLUDE_ARGS}
                -o ${bc_file}
        DEPENDS ${src}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        VERBATIM
      )
      list(APPEND BC_FILES ${bc_file})
    endforeach()

    add_custom_target(tensorium_bc ALL
      COMMAND llvm-link ${BC_FILES} -o ${BC_OUTPUT_DIR}/Tensorium_full.bc
      DEPENDS ${BC_FILES}
      VERBATIM
    )
  endif()
endif()

