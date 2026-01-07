
file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Derivatives/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSystems/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Tensor/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Grid/*.cpp
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
# ─────────────────────────────── ASM / LLVM dump options ───────────────────────────────
option(DUMP_ASM "Dump generated assembly (.s files) for inspection" OFF)
option(DUMP_LLVM "Dump LLVM IR (.ll files) for inspection" OFF)

if(DUMP_ASM)
  message(STATUS "Assembly dump enabled (-S -masm=intel -fverbose-asm)")
  add_compile_options(-S -masm=intel -fverbose-asm)
endif()

if(DUMP_LLVM)
  message(STATUS "LLVM IR dump enabled (-emit-llvm -S)")
  add_compile_options(-emit-llvm -S)
endif()

find_library(OPENBLAS_LIB
    NAMES openblasp openblas
    PATHS /opt/OpenBLAS/lib /usr/lib64 /usr/lib /usr/local/lib
)

if(OPENBLAS_LIB)
  message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
  target_link_libraries(TensoriumTests PRIVATE ${OPENBLAS_LIB})

  if(EXISTS "/opt/OpenBLAS/include/cblas.h")
    target_include_directories(TensoriumTests PRIVATE /opt/OpenBLAS/include)
  elseif(EXISTS "/usr/include/openblas/cblas.h")
    target_include_directories(TensoriumTests PRIVATE /usr/include/openblas)
  elseif(EXISTS "/usr/include/cblas.h")
    target_include_directories(TensoriumTests PRIVATE /usr/include)
  endif()

else()
  find_package(BLAS)
  if(BLAS_FOUND)
    message(STATUS "CMake BLAS found: ${BLAS_LIBRARIES}")
    target_link_libraries(TensoriumTests PRIVATE ${BLAS_LIBRARIES})
  else()
    message(WARNING " No BLAS/OpenBLAS library found — BLAS tests will be skipped.")
  endif()
endif()



# set(DISPATCH_PLUGIN_PATH "${CMAKE_BINARY_DIR}/Plugins/libTensoriumDispatchPlugin.so")
#
# add_dependencies(TensoriumTests TensoriumDispatchPlugin)
#
# # inject the clang plugin (frontend)
# set_property(TARGET TensoriumTests APPEND_STRING PROPERTY COMPILE_FLAGS
#     " -Xclang -load -Xclang ${DISPATCH_PLUGIN_PATH} -Xclang -add-plugin -Xclang tensorium-dispatch"
# )
