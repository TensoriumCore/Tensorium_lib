
file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Derivatives/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearSystems/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Tensor/*.cpp
)

add_executable(TensoriumTests ${TEST_SOURCES})

# --- OpenMP flags selon architecture ---
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  include_directories(${OPENMP_INCLUDE_PATH})
  link_directories(${OPENMP_LIB_PATH})
  target_compile_options(TensoriumTests PRIVATE -Xpreprocessor -fopenmp)
  target_link_libraries(TensoriumTests PRIVATE ${OPENMP_LIB})
elseif(DEFINED OPENMP_LIB)
  target_link_libraries(TensoriumTests PRIVATE ${OPENMP_LIB})
endif()

# --- MPI / KNL ---
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

