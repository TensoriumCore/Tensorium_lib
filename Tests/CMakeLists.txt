enable_testing()

enable_testing()

set(TEST_FRAMEWORK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/TestRegistry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/TestRunner.cpp)

set(TEST_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/MatrixTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/VectorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/TensorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/LinearSystemsTests.cpp)

set(TEST_GRID_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/grid/DerivativeTests.cpp)

set(TEST_BSSN_INITIAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/initial_data/InitialDataTests.cpp)
set(TEST_BSSN_CONSTRAINT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/constraints/ConstraintTests.cpp)
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
    list(APPEND TEST_BSSN_CONSTRAINT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/bssn/constraints/TildeGammaSymbolsValidationTests.cpp)
    list(APPEND TEST_BSSN_CONSTRAINT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/bssn/constraints/GammaSourceValidationTests.cpp)
endif()
set(TEST_BSSN_RICCI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/ricci/RicciTests.cpp)
set(TEST_BSSN_SOLVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/solvers/SolverTests.cpp)
set(TEST_BSSN_EVOLUTION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/ATildeEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/KEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/GammaEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/GaugeEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/CCZ4SchwarzschildStabilityTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/evolution/RHSHaloSentinelTests.cpp)

set(TEST_BSSN_STABILITY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/minkowski_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/schwarzschild_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/kerr_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/bowen_york_boost_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/ccz4_prime_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/brown_spherical_analytic_constraints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/brown_hyperbolicity_speeds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bssn/boundary_conditions.cpp)

set(BENCHMARK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/MatrixBenchmark.cpp)

function(tensorium_add_object_library target)
    add_library(${target} OBJECT ${ARGN})
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/includes
        ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

tensorium_add_object_library(test_core_objs ${TEST_CORE_SOURCES})
tensorium_add_object_library(test_grid_objs ${TEST_GRID_SOURCES})
tensorium_add_object_library(test_bssn_initial_objs ${TEST_BSSN_INITIAL_SOURCES})
tensorium_add_object_library(test_bssn_constraints_objs ${TEST_BSSN_CONSTRAINT_SOURCES})
tensorium_add_object_library(test_bssn_ricci_objs ${TEST_BSSN_RICCI_SOURCES})
tensorium_add_object_library(test_bssn_solvers_objs ${TEST_BSSN_SOLVER_SOURCES})
tensorium_add_object_library(test_bssn_evolution_objs ${TEST_BSSN_EVOLUTION_SOURCES})
tensorium_add_object_library(test_bssn_stability_objs ${TEST_BSSN_STABILITY_SOURCES})

set(BSSN_OBJECTS
    $<TARGET_OBJECTS:test_bssn_initial_objs>
    $<TARGET_OBJECTS:test_bssn_constraints_objs>
    $<TARGET_OBJECTS:test_bssn_ricci_objs>
    $<TARGET_OBJECTS:test_bssn_solvers_objs>
    $<TARGET_OBJECTS:test_bssn_evolution_objs>
    $<TARGET_OBJECTS:test_bssn_stability_objs>)

add_executable(TensoriumChiEvolutionDemo
    bssn/binaries/BSSNChiEvolutionDemo.cpp)
target_include_directories(TensoriumChiEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumGammaEvolutionDemo
    bssn/binaries/BSSNGammaEvolutionDemo.cpp)
target_include_directories(TensoriumGammaEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumGaugeEvolutionDemo
    bssn/binaries/BSSNGaugeEvolutionDemo.cpp)
target_include_directories(TensoriumGaugeEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumATildeEvolutionDemo
    bssn/binaries/BSSNATildeEvolutionDemo.cpp)
target_include_directories(TensoriumATildeEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumKEvolutionDemo
    bssn/binaries/BSSNKEvolutionDemo.cpp)
target_include_directories(TensoriumKEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumTests
    main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    $<TARGET_OBJECTS:test_core_objs>
    $<TARGET_OBJECTS:test_grid_objs>
    ${BSSN_OBJECTS})

target_include_directories(TensoriumTests PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumTestCore
    core_main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    $<TARGET_OBJECTS:test_core_objs>)

target_include_directories(TensoriumTestCore PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumTestGrid
    grid_main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    $<TARGET_OBJECTS:test_grid_objs>)

target_include_directories(TensoriumTestGrid PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumTestBSSN
    bssn_main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    ${BSSN_OBJECTS})

target_include_directories(TensoriumTestBSSN PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumBenchmarks ${BENCHMARK_SOURCES})
target_include_directories(TensoriumBenchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_EXECUTABLES TensoriumTests TensoriumTestCore TensoriumTestGrid TensoriumTestBSSN)
set(TENSORIUM_ALL_EXECUTABLES ${TEST_EXECUTABLES} TensoriumBenchmarks
    TensoriumChiEvolutionDemo TensoriumGammaEvolutionDemo TensoriumATildeEvolutionDemo
    TensoriumKEvolutionDemo TensoriumGaugeEvolutionDemo)

add_test(NAME tensorium.tests.all COMMAND TensoriumTests)
add_test(NAME tensorium.tests.core COMMAND TensoriumTestCore)
add_test(NAME tensorium.tests.grid COMMAND TensoriumTestGrid)
add_test(NAME tensorium.tests.bssn COMMAND TensoriumTestBSSN)
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
  add_test(NAME gamma_source_validation
           COMMAND TensoriumTests --test bssn.constraints.gamma_source_validation)
endif()
add_test(NAME ccz4_schwarzschild_stability
         COMMAND TensoriumTests --test bssn.evolution.ccz4_schwarzschild_stability)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  include_directories(${OPENMP_INCLUDE_PATH})
  link_directories(${OPENMP_LIB_PATH})
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_compile_options(${target} PRIVATE -Xpreprocessor -fopenmp)
    target_link_libraries(${target} PRIVATE ${OPENMP_LIB})
  endforeach()
elseif(DEFINED OPENMP_LIB)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE ${OPENMP_LIB})
  endforeach()
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${target} PRIVATE MORPHEUS_USE_MPI)
  endforeach()
endif()

if(USE_KNL)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_compile_options(${target} PRIVATE -mtune=knl -mfma -mavx512f -mavx512cd)
    target_compile_definitions(${target} PRIVATE USE_KNL)
    target_link_libraries(${target} PRIVATE memkind)
  endforeach()
endif()

find_library(OPENBLAS_LIB
    NAMES openblasp openblas
    PATHS /opt/OpenBLAS/lib /usr/lib64 /usr/lib /usr/local/lib)

if(OPENBLAS_LIB)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE ${OPENBLAS_LIB})
  endforeach()

  if(EXISTS "/opt/OpenBLAS/include/cblas.h")
    include_directories(/opt/OpenBLAS/include)
  elseif(EXISTS "/usr/include/openblas/cblas.h")
    include_directories(/usr/include/openblas)
  elseif(EXISTS "/usr/include/cblas.h")
    include_directories(/usr/include)
  endif()
else()
  find_package(BLAS)
  if(BLAS_FOUND)
    foreach(target ${TENSORIUM_ALL_EXECUTABLES})
      target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
    endforeach()
  else()
    message(WARNING " No BLAS/OpenBLAS library found — BLAS tests will be skipped.")
  endif()
endif()

add_test(NAME core COMMAND TensoriumTestCore)
add_test(NAME grid COMMAND TensoriumTestGrid)
add_test(NAME bssn COMMAND TensoriumTestBSSN)
add_test(NAME bssn.initial_data COMMAND TensoriumTests --test bssn.initial_data)
add_test(NAME bssn.constraints COMMAND TensoriumTests --test bssn.constraints)
add_test(NAME bssn.ricci COMMAND TensoriumTests --test bssn.ricci)
add_test(NAME bssn.solvers COMMAND TensoriumTests --test bssn.solvers)
