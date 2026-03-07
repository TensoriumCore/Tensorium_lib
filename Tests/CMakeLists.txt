enable_testing()

enable_testing()

set(TEST_FRAMEWORK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/TestRegistry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/TestRunner.cpp)

set(TEST_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/MatrixTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/VectorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/TensorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/LinearSystemsTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/DerivateTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/SpectralTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/AllocatorTests.cpp)

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

set(TWOPUNCTURES_C_TARGET "")
set(TWOPUNCTURES_C_ROOT "${PROJECT_SOURCE_DIR}/../TwoPuncturesC")
if(EXISTS "${TWOPUNCTURES_C_ROOT}/include/TwoPunctures.h")
    find_program(GSL_CONFIG_EXECUTABLE gsl-config)
    if(GSL_CONFIG_EXECUTABLE)
        execute_process(COMMAND ${GSL_CONFIG_EXECUTABLE} --cflags
                        OUTPUT_VARIABLE TP_GSL_CFLAGS
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${GSL_CONFIG_EXECUTABLE} --libs
                        OUTPUT_VARIABLE TP_GSL_LIBS
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        separate_arguments(TP_GSL_CFLAGS_LIST UNIX_COMMAND "${TP_GSL_CFLAGS}")
        separate_arguments(TP_GSL_LIBS_LIST UNIX_COMMAND "${TP_GSL_LIBS}")

        # Extract include directories from gsl-config output so every target that
        # sees TwoPunctures headers also sees the GSL headers they rely on.
        set(TP_GSL_INCLUDE_DIRS "")
        foreach(flag ${TP_GSL_CFLAGS_LIST})
            if(flag MATCHES "^-I(.+)")
                list(APPEND TP_GSL_INCLUDE_DIRS "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        set(TWOPUNCTURES_C_SOURCES
            ${TWOPUNCTURES_C_ROOT}/src/TwoPunctures.c
            ${TWOPUNCTURES_C_ROOT}/src/TP_CoordTransf.c
            ${TWOPUNCTURES_C_ROOT}/src/TP_Equations.c
            ${TWOPUNCTURES_C_ROOT}/src/TP_FuncAndJacobian.c
            ${TWOPUNCTURES_C_ROOT}/src/TP_Newton.c
            ${TWOPUNCTURES_C_ROOT}/src/TP_Utilities.c)
        add_library(tensorium_twopunctures_c STATIC ${TWOPUNCTURES_C_SOURCES})
        target_compile_definitions(tensorium_twopunctures_c PRIVATE TP_OMP)
        add_compile_definitions(TENSORIUM_TWOPUNCTURES_OMP)
        target_include_directories(tensorium_twopunctures_c PUBLIC
            ${TWOPUNCTURES_C_ROOT}/include
            ${TP_GSL_INCLUDE_DIRS})
        if(TARGET OpenMP::OpenMP_C)
            target_link_libraries(tensorium_twopunctures_c PUBLIC OpenMP::OpenMP_C)
        elseif(TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(tensorium_twopunctures_c PUBLIC OpenMP::OpenMP_CXX)
        elseif(OPENMP_LIB)
            target_link_libraries(tensorium_twopunctures_c PUBLIC ${OPENMP_LIB})
        endif()
        if(OPENMP_COMPILE_FLAGS)
            target_compile_options(tensorium_twopunctures_c PRIVATE ${OPENMP_COMPILE_FLAGS})
        endif()
        if(TP_GSL_CFLAGS_LIST)
            target_compile_options(tensorium_twopunctures_c PRIVATE ${TP_GSL_CFLAGS_LIST})
        endif()
        if(TP_GSL_LIBS_LIST)
            target_link_libraries(tensorium_twopunctures_c PUBLIC ${TP_GSL_LIBS_LIST})
        endif()
        add_compile_definitions(TENSORIUM_HAS_TWOPUNCTURES_C)
        if(TP_GSL_INCLUDE_DIRS)
            include_directories(${TP_GSL_INCLUDE_DIRS})
        endif()
        include_directories(${TWOPUNCTURES_C_ROOT}/include)
        set(TWOPUNCTURES_C_TARGET tensorium_twopunctures_c)
        message(STATUS "TwoPuncturesC backend enabled")
    else()
        message(STATUS "TwoPuncturesC found but gsl-config missing; backend disabled")
    endif()
else()
    message(STATUS "TwoPuncturesC source tree not found; backend disabled")
endif()

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
set(TEST_OBJECT_TARGETS
    test_core_objs
    test_grid_objs
    test_bssn_initial_objs
    test_bssn_constraints_objs
    test_bssn_ricci_objs
    test_bssn_solvers_objs
    test_bssn_evolution_objs
    test_bssn_stability_objs)

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

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFTW3 QUIET fftw3)
endif()

if(NOT FFTW3_FOUND)
  find_path(FFTW3_INCLUDE_DIR fftw3.h
            PATHS /opt/homebrew/include /usr/local/include /opt/local/include)
  find_library(FFTW3_LIBRARY NAMES fftw3
               PATHS /opt/homebrew/lib /usr/local/lib /opt/local/lib)
  if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    set(FFTW3_FOUND TRUE)
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
  endif()
endif()

if(FFTW3_FOUND)
  target_include_directories(TensoriumBenchmarks PRIVATE ${FFTW3_INCLUDE_DIRS})
  target_link_libraries(TensoriumBenchmarks PRIVATE ${FFTW3_LIBRARIES})
  target_compile_definitions(TensoriumBenchmarks PRIVATE TENSORIUM_USE_FFTW)
endif()

set(TEST_EXECUTABLES TensoriumTests TensoriumTestCore TensoriumTestGrid TensoriumTestBSSN)
set(TENSORIUM_ALL_EXECUTABLES ${TEST_EXECUTABLES} TensoriumBenchmarks
    TensoriumChiEvolutionDemo TensoriumGammaEvolutionDemo TensoriumATildeEvolutionDemo
    TensoriumKEvolutionDemo TensoriumGaugeEvolutionDemo)
set(TENSORIUM_OPENMP_TARGETS ${TEST_OBJECT_TARGETS} ${TENSORIUM_ALL_EXECUTABLES})

if(TWOPUNCTURES_C_TARGET)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE ${TWOPUNCTURES_C_TARGET})
  endforeach()
endif()

add_test(NAME tensorium.tests.all COMMAND TensoriumTests)
add_test(NAME tensorium.tests.core COMMAND TensoriumTestCore)
add_test(NAME tensorium.tests.grid COMMAND TensoriumTestGrid)
add_test(NAME tensorium.tests.bssn COMMAND TensoriumTestBSSN)
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
  add_test(NAME gamma_source_validation
           COMMAND TensoriumTests --test bssn.constraints.gamma_source_validation)
  add_test(NAME tilde_gamma_symbols
           COMMAND TensoriumTests --test bssn.constraints.tilde_gamma_symbols_validation)
endif()
add_test(NAME ccz4_schwarzschild_stability
         COMMAND TensoriumTests --test bssn.evolution.ccz4_schwarzschild_stability)

if(OPENMP_LIB)
  if(APPLE AND IS_ABSOLUTE "${OPENMP_LIB}")
    get_filename_component(OPENMP_LIB_DIR "${OPENMP_LIB}" DIRECTORY)
  endif()
  foreach(target ${TENSORIUM_OPENMP_TARGETS})
    if(OPENMP_COMPILE_FLAGS)
      target_compile_options(${target} PRIVATE ${OPENMP_COMPILE_FLAGS})
    endif()
    target_link_libraries(${target} PRIVATE ${OPENMP_LIB})
    if(APPLE AND OPENMP_LIB_DIR)
      target_link_options(${target} PRIVATE "-Wl,-rpath,${OPENMP_LIB_DIR}")
    endif()
  endforeach()
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${target} PRIVATE MORPHEUS_USE_MPI TENSORIUM_USE_MPI)
  endforeach()
endif()

find_library(TENSORIUM_MEMKIND_LIB NAMES memkind)
if(USE_KNL)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_compile_options(${target} PRIVATE -mtune=knl -mfma -mavx512f -mavx512cd)
    target_compile_definitions(${target} PRIVATE USE_KNL)
    if(TENSORIUM_MEMKIND_LIB)
      target_link_libraries(${target} PRIVATE ${TENSORIUM_MEMKIND_LIB})
    endif()
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
