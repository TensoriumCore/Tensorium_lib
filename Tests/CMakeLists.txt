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

set(TEST_Z4C_INITIAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/initial_data/InitialDataTests.cpp)
set(TEST_Z4C_CONSTRAINT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/constraints/ConstraintTests.cpp)
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
    list(APPEND TEST_Z4C_CONSTRAINT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/z4c/constraints/TildeGammaSymbolsValidationTests.cpp)
    list(APPEND TEST_Z4C_CONSTRAINT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/z4c/constraints/GammaSourceValidationTests.cpp)
endif()
set(TEST_Z4C_RICCI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/ricci/RicciTests.cpp)
set(TEST_Z4C_SOLVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/solvers/SolverTests.cpp)
set(TEST_Z4C_EVOLUTION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/ATildeEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/KEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/GammaEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/GaugeEvolutionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/FMRHierarchyTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/MovingPunctureEccentricityControlTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/CCZ4SchwarzschildStabilityTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/evolution/RHSHaloSentinelTests.cpp)

set(TEST_Z4C_STABILITY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/minkowski_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/schwarzschild_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/kerr_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/bowen_york_boost_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/ccz4_prime_stability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/brown_spherical_analytic_constraints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/brown_hyperbolicity_speeds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/z4c/boundary_conditions.cpp)

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
tensorium_add_object_library(test_z4c_initial_objs ${TEST_Z4C_INITIAL_SOURCES})
tensorium_add_object_library(test_z4c_constraints_objs ${TEST_Z4C_CONSTRAINT_SOURCES})
tensorium_add_object_library(test_z4c_ricci_objs ${TEST_Z4C_RICCI_SOURCES})
tensorium_add_object_library(test_z4c_solvers_objs ${TEST_Z4C_SOLVER_SOURCES})
tensorium_add_object_library(test_z4c_evolution_objs ${TEST_Z4C_EVOLUTION_SOURCES})
tensorium_add_object_library(test_z4c_stability_objs ${TEST_Z4C_STABILITY_SOURCES})
set(TEST_OBJECT_TARGETS
    test_core_objs
    test_grid_objs
    test_z4c_initial_objs
    test_z4c_constraints_objs
    test_z4c_ricci_objs
    test_z4c_solvers_objs
    test_z4c_evolution_objs
    test_z4c_stability_objs)

set(Z4C_OBJECTS
    $<TARGET_OBJECTS:test_z4c_initial_objs>
    $<TARGET_OBJECTS:test_z4c_constraints_objs>
    $<TARGET_OBJECTS:test_z4c_ricci_objs>
    $<TARGET_OBJECTS:test_z4c_solvers_objs>
    $<TARGET_OBJECTS:test_z4c_evolution_objs>
    $<TARGET_OBJECTS:test_z4c_stability_objs>)

add_executable(TensoriumChiEvolutionDemo
    z4c/binaries/Z4cChiEvolutionDemo.cpp)
target_include_directories(TensoriumChiEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumGammaEvolutionDemo
    z4c/binaries/Z4cGammaEvolutionDemo.cpp)
target_include_directories(TensoriumGammaEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumGaugeEvolutionDemo
    z4c/binaries/Z4cGaugeEvolutionDemo.cpp)
target_include_directories(TensoriumGaugeEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumATildeEvolutionDemo
    z4c/binaries/Z4cATildeEvolutionDemo.cpp)
target_include_directories(TensoriumATildeEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumKEvolutionDemo
    z4c/binaries/Z4cKEvolutionDemo.cpp)
target_include_directories(TensoriumKEvolutionDemo PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(TensoriumMovingPuncture
    z4c/binaries/MovingPunctureRunner.cpp)
target_include_directories(TensoriumMovingPuncture PRIVATE
    ${PROJECT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR})
if(TENSORIUM_HDF5_FOUND)
  target_link_libraries(TensoriumMovingPuncture PRIVATE ${TENSORIUM_HDF5_LIBRARIES})
endif()

add_executable(TensoriumTests
    main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    $<TARGET_OBJECTS:test_core_objs>
    $<TARGET_OBJECTS:test_grid_objs>
    ${Z4C_OBJECTS})

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

add_executable(TensoriumTestZ4c
    z4c_main.cpp
    ${TEST_FRAMEWORK_SOURCES}
    ${Z4C_OBJECTS})

target_include_directories(TensoriumTestZ4c PRIVATE
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

set(TEST_EXECUTABLES TensoriumTests TensoriumTestCore TensoriumTestGrid TensoriumTestZ4c)
set(TENSORIUM_ALL_EXECUTABLES ${TEST_EXECUTABLES} TensoriumBenchmarks
    TensoriumChiEvolutionDemo TensoriumGammaEvolutionDemo TensoriumATildeEvolutionDemo
    TensoriumKEvolutionDemo TensoriumGaugeEvolutionDemo TensoriumMovingPuncture)
set(TENSORIUM_OPENMP_TARGETS ${TEST_OBJECT_TARGETS} ${TENSORIUM_ALL_EXECUTABLES})

if(TWOPUNCTURES_C_TARGET)
  foreach(target ${TENSORIUM_ALL_EXECUTABLES})
    target_link_libraries(${target} PRIVATE ${TWOPUNCTURES_C_TARGET})
  endforeach()
endif()

add_test(NAME tensorium.tests.all COMMAND TensoriumTests)
add_test(NAME tensorium.tests.core COMMAND TensoriumTestCore)
add_test(NAME tensorium.tests.grid COMMAND TensoriumTestGrid)
add_test(NAME tensorium.tests.z4c COMMAND TensoriumTestZ4c)
if(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
  add_test(NAME gamma_source_validation
           COMMAND TensoriumTests --test z4c.constraints.gamma_source_validation)
  add_test(NAME tilde_gamma_symbols
           COMMAND TensoriumTests --test z4c.constraints.tilde_gamma_symbols_validation)
endif()
add_test(NAME ccz4_schwarzschild_stability
         COMMAND TensoriumTests --test z4c.evolution.ccz4_schwarzschild_stability)

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

  # MPI-specific executables
  add_executable(TensoriumMPIMovingPuncture
      z4c/mpi/mpi_moving_puncture.cpp)
  target_include_directories(TensoriumMPIMovingPuncture PRIVATE
      ${PROJECT_SOURCE_DIR}/includes
      ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(TensoriumMPIMovingPuncture PRIVATE MPI::MPI_CXX)
  if(TENSORIUM_HDF5_FOUND)
    target_link_libraries(TensoriumMPIMovingPuncture PRIVATE ${TENSORIUM_HDF5_LIBRARIES})
  endif()
  target_compile_definitions(TensoriumMPIMovingPuncture PRIVATE
      TENSORIUM_ENABLE_MPI TENSORIUM_USE_MPI)

  if(OPENMP_LIB)
    if(OPENMP_COMPILE_FLAGS)
      target_compile_options(TensoriumMPIMovingPuncture PRIVATE ${OPENMP_COMPILE_FLAGS})
    endif()
    target_link_libraries(TensoriumMPIMovingPuncture PRIVATE ${OPENMP_LIB})
    if(APPLE AND OPENMP_LIB_DIR)
      target_link_options(TensoriumMPIMovingPuncture PRIVATE "-Wl,-rpath,${OPENMP_LIB_DIR}")
    endif()
  endif()

  if(TWOPUNCTURES_C_TARGET)
    target_link_libraries(TensoriumMPIMovingPuncture PRIVATE ${TWOPUNCTURES_C_TARGET})
  endif()

  if(OPENBLAS_LIB)
    target_link_libraries(TensoriumMPIMovingPuncture PRIVATE ${OPENBLAS_LIB})
  elseif(BLAS_FOUND)
    target_link_libraries(TensoriumMPIMovingPuncture PRIVATE ${BLAS_LIBRARIES})
  endif()

  # MPI test
  add_test(NAME mpi.moving_puncture
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   $<TARGET_FILE:TensoriumMPIMovingPuncture> --nx 64 --steps 10)
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
add_test(NAME z4c COMMAND TensoriumTestZ4c)
add_test(NAME z4c.initial_data COMMAND TensoriumTests --test z4c.initial_data)
add_test(NAME z4c.constraints COMMAND TensoriumTests --test z4c.constraints)
add_test(NAME z4c.ricci COMMAND TensoriumTests --test z4c.ricci)
add_test(NAME z4c.solvers COMMAND TensoriumTests --test z4c.solvers)
