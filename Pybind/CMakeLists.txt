
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
pybind11_add_module(tensorium MODULE bindings.cpp)
target_include_directories(tensorium PRIVATE ${CMAKE_SOURCE_DIR}/includes)

set(TWOPUNCTURES_C_ROOT "${PROJECT_SOURCE_DIR}/../TwoPuncturesC")
set(TWOPUNCTURES_C_PYBIND_TARGET "")
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

    add_library(tensorium_twopunctures_c_pybind STATIC ${TWOPUNCTURES_C_SOURCES})
    set(TWOPUNCTURES_C_PYBIND_TARGET tensorium_twopunctures_c_pybind)
    target_include_directories(tensorium_twopunctures_c_pybind PUBLIC
      ${TWOPUNCTURES_C_ROOT}/include
      ${TP_GSL_INCLUDE_DIRS})
    target_compile_definitions(tensorium_twopunctures_c_pybind PRIVATE TP_OMP)

    if(TARGET OpenMP::OpenMP_C)
      target_link_libraries(tensorium_twopunctures_c_pybind PUBLIC OpenMP::OpenMP_C)
    elseif(TARGET OpenMP::OpenMP_CXX)
      target_link_libraries(tensorium_twopunctures_c_pybind PUBLIC OpenMP::OpenMP_CXX)
    elseif(OPENMP_LIB)
      target_link_libraries(tensorium_twopunctures_c_pybind PUBLIC ${OPENMP_LIB})
    endif()
    if(OPENMP_COMPILE_FLAGS)
      target_compile_options(tensorium_twopunctures_c_pybind PRIVATE ${OPENMP_COMPILE_FLAGS})
    endif()

    if(TP_GSL_CFLAGS_LIST)
      target_compile_options(tensorium_twopunctures_c_pybind PRIVATE ${TP_GSL_CFLAGS_LIST})
    endif()
    if(TP_GSL_LIBS_LIST)
      target_link_libraries(tensorium_twopunctures_c_pybind PUBLIC ${TP_GSL_LIBS_LIST})
    endif()

    target_include_directories(tensorium PRIVATE
      ${TWOPUNCTURES_C_ROOT}/include
      ${TP_GSL_INCLUDE_DIRS})
    target_compile_definitions(tensorium PRIVATE
      TENSORIUM_HAS_TWOPUNCTURES_C
      TENSORIUM_TWOPUNCTURES_OMP)
    target_link_libraries(tensorium PRIVATE ${TWOPUNCTURES_C_PYBIND_TARGET})

    message(STATUS "TwoPuncturesC backend enabled for Python bindings")
  else()
    message(STATUS "TwoPuncturesC found but gsl-config missing; Python backend disabled")
  endif()
else()
  message(STATUS "TwoPuncturesC source tree not found for Python bindings")
endif()
