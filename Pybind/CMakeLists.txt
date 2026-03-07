
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
pybind11_add_module(tensorium MODULE bindings.cpp)
target_include_directories(tensorium PRIVATE ${CMAKE_SOURCE_DIR}/includes)
