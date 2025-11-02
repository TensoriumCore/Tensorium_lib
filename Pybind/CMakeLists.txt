
find_package(pybind11 REQUIRED)
pybind11_add_module(tensorium_bindings bindings.cpp)
target_include_directories(tensorium_bindings PRIVATE ${CMAKE_SOURCE_DIR}/Includes)
