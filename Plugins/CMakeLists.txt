
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Clang AST plugin
add_library(TensoriumDispatchPlugin SHARED TensoriumDispatchPlugin.cpp)
target_link_libraries(TensoriumDispatchPlugin PRIVATE clang-cpp LLVM)

# LLVM IR pass
add_library(TensoriumLLVM_IRCheck SHARED TensoriumLLVM_IRCheck.cpp)
target_link_libraries(TensoriumLLVM_IRCheck PRIVATE LLVM)

# MLIR plugin (if present)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/MLIR/CMakeLists.txt)
  add_subdirectory(MLIR)
endif()
