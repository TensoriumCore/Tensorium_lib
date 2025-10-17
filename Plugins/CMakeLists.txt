find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# AST plugin
add_library(TensoriumDispatchPlugin SHARED TensoriumDispatchPlugin.cpp)
target_link_libraries(TensoriumDispatchPlugin PRIVATE clang-cpp LLVM)
set_target_properties(TensoriumDispatchPlugin PROPERTIES COMPILE_FLAGS "")

# LLVM IR plugin
add_library(TensoriumLLVM_IRCheck SHARED TensoriumLLVM_IRCheck.cpp)
target_link_libraries(TensoriumLLVM_IRCheck PRIVATE LLVM)
set_target_properties(TensoriumLLVM_IRCheck PROPERTIES COMPILE_FLAGS "")
