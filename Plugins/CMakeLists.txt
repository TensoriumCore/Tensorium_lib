find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# List of components to map to libraries
set(LLVM_COMPONENTS
    Core
    Support
    IRReader
    Analysis
    Passes
    TransformUtils
)

llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_COMPONENTS})

# Tensorium Dispatch Plugin (Clang AST)
add_library(TensoriumDispatchPlugin SHARED TensoriumDispatchPlugin.cpp)
target_link_libraries(TensoriumDispatchPlugin PRIVATE clang-cpp ${LLVM_LIBS})

# Tensorium LLVM IR Plugin
add_library(TensoriumLLVM_IRCheck SHARED TensoriumLLVM_IRCheck.cpp)
target_link_libraries(TensoriumLLVM_IRCheck PRIVATE ${LLVM_LIBS})
