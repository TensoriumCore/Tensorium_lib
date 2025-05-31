cmake_minimum_required(VERSION 3.13)
project(ConvertMemRefToLLVMPlugin)

# Trouver MLIR (on pointe MLIR_DIR vers le dossier contenant MLIRConfig.cmake)
find_package(MLIR REQUIRED CONFIG
  PATHS ${MLIR_DIR}
  NO_DEFAULT_PATH
)

# Notre plugin en tant que module partagé
add_library(ConvertMemRefToLLVMPlugin MODULE
  ConvertMemRefToLLVMPlugin.cpp
)

# On veut un .so sans le préfixe "lib"
set_target_properties(ConvertMemRefToLLVMPlugin PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ConvertMemRefToLLVMPlugin"
)

# Lier uniquement les libs MLIR nécessaires
target_link_libraries(ConvertMemRefToLLVMPlugin
  PRIVATE
    MLIRCore
    MLIRConversion
    MLIRTransforms
    MLIRPass
    MLIRSupport
)

# Include paths MLIR
target_include_directories(ConvertMemRefToLLVMPlugin
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
)
