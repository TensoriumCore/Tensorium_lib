#pragma once
#include <iostream>
#include <vector>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/NVVM/IR/NVVMDialect.h"
#include <string>

struct MorpheusTarget {
	std::string platform;
	std::string isa;
	clang::SourceLocation loc;
};

static std::vector<MorpheusTarget> MorpheusTargetTable;

