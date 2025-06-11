#include "LLVM_Handler.hpp"
#include "llvm/Support/raw_ostream.h"
#include <memory>

void extractFunctionToMLIR(clang::FunctionDecl *FD, clang::ASTContext &Context) {
    const std::string funcName = FD->getNameAsString();
    const std::string path = "output.ll";

    llvm::LLVMContext             llvmCtx;
    llvm::SMDiagnostic            err;
    std::unique_ptr<llvm::Module> fullMod = llvm::parseIRFile(path, err, llvmCtx);
    if (!fullMod) {
        llvm::errs() << "could not load LLVM module from " << path << "\n";
        return;
    }

    auto *func = fullMod->getFunction(funcName);
    if (!func) {
        llvm::errs() << "function " << funcName << " not found in " << path << "\n";
        return;
    }

    auto newMod = std::make_unique<llvm::Module>("tensorium_gpu_kernel", llvmCtx);
    llvm::ValueToValueMapTy VMap;
    auto                   *clonedFunc = llvm::CloneFunction(func, VMap);
    newMod->getFunctionList().push_back(clonedFunc);

    mlir::MLIRContext mlirCtx;
    mlirCtx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    mlirCtx.getOrLoadDialect<mlir::vector::VectorDialect>();
    mlirCtx.getOrLoadDialect<mlir::gpu::GPUDialect>();
    mlirCtx.getOrLoadDialect<mlir::NVVM::NVVMDialect>();

    auto mlirModule = mlir::translateLLVMIRToMLIR(mlirCtx, std::move(newMod));
    if (!mlirModule) {
        llvm::errs() << "MLIR translation failed\n";
        return;
    }

    mlir::PassManager pm(&mlirCtx);
    pm.addPass(mlir::createConvertVectorToGPU());
    pm.addPass(mlir::createGPUKernelOutliningPass());
    pm.addPass(mlir::createConvertGPUToNVVMPass());

    if (mlir::failed(pm.run(*mlirModule))) {
        llvm::errs() << "MLIR passes failed\n";
        return;
    }

    std::string          outPath = "/tmp/" + funcName + "_gpu.mlir";
    std::error_code      ec;
    llvm::raw_fd_ostream out(outPath, ec);
    if (ec) {
    llvm::errs() << "could not write to " << outPath << ": " << ec.message()
