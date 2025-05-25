// tensorium-opt.cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/PassPlugin.h"          // Pour EmptyPipelineOptions
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct ConvertMemRefToLLVMPass
    : public PassWrapper<ConvertMemRefToLLVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    LLVMTypeConverter converter(&ctx);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

    ConversionTarget target(ctx);
    target.addLegalDialect("llvm");                  // dialecte LLVM
    target.addIllegalDialect<memref::MemRefDialect>(); // dialecte MemRef

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

// --- Enregistrement du pipeline CLI ----------------------
// On utilise EmptyPipelineOptions et une lambda à 1 paramètre.
static PassPipelineRegistration<EmptyPipelineOptions> pipeline(
    "convert-memref-to-llvm",
    "Convert MemRef dialect to LLVM dialect",
    [](OpPassManager &pm) {
      pm.addPass(std::make_unique<ConvertMemRefToLLVMPass>());
    });

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllPasses();
  return asMainReturnCode(
      MlirOptMain(argc, argv, "Tensorium MLIR optimizer\n", registry));
}
