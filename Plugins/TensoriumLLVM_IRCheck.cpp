#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace llvm;

namespace {
class TensoriumAlignPass : public PassInfoMixin<TensoriumAlignPass> {
  public:
    bool isExternalLibFile(const DebugLoc &Loc) {
        if (!Loc)
            return true;

        std::string File = Loc->getFilename().str();
        std::string Dir = Loc->getDirectory().str();
        std::string FullPath = Dir + "/" + File;

        return !(FullPath.find("/nix/store/") == 0 || FullPath.find("/usr/include") == 0 ||
                 // FullPath.find("/usr/lib") == 0 ||
                 FullPath.find("/opt/") == 0 || FullPath.find("/Applications/Xcode.app") == 0 ||
                 FullPath.find("/Library/Developer") == 0);
    }

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        for (Function &F : M) {
            LLVMContext &Ctx = F.getContext();
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (auto *AI = dyn_cast<AllocaInst>(&I)) {
                        unsigned align = AI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (isExternalLibFile(Loc))
                                continue;
                            if (!Loc)
                                continue;
                            std::string Msg = "alloca has alignment = " + std::to_string(align) +
                                              " (recommended 32 if you want to enforce AVX vectorization, )";
                            Ctx.diagnose(OptimizationRemarkAnalysis("tensorium-align",
                                                                    "LowAlignAlloca", Loc, &F)
                                         << Msg);
                        }
                    }
                    if (auto *LI = dyn_cast<LoadInst>(&I)) {
                        unsigned align = LI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (isExternalLibFile(Loc))
                                continue;
                            if (!Loc)
                                continue;
                            std::string Msg = "load has alignment = " + std::to_string(align) +
                                              " (expected >= 32)";
                            Ctx.diagnose(OptimizationRemarkAnalysis("tensorium-align",
                                                                    "LowAlignLoad", Loc, &F)
                                         << Msg);
                        }
                    }
                    if (auto *SI = dyn_cast<StoreInst>(&I)) {
                        unsigned align = SI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (isExternalLibFile(Loc))
                                continue;
                            if (!Loc)
                                continue;
                            std::string Msg = "store has alignment = " + std::to_string(align) +
                                              " (expected >= 32)";
                            Ctx.diagnose(OptimizationRemarkAnalysis("tensorium-align",
                                                                    "LowAlignStore", Loc, &F)
                                         << Msg);
                        } else if(align >= 32 && align < 64) {
                                DebugLoc Loc = I.getDebugLoc();
                                if (isExternalLibFile(Loc))
                                    continue;
                                if (!Loc)
                                    continue;
                                std::string Msg = "store has alignment = " + std::to_string(align) +
                                                  " (expected >= 64)";
                                Ctx.diagnose(OptimizationRemarkAnalysis("tensorium-align",
                                                                        "LowAlignStore", Loc, &F)
                                             << Msg);
                            }
                    }
                }
            }
        }

        return PreservedAnalyses::all();
    }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "TensoriumAlignPass", LLVM_VERSION_STRING,
            [](llvm::PassBuilder &PB) {
                PB.registerPipelineStartEPCallback(
                    [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
                        MPM.addPass(TensoriumAlignPass());
                    });

                PB.registerPipelineParsingCallback(
                    [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
                       llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                        if (Name == "tensorium-align") {
                            MPM.addPass(TensoriumAlignPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
