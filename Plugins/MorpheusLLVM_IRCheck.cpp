#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
class MorpheusAlignPass : public PassInfoMixin<MorpheusAlignPass> {
public:
 PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        for (Function &F : M) {
            LLVMContext &Ctx = F.getContext();
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (auto *AI = dyn_cast<AllocaInst>(&I)) {
                        unsigned align = AI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (!Loc) continue;
                            std::string Msg = "alloca has alignment = " + std::to_string(align) + " (expected >= 32)";
                            Ctx.diagnose(OptimizationRemarkAnalysis("morpheus-align", "LowAlignAlloca", Loc, &F) << Msg);
                        }
                    }
                    if (auto *LI = dyn_cast<LoadInst>(&I)) {
                        unsigned align = LI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (!Loc) continue;
                            std::string Msg = "load has alignment = " + std::to_string(align) + " (expected >= 32)";
                            Ctx.diagnose(OptimizationRemarkAnalysis("morpheus-align", "LowAlignLoad", Loc, &F) << Msg);
                        }
                    }
                    if (auto *SI = dyn_cast<StoreInst>(&I)) {
                        unsigned align = SI->getAlign().value();
                        if (align < 32) {
                            DebugLoc Loc = I.getDebugLoc();
                            if (!Loc) continue;
                            std::string Msg = "store has alignment = " + std::to_string(align) + " (expected >= 32)";
                            Ctx.diagnose(OptimizationRemarkAnalysis("morpheus-align", "LowAlignStore", Loc, &F) << Msg);
                        }
                    }
                }
            }
        }

        return PreservedAnalyses::all();
    }
};
} 


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "MorpheusAlignPass",
        LLVM_VERSION_STRING,
        [](llvm::PassBuilder &PB) {
			PB.registerPipelineStartEPCallback(
					[](llvm::ModulePassManager &MPM,  
						llvm::OptimizationLevel) { 
					MPM.addPass(MorpheusAlignPass());
					});

            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name,
                   llvm::ModulePassManager &MPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (Name == "morpheus-align") {
                        MPM.addPass(MorpheusAlignPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}

