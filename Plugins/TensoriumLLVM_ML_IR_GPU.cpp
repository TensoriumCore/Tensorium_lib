#include "llvm/IR/PassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
	class AVXToSIMTPass : public PassInfoMixin<AVXToSIMTPass> {
		public:
			PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
				for (Function &F : M) {
					for (auto &BB : F) {
						for (auto It = BB.begin(); It != BB.end();) {
							Instruction &I = *It++;

							auto *CI = dyn_cast<CallInst>(&I);
							if (!CI || !CI->getCalledFunction()) continue;

							StringRef CalleeName = CI->getCalledFunction()->getName();
							if (!CalleeName.contains("llvm.x86.avx.add.ps.256")) continue;

							IRBuilder<> B(CI);
							Value *A = CI->getArgOperand(0);
							Value *Bv = CI->getArgOperand(1);

							VectorType *VecTy = cast<VectorType>(A->getType());
							unsigned NumElements = VecTy->getElementCount().getFixedValue();

							SmallVector<Value *, 8> AddedElems;
							for (unsigned i = 0; i < NumElements; ++i) {
								Value *Ai = B.CreateExtractElement(A, B.getInt32(i));
								Value *Bi = B.CreateExtractElement(Bv, B.getInt32(i));
								Value *Sum = B.CreateFAdd(Ai, Bi);
								AddedElems.push_back(Sum);
							}

							Value *NewVec = UndefValue::get(VecTy);
							for (unsigned i = 0; i < NumElements; ++i) {
								NewVec = B.CreateInsertElement(NewVec, AddedElems[i], B.getInt32(i));
							}

							CI->replaceAllUsesWith(NewVec);
							CI->eraseFromParent();
						}
					}
				}
				return PreservedAnalyses::none();
			}
	};
} // namespace


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "AVXToSIMTPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(AVXToSIMTPass());
                });

            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "simd-to-gpu") {
                        MPM.addPass(AVXToSIMTPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}

