#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"

using namespace llvm;

namespace {

Value *scalarOp(IRBuilder<> &B, StringRef Callee, Value *A, Value *Bv, unsigned i) {
    Value *Ai = B.CreateExtractElement(A, B.getInt32(i));
    Value *Bi = B.CreateExtractElement(Bv, B.getInt32(i));

    if (Callee.contains("add"))
        return B.CreateFAdd(Ai, Bi);
    if (Callee.contains("sub"))
        return B.CreateFSub(Ai, Bi);
    if (Callee.contains("mul"))
        return B.CreateFMul(Ai, Bi);
    if (Callee.contains("div"))
        return B.CreateFDiv(Ai, Bi);

    return nullptr;
}

class AVXToSIMTPass : public PassInfoMixin<AVXToSIMTPass> {
  public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        for (Function &F : M) {
            for (auto &BB : F) {
                for (auto It = BB.begin(); It != BB.end();) {
                    Instruction &I = *It++;

                    auto *CI = dyn_cast<CallInst>(&I);
                    if (!CI || !CI->getCalledFunction())
                        continue;

                    StringRef CalleeName = CI->getCalledFunction()->getName();

                    if (!(CalleeName.contains("llvm.x86.avx.add.ps.256") ||
                          CalleeName.contains("llvm.x86.avx.sub.ps.256") ||
                          CalleeName.contains("llvm.x86.avx.mul.ps.256") ||
                          CalleeName.contains("llvm.x86.avx.div.ps.256")))
                        continue;

                    IRBuilder<> B(CI);
                    Value      *A = CI->getArgOperand(0);
                    Value      *Bv = CI->getArgOperand(1);
                    VectorType *VecTy = cast<VectorType>(A->getType());
                    unsigned    NumElements = VecTy->getElementCount().getFixedValue();

                    SmallVector<Value *, 8> Elems;
                    for (unsigned i = 0; i < NumElements; ++i) {
                        if (Value *V = scalarOp(B, CalleeName, A, Bv, i))
                            Elems.push_back(V);
                    }

                    Value *NewVec = UndefValue::get(VecTy);
                    for (unsigned i = 0; i < NumElements; ++i) {
                        NewVec = B.CreateInsertElement(NewVec, Elems[i], B.getInt32(i));
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
    return {LLVM_PLUGIN_API_VERSION, "AVXToSIMTPass", LLVM_VERSION_STRING, [](PassBuilder &PB) {
                PB.registerPipelineStartEPCallback([](ModulePassManager &MPM, OptimizationLevel) {
                    MPM.addPass(AVXToSIMTPass());
                });

                PB.registerPipelineParsingCallback([](StringRef Name, ModulePassManager &MPM,
                                                      ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "simd-to-gpu") {
                        MPM.addPass(AVXToSIMTPass());
                        return true;
                    }
                    return false;
                });
            }};
}
