#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace clang;

namespace {

class MorpheusPragmaHandler : public PragmaHandler {
public:
    MorpheusPragmaHandler() : PragmaHandler("morpheus") {}

    void HandlePragma(Preprocessor &PP, PragmaIntroducer, Token &Tok) override {
        PP.Lex(Tok);
        if (Tok.isNot(tok::identifier)) return;
        llvm::StringRef kind = Tok.getIdentifierInfo()->getName();

        if (kind == "dispatch") injectDispatch(PP);
        else if (kind == "restrict") injectRestrict(PP);
    }

private:
    static void pushBuffer(Preprocessor &PP, llvm::StringRef Code, llvm::StringRef Name) {
        auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code, Name);
        FileID F = PP.getSourceManager().createFileID(std::move(Buf));
        PP.EnterSourceFile(F, nullptr, SourceLocation());
    }

    static void injectDispatch(Preprocessor &PP) {
        Token T;
        do { PP.Lex(T); } while (T.isNot(tok::eod));
        pushBuffer(PP,
R"cpp(
dispatch_simd([](auto simd){
    using T = decltype(simd);
    constexpr size_t W = T::width;
    constexpr size_t A = T::alignment;
    std::cout << "SIMD selected: width=" << W
              << ", alignment=" << A << "\n";
});
)cpp",
                   "morph_dispatch");
    }

    static void injectRestrict(Preprocessor &PP) {
        Token T;
        PP.Lex(T);                     // '('
        std::string code;
        while (true) {
            PP.Lex(T);
            if (T.is(tok::identifier)) {
                llvm::StringRef v = T.getIdentifierInfo()->getName();
                code += "auto * __restrict " + v.str() + "_re = " + v.str() + ";\n";
            }
            PP.Lex(T);
            if (T.is(tok::r_paren)) break;
            if (T.isNot(tok::comma)) return;
        }
        do { PP.Lex(T); } while (T.isNot(tok::eod));
        if (!code.empty()) pushBuffer(PP, code, "morph_restrict");
    }
};

class MorpheusPluginAction : public PluginASTAction {
protected:
    bool ParseArgs(const CompilerInstance&, const std::vector<std::string>&) override { return true; }
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
        CI.getPreprocessor().AddPragmaHandler(new MorpheusPragmaHandler);
        return std::make_unique<ASTConsumer>();
    }
    void ExecuteAction() override {}
};

}

static FrontendPluginRegistry::Add<MorpheusPluginAction>
X("morpheus-dispatch", "Handle #pragma morpheus directives");
