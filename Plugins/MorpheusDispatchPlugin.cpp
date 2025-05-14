#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/Support/MemoryBuffer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
/**
 * @file MorpheusPlugin.cpp
 * @brief Clang plugin for detecting alignment issues and handling custom Morpheus pragmas.
 */
using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

/**
 * @brief AST consumer that matches specific patterns in the AST related to Morpheus usage.
 */
class MorpheusASTConsumer : public ASTConsumer {
public:
	/**
     * @brief Constructor
     * @param CI The Clang compiler instance
     */
    explicit MorpheusASTConsumer(CompilerInstance &CI) : CI(CI) {}
    /**
     * @brief Called once AST is fully parsed; sets up matchers
     * @param Context ASTContext of the parsed translation unit
     */
    void HandleTranslationUnit(ASTContext &Context) override {
        MatchFinder *Finder = new MatchFinder();
        auto *Checker = new AlignedChecker(CI);

        Finder->addMatcher(
            varDecl(
                hasType(recordDecl(hasName("::std::vector"))),
                unless(hasType(TypeMatcher(hasDescendant(recordDecl(hasName("aligned_vector"))))))
            ).bind("unaligned_vector"),
            Checker
        );

        Finder->addMatcher(
            varDecl(
                hasType(pointerType(pointee(builtinType())))
            ).bind("raw_pointer"),
            Checker
        );

        Finder->addMatcher(
            varDecl(
                hasType(qualType(hasDeclaration(classTemplateSpecializationDecl(
                    matchesName("::morpheus::(Vector|Matrix|Tensor)")
                ))))
            ).bind("morpheus_type"),
            Checker
        );

        Finder->addMatcher(
            varDecl(
                unless(hasType(qualType(hasDeclaration(classTemplateSpecializationDecl(
                    matchesName("::morpheus::(Vector|Matrix|Tensor)")
                )))))
            ).bind("non_morpheus"),
            Checker
        );

        Finder->matchAST(Context);
    }

private:
    CompilerInstance &CI;

    /**
     * @brief Callback for handling matches from the AST.
     */
    class AlignedChecker : public MatchFinder::MatchCallback {
    public:
	     /**
         * @brief Constructor
         * @param CI The Clang compiler instance
         */
        explicit AlignedChecker(CompilerInstance &CI) : CI(CI) {}
        /**
         * @brief Called for each AST match
         * @param Result Match result
         */
		void run(const MatchFinder::MatchResult &Result) override {
			const SourceManager &SM = *Result.SourceManager;

			if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("unaligned_vector")) {
				if (!SM.isWrittenInMainFile(VD->getLocation())) return;
				DiagnosticsEngine &DE = CI.getDiagnostics();
				unsigned ID = DE.getCustomDiagID(
						DiagnosticsEngine::Warning,
						"Unaligned std::vector detected; consider morpheus::aligned_vector instead"
						);
				DE.Report(VD->getLocation(), ID);
			}

			else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("raw_pointer")) {
				if (!SM.isWrittenInMainFile(VD->getLocation())) return;
				QualType QT = VD->getType();
				if (QT->isPointerType()) {
					Qualifiers Qs = QT.getQualifiers();
					if (!Qs.hasRestrict()) {
						DiagnosticsEngine &DE = CI.getDiagnostics();
						unsigned ID = DE.getCustomDiagID(
								DiagnosticsEngine::Warning,
								"Raw pointer without __restrict qualifier or alignment"
								);
						DE.Report(VD->getLocation(), ID);
					}
				}
			}


		
			else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("non_morpheus")) {
				const SourceManager &SM = *Result.SourceManager;
				if (!SM.isWrittenInMainFile(VD->getLocation()))
					return;

				QualType CanonQT = VD->getType().getCanonicalType();
				const CXXRecordDecl *RD = CanonQT->getAsCXXRecordDecl();

				if (!RD) return;

				std::string canonicalName = RD->getQualifiedNameAsString();

				if (canonicalName.find("morpheus::Vector") == 0 ||
						canonicalName.find("morpheus::Matrix") == 0 ||
						canonicalName.find("morpheus::Tensor") == 0 ||
						canonicalName.find("morpheus::Derivate") == 0)
					return;

				if (canonicalName.find("std::") == 0) return;

				DiagnosticsEngine &DE = CI.getDiagnostics();
				unsigned ID = DE.getCustomDiagID(
						DiagnosticsEngine::Remark,
						"Variable not using Morpheus aligned types (Vector/Matrix/Tensor)"
						);
				DE.Report(VD->getLocation(), ID);
			}

		}


	private:
		CompilerInstance &CI;
	};
};

namespace {

/**
 * @brief Handles `#pragma morpheus ...` pragmas for code injection.
 */
class MorpheusPragmaHandler : public PragmaHandler {
public:
    MorpheusPragmaHandler() : PragmaHandler("morpheus") {}
    /**
     * @brief Handles a custom `#pragma morpheus` directive.
     * @param PP Preprocessor instance
     * @param Introducer Pragma introducer
     * @param Tok Token after 'morpheus'
     */
    void HandlePragma(Preprocessor &PP, PragmaIntroducer, Token &Tok) override {
        PP.Lex(Tok);
        if (Tok.isNot(tok::identifier)) return;
        llvm::StringRef kind = Tok.getIdentifierInfo()->getName();

        if (kind == "dispatch") injectDispatch(PP);
        else if (kind == "restrict") injectRestrict(PP);
		else if (kind == "target") parseTarget(PP);
    }

private:
	 /**
     * @brief Push a string as a virtual buffer for the preprocessor to consume.
     * @param PP Preprocessor
     * @param Code C++ code to inject
     * @param Name Virtual file name
     */
    static void pushBuffer(Preprocessor &PP, llvm::StringRef Code, llvm::StringRef Name) {
        auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Code, Name);
        FileID F = PP.getSourceManager().createFileID(std::move(Buf));
        PP.EnterSourceFile(F, nullptr, SourceLocation());
    }
    /**
     * @brief Injects SIMD dispatch code.
     * @param PP Preprocessor
     */
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
/**
 * @brief Main plugin action class for Morpheus plugin.
 */
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


static void parseTarget(Preprocessor &PP) {
	Token Tok;
	PP.Lex(Tok); // '('
	if (Tok.isNot(tok::l_paren)) return;

	PP.Lex(Tok);
	if (Tok.isNot(tok::identifier)) return;

	std::string platform = PP.getSpelling(Tok);
	std::string isa;

	PP.Lex(Tok);
	if (Tok.is(tok::comma)) {
		PP.Lex(Tok);
		if (Tok.isNot(tok::identifier)) return;
		isa = PP.getSpelling(Tok);
		PP.Lex(Tok); // ')'
	}

	if (Tok.isNot(tok::r_paren)) return;

	// On va juste imprimer pour l’instant (à terme stocker dans une table)
	llvm::errs() << "[morpheus] target detected: " << platform;
	if (!isa.empty()) llvm::errs() << " with ISA " << isa;
	llvm::errs() << "\n";

	// Tu pourrais ensuite stocker cette info ici
	// e.g. MorpheusTargetTable.push_back({platform, isa, location})
}

};
/**
 * @brief Main plugin action class for Morpheus plugin.
 */
class MorpheusPluginAction : public PluginASTAction {
	protected:
		/**
		 * @brief Parse plugin arguments (none used here).
		 */
		bool ParseArgs(const CompilerInstance&, const std::vector<std::string>&) override { return true; }
		/**
		 * @brief Create the AST consumer.
		 */
		std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
			CI.getPreprocessor().AddPragmaHandler(new MorpheusPragmaHandler);
			return std::make_unique<MorpheusASTConsumer>(CI);
		}

		/**
		 * @brief Main plugin execution (unused here).
		 */
		void ExecuteAction() override {}
};

}

/// @brief Register the plugin under the name "morpheus-dispatch"
static FrontendPluginRegistry::Add<MorpheusPluginAction>
X("morpheus-dispatch", "Handle #pragma morpheus directives");
