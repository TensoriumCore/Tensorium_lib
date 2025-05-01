#include "test.hpp"

using namespace morpheus;



void test_lexer(const std::string& input) {
	Lexer lexer(input);
	auto tokens = lexer.tokenize();

	std::cout << "Input: " << input << "\n";
	for (const auto& token : tokens) {
		std::cout << "Token: \"" << token.value << "\" — type: " << static_cast<int>(token.type) << "\n";
	}
	std::cout << "------\n";
}

int main() {
	dispatch_simd([](auto simd) {
			using T = decltype(simd);
			constexpr size_t W = T::width;
			constexpr size_t A = T::alignment;
			std::cout << "SIMD selected: width=" << W << ", alignment=" << A << "\n";
			});
	/* Lexer lex(R"( \int x^2 + \partial_y T )"); */
	/* for (const auto& tok : lex.tokenize()) { */
	/* 	std::cout << "Token: " << tok.value << " — type: " << static_cast<int>(tok.type) << "\n"; */
	/* } */
	/* test_lexer("\\int x^2 + \\partial_y T"); */
	/* test_lexer("\\alpha + \\beta - 3.14 * \\Gamma"); */
	/* test_lexer("A_{\\mu\\nu} B^{\\rho\\sigma}"); */
	/* test_lexer("f(x) = x^2 + 2*x + 1"); */
	/* test_lexer("\\cdot \\otimes T"); */
	deriv_test();	
	linear_solver_test();
	matrix_tests();
	tensor_test();
	vector_tests();

	return 0;
}
