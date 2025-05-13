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
#pragma morpheus dispatch
	float __attribute__((aligned(8))) low_align[32];

	deriv_test();	
	linear_solver_test();
	matrix_tests();
	tensor_test();
	vector_tests();

	return 0;
}
