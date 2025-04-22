#include "../../includes/Morpheus/Morpheus.hpp"

using namespace morpheus;

int comb() {
	using ISA = morpheus::avx2_t;
	constexpr size_t W = ISA::width;
	using reg = ISA::reg;

	std::cout << "\n=== Test linear_combination() ===\n";

	std::vector<morpheus::Vector<float>> basis = {
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	};

	std::vector<float> coefs = {2.0f, -1.0f, 3.5f};

	auto result = morpheus::linear_combination_vec(basis, coefs);

	std::cout << "Expected result: [2, -1, 3.5, 0, ..., 0]\n";
	std::cout << "Result vector:\n";
	result.print();

	std::cout << "\n=== Test linear_combination() with different sizes ===\n";

	auto a = morpheus::Vector<float>{0, 0, 0, 0, 0, 0, 0, 0};
	auto b = morpheus::Vector<float>{1, 1, 1, 1, 1, 1, 1, 1};

	auto mid = morpheus::lerp_vec(a, b, 0.5f);
		mid.print();

	return 0;
}
