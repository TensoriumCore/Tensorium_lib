#include "../../includes/Morpheus/Morpheus.hpp"

using namespace morpheus;

void test_transpose_matrix() {
	std::cout << "\n=== Test Matrix::transpose() ===\n";

	std::vector<std::pair<size_t, size_t>> sizes = {
		{1, 1}, {1, 5}, {5, 1},
		{2, 3}, {3, 2},
		{4, 4},
		{8, 8}, {16, 16},
		{5, 7}, {7, 5}
	};

	for (auto [rows, cols] : sizes) {
		Matrix<float> A(rows, cols);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				A(i, j) = static_cast<float>(i * cols + j);

		Matrix<float> At = morpheus::transpose_mat(A);

		std::cout << "\nOriginal (" << rows << "x" << cols << "):\n";
		A.print();
		std::cout << "Transposed (" << At.rows << "x" << At.cols << "):\n";
		At.print();

		bool ok = true;
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				if (A(i, j) != At(j, i)) {
					std::cerr << "Mismatch at (" << i << ", " << j << "): " 
					          << A(i, j) << " != " << At(j, i) << "\n";
					ok = false;
				}
			}
		}

		std::cout << (ok ? "[PASS]" : "[FAIL]") << " for size " << rows << "x" << cols << "\n";
	}
}
