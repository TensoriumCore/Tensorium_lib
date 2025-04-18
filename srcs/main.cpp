
#include "../includes/Morpheus/Morpheus.hpp"

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


int test_flatten_index()
{
	using namespace morpheus;

	std::array<size_t, 4> dims = {2, 3, 4, 5};
	Tensor<float, 4> T4(dims);

	for (size_t i = 0; i < dims[0]; ++i)
		for (size_t j = 0; j < dims[1]; ++j)
			for (size_t k = 0; k < dims[2]; ++k)
				for (size_t l = 0; l < dims[3]; ++l) {
					std::array<size_t, 4> idx = {i, j, k, l};
					T4(idx) = static_cast<float>(T4.flatten_index(idx));
				}

	bool ok = true;
	for (size_t i = 0; i < dims[0]; ++i)
		for (size_t j = 0; j < dims[1]; ++j)
			for (size_t k = 0; k < dims[2]; ++k)
				for (size_t l = 0; l < dims[3]; ++l) {
					std::array<size_t, 4> idx = {i, j, k, l};
					size_t flat_expected = T4.flatten_index(idx);
					size_t flat_simd = T4.flatten_index_simd(idx.data(), T4.strides.data());

					if (flat_expected != flat_simd) {
						std::cerr << "Mismatch at " << i << "," << j << "," << k << "," << l
							<< " : expected " << flat_expected << ", got " << flat_simd << "\n";
						ok = false;
					}
				}

	if (ok)
		std::cout << "[✓] flatten_index tests passed\n";
	else
		std::cout << "[x] flatten_index tests failed\n";

	return ok ? 0 : 1;
}


int bench() {
	constexpr size_t N = 8192;
	morpheus::Matrix<float> A(N, N);
	morpheus::Matrix<float> B(N, N);

	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			A(i, j) = 1.0f, B(i, j) = 2.0f;

	std::cout << "\n=== Benchmarking ===\n";
	std::cout << "Benchmarking AVX2 mul_mat() for size " << N << "x" << N << "\n";

	auto start = std::chrono::high_resolution_clock::now();
	auto C = morpheus::mul_mat(A, B);
	auto end = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration<double>(end - start).count();
	std::cout << "Time: " << elapsed << " seconds\n";
	std::cout << "Sample result: C(0, 0) = " << C(0, 0) << "\n";

	double flops = 2.0 * N * N * N;
	double gflops = flops / (1e9 * elapsed);
	std::cout << "Performance: " << gflops << " GFLOP/s\n";

	return 0;
}

int test_contract()
{
	using namespace morpheus;

	std::array<size_t, 3> shape = {2, 2, 2};
	Tensor<float, 3> T3(shape);

	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j)
			for (size_t k = 0; k < 2; ++k)
				T3({i, j, k}) = static_cast<float>(i + j + k);

	auto T_contracted = T3.contract_tensor<1, 2>();

	std::cout << "=== Test contraction T(i,j,j) ===\n";
	for (size_t i = 0; i < 2; ++i) {
		float expected = 0.f;
		for (size_t j = 0; j < 2; ++j)
			expected += T3({i, j, j});

		std::cout << "T(" << i << ") = " << T_contracted({i}) << " (expected " << expected << ")\n";

		if (std::abs(T_contracted({i}) - expected) > 1e-5f) {
			std::cerr << "Mismatch in contraction\n";
			return 1;
		}
	}

	std::cout << "[✓] contract_simd test passed\n";
	return 0;
}


int main() {
	dispatch_simd([](auto simd) {
			using T = decltype(simd);
			constexpr size_t W = T::width;
			constexpr size_t A = T::alignment;
			std::cout << "SIMD selected: width=" << W << ", alignment=" << A << "\n";
			});


	std::cout << "\n=== Vector Tests ===\n";
	Vector<float> v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	Vector<float> v2 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

	std::cout << "\n[v1] + [v2]:\n";
	morpheus::add_vec(v1, v2).print();

	std::cout << "\n[v1] - [v2]:\n";
	morpheus::sub_vec(v1, v2).print();

	std::cout << "\n[v1] * 0.5:\n";
	morpheus::scl_vec(v1, 0.5f).print();

	std::cout << "\n=== Matrix Tests ===\n";
	Matrix<float> m1(2, 8); 
	Matrix<float> m2(2, 8);

	for (size_t i = 0; i < m1.rows; ++i)
		for (size_t j = 0; j < m1.cols; ++j) {
			m1(i, j) = i * 10 + j;
			m2(i, j) = 1.0f;
		}

	std::cout << "\n[m1] + [m2]:\n";
	morpheus::add_mat(m1, m2).print();

	std::cout << "\n[m1] - [m2]:\n";
	morpheus::sub_mat(m1, m2).print();

	std::cout << "\n[m1] * 2.0:\n";
	morpheus::scl_mat(m1, 2.0f).print();

	std::cout << "\n=== Matrix and Vector Tests ===\n";
	comb();

	Vector<float> a = {1, 2, 3, 4};
	Vector<float> b = {5, 6, 7, 8};
	float result = morpheus::dot_vec(a, b);
	std::cout << "Dot product: " << result << "\n";

	Vector<float> v = {-3, 4, -5};
	std::cout << "Norm 1  = " << morpheus::norm1_vec(v) << "\n"; 
	std::cout << "Norm 2  = " << morpheus::norm2_vec(v) << "\n";
	std::cout << "Norm ∞  = " << morpheus::normInf_vec(v) << "\n";

	std::cout << "cos(angle) = " << morpheus::cosine_vec(a, b) << "\n";

	Vector<float> c = {1, 1};
	Vector<float> d = {2, 2};
	std::cout << "cos(angle) = " << morpheus::cosine_vec(c, d) << "\n";

	std::cout << "\n=== Test cross_product() ===\n";
	Vector<float> u3 = {1, 0, 0};
	Vector<float> v3 = {0, 1, 0};
	std::cout << "u = [1, 0, 0]\n";
	std::cout << "v = [0, 1, 0]\n";
	std::cout << "u × v = ";

	Matrix<float> A(3, 3);
	Matrix<float> B(3, 3);

	A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
	A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
	A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;

	B(0, 0) = 1.0; B(0, 1) = 0.0; B(0, 2) = 0.0;
	B(1, 0) = 0.0; B(1, 1) = 1.0; B(1, 2) = 0.0;
	B(2, 0) = 0.0; B(2, 1) = 0.0; B(2, 2) = 1.0;

	Matrix<float> C = morpheus::mul_mat(A, B);
	std::cout << "\n=== Matrix Multiplication (Analytic Test) ===\n";

	Matrix<float> M1(2, 2);
	Matrix<float> M2(2, 2);

	M1(0, 0) = 1.0f; M1(0, 1) = 2.0f;
	M1(1, 0) = 3.0f; M1(1, 1) = 4.0f;

	M2(0, 0) = 5.0f; M2(0, 1) = 6.0f;
	M2(1, 0) = 7.0f; M2(1, 1) = 8.0f;

	Matrix<float> M3 = morpheus::mul_mat(M1, M2);

	std::cout << "Expected:\n[19 22]\n[43 50]\n";
	std::cout << "Result:\n";
	M3.print();

	std::cout << "Matrix C (A * B):\n";
	C.print();
	test_transpose_matrix();

	Matrix<float> A_solve(2, 2);
	A_solve(0, 0) = 2.0; A_solve(0, 1) = 1.0;
	A_solve(1, 0) = 5.0; A_solve(1, 1) = 7.0;
	Vector<float> b_solve = { 11.0, 13.0 };

	Vector<float> x_solve = morpheus::gauss_solve(A_solve, b_solve);
	std::cout << "Solution x:\n";
	x_solve.print();

	Vector<float> b_check = A_solve * x_solve;
	std::cout << "Check Ax = b:\n";
	b_check.print();
	

	Matrix<float> A2_solve(3, 3);
	A2_solve(0, 0) = 10.0f; A2_solve(0, 1) = -1.0f; A2_solve(0, 2) = 2.0f;
	A2_solve(1, 0) = -1.0f; A2_solve(1, 1) = 11.0f; A2_solve(1, 2) = -1.0f;
	A2_solve(2, 0) = 2.0f;  A2_solve(2, 1) = -1.0f; A2_solve(2, 2) = 10.0f;

	Vector<float> b2_solve = { 6.0f, 25.0f, -11.0f };

	Vector<float> x2_solve = morpheus::jacobi_solve(A2_solve, b2_solve);
	std::cout << "Solution x2:\n";	
	x2_solve.print();
	Vector<float> b2_check = morpheus::mul_vec(A2_solve, x2_solve);
	std::cout << "Check Ax2 = b2:\n";
	b2_check.print();

	std::cout << "\n=== Benchmarking ===\n";
	bench();
	std::cout << "\n=== Test contraction ===\n";
	test_flatten_index();
	test_contract();
	Matrix<float> expected(2, 2);
	expected(0, 0) = 19.0f; expected(0, 1) = 22.0f;
	expected(1, 0) = 43.0f; expected(1, 1) = 50.0f;

	bool ok = true;
	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j)
			if (std::fabs(M3(i, j) - expected(i, j)) > 1e-4f) {
				ok = false;
				std::cout << "Mismatch at (" << i << ", " << j << "): got " << M3(i, j) << ", expected " << expected(i, j) << "\n";
			}

	if (ok)
		std::cout << "[PASS] Matrix multiplication result is correct.\n";
	else
		std::cerr << "[FAIL] Matrix multiplication mismatch.\n";

	return 0;
}
