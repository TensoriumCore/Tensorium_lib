#include "../test.hpp"

#define CHECK(expr) \
	do { \
		if (!(expr)) { \
			std::cerr << "\u274c CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(1); \
		} \
	} while (0)

int test_flatten_index() {
	using namespace tensorium;
	std::array<size_t, 4> dims = {2, 3, 4, 5};
	Tensor<float, 4> T4(dims);

	for (size_t i = 0; i < dims[0]; ++i)
		for (size_t j = 0; j < dims[1]; ++j)
			for (size_t k = 0; k < dims[2]; ++k)
				for (size_t l = 0; l < dims[3]; ++l)
					T4({i, j, k, l}) = static_cast<float>(T4.flatten_index({i, j, k, l}));


	for (size_t i = 0; i < dims[0]; ++i)
		for (size_t j = 0; j < dims[1]; ++j)
			for (size_t k = 0; k < dims[2]; ++k)
				for (size_t l = 0; l < dims[3]; ++l) {
					std::array<size_t, 4> idx = {i, j, k, l};
					size_t expected = T4.flatten_index(idx);
					T4(idx) = static_cast<float>(expected);
					CHECK(std::abs(T4(idx) - expected) < 1e-5);
				}


	std::cout << "[\u2713] flatten_index test passed\n";
	return 0;
}

int test_tensor_mul() {
	using namespace tensorium;
	Tensor<float, 2> A({2, 2}), B({2, 2});
	A({0,0}) = 1; A({0,1}) = 2; A({1,0}) = 3; A({1,1}) = 4;
	B({0,0}) = 10; B({0,1}) = 20; B({1,0}) = 30; B({1,1}) = 40;
	auto C = mul_tensor(A, B);

	float expected_vals[2][2][2][2] = {
		{{1*10, 1*20}, {2*30, 2*40}},
		{{3*10, 3*20}, {4*30, 4*40}}
	};

	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j)
			for (size_t k = 0; k < 2; ++k)
				for (size_t l = 0; l < 2; ++l)

	std::cout << "[\u2713] Tensor product test passed\n";
	return 0;
}

int benchmark_tensor_mul() {
	using namespace tensorium;
	constexpr size_t N = 128;
	Tensor<float, 2> A({N, N}), B({N, N});
	A.fill(1.0f); B.fill(2.0f);

	auto start = std::chrono::high_resolution_clock::now();
	auto C = mul_tensor(A, B);
	auto end = std::chrono::high_resolution_clock::now();

	double seconds = std::chrono::duration<double>(end - start).count();
	std::cout << "[\u2713] Benchmark completed in " << seconds << " s\n";

	float error = 0;
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			for (size_t k = 0; k < N; ++k)
				for (size_t l = 0; l < N; ++l)
					error += std::abs(C({i,j,k,l}) - 2.0f);

	std::cout << "Total absolute error: " << error << "\n";
	return 0;
}

int test_contract() {
	using namespace tensorium;
	Tensor<float, 3> T3({2, 2, 2});
	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j)
			for (size_t k = 0; k < 2; ++k)
				T3({i,j,k}) = float(i + j + k);

	auto T_c = T3.contract_tensor<1, 2>();

	for (size_t i = 0; i < 2; ++i) {
		float expected = T3({i,0,0}) + T3({i,1,1});
		CHECK(std::abs(T_c({i}) - expected) < 1e-5);
	}

	std::cout << "[\u2713] contract_tensor test passed\n";
	return 0;
}

int tensor_test() {
	CHECK(test_flatten_index() == 0);
	CHECK(test_tensor_mul() == 0);
	CHECK(test_contract() == 0);
	CHECK(benchmark_tensor_mul() == 0);
	std::cout << "\n\u2705 All Tensor tests passed successfully.\n";
	return 0;
}
