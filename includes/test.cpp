
#include "Vector.h"
#include "Matrix.h"

int comb() {

	using ISA = avx2_t;
	const size_t W = ISA::width;
	using reg = ISA::reg;

	// maintenant tu peux utiliser _mm256_* sans souci


	std::cout << "\n=== Test linear_combination() ===\n";

	std::vector<Vector<float>> basis = {
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	};

	std::vector<float> coefs = {2.0f, -1.0f, 3.5f};

	Vector<float> result = Vector<float>::linear_combination(basis, coefs);

	std::cout << "Expected result: [2, -1, 3.5, 0, ..., 0]\n";
	std::cout << "Result vector:\n";
	result.print();
	std::cout << "\n=== Test linear_combination() with different sizes ===\n";

	Vector<float> a = {0, 0, 0, 0, 0, 0, 0, 0};
	Vector<float> b = {1, 1, 1, 1, 1, 1, 1, 1};

	Vector<float> mid = Vector<float>::lerp(a, b, 0.5f);
	mid.print(); 
	return 0;
}


int bench() {
	size_t N = 8192;
	Matrix<float> A(N, N);
	Matrix<float> B(N, N);
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			A(i, j) = 1.0f, B(i, j) = 2.0f;

	std::cout << "\n=== Benchmarking ===\n";
	std::cout << "Benchmarking AVX2 mul_mat() for size " << N << "x" << N << "\n";
	auto start = std::chrono::high_resolution_clock::now();
	Matrix<float> C = A.mul_mat(B);
	auto end = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration<double>(end - start).count();
	std::cout << "Time: " << elapsed << " seconds\n";
	std::cout << "Sample result: C(0, 0) = " << C(0, 0) << "\n";

	double flops = 2.0 * N * N * N;
	double gflops = flops / (1e9 * elapsed);
	std::cout << "Performance: " << gflops << " GFLOP/s\n";

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
	v1.add(v2);
	v1.print();

	std::cout << "\n[v1] - [v2]:\n";
	v1.sub(v2);
	v1.print();

	std::cout << "\n[v1] * 0.5:\n";
	v1.scl(0.5f);
	v1.print();

	std::cout << "\n=== Matrix Tests ===\n";
	Matrix<float> m1(2, 8); 
	Matrix<float> m2(2, 8);

	for (size_t i = 0; i < m1.rows; ++i)
		for (size_t j = 0; j < m1.cols; ++j) {
			m1(i, j) = i * 10 + j;
			m2(i, j) = 1.0f;
		}

	std::cout << "\n[m1] + [m2]:\n";
	m1.add(m2);
	m1.print();

	std::cout << "\n[m1] - [m2]:\n";
	m1.sub(m2);
	m1.print();

	std::cout << "\n[m1] * 2.0:\n";
	m1.scl(2.0f);
	m1.print();
	std::cout << "\n=== Matrix and Vector Tests ===\n";
	comb();

	Vector<float> a = {1, 2, 3, 4};
	Vector<float> b = {5, 6, 7, 8};
	float result = a.dot(b);
	std::cout << "Dot product: " << result << "\n";
	Vector<float> v = { -3, 4, -5 };

	std::cout << "Norm 1  = " << v.norm_1() << "\n";  
	std::cout << "Norm 2  = " << v.norm_2() << "\n";  
	std::cout << "Norm ∞  = " << v.norm_inf() << "\n";
	
	std::cout << "cos(angle) = " << Vector<float>::angle_cos(a, b) << "\n";

	Vector<float> c = {1, 1};
	Vector<float> d = {2, 2};
	std::cout << "cos(angle) = " << Vector<float>::angle_cos(c, d) << "\n"; // ≈ 1

	std::cout << "\n=== Test cross_product() ===\n";
	Vector<float> u3 = {1, 0, 0};
	Vector<float> v3 = {0, 1, 0};

	Vector<float> cross = Vector<float>::cross_product(u3, v3);

	std::cout << "u = [1, 0, 0]\n";
	std::cout << "v = [0, 1, 0]\n";
	std::cout << "u × v = ";
	cross.print(); 
    Matrix<float> A(3, 3);
    Matrix<float> B(3, 3);

    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;

    B(0, 0) = 1.0; B(0, 1) = 0.0; B(0, 2) = 0.0;
    B(1, 0) = 0.0; B(1, 1) = 1.0; B(1, 2) = 0.0;
    B(2, 0) = 0.0; B(2, 1) = 0.0; B(2, 2) = 1.0;

    Matrix<float> C = A.mul_mat(B);

    std::cout << "Matrix C (A * B):\n";
    C.print();
	std::cout << "\n=== Benchmarking ===\n";
	bench();
	return 0;
}

