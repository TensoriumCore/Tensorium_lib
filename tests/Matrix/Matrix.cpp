#include "../test.hpp"
using namespace morpheus;


#define CHECK(expr) \
	do { \
		if (!(expr)) { \
			std::cerr << "\u274c CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(1); \
		} \
	} while (0)


int matrix_bench() {
    constexpr std::size_t N = 8192;
    morpheus::Matrix<float> A(N, N);
    morpheus::Matrix<float> B(N, N);

#pragma omp parallel
    {
        std::mt19937 rng(42 + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.f, 1.f);

#pragma omp for schedule(static)
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                A(i, j) = dist(rng);
                B(i, j) = dist(rng);
            }
    }

    std::cout << "\n=== Benchmarking ===\n"
              << "Matrix size: " << N << " x " << N << '\n';

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto C = morpheus::mul_mat(A, B);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
	std::cout << std::fixed << std::setprecision(3);
    const long double flops = 2.0L * N * N * N;    
    const double gflops = static_cast<double>(flops / 1.0e9L) / elapsed;

    std::cout << "Time      : " << elapsed << "  s\n";
    std::cout << "GFLOP/s   : " << gflops  << '\n';
    std::cout << "Sample C(0,0): " << C(0, 0) << '\n';
	

	std::cout << "[✓] Benchmark test passed\n";
	std::cout << "=== Benchmarking complete ===\n";

	std::cout << "\n === test on a little matrix ===\n";
	Matrix<float> A2(2, 2);
	Matrix<float> B2(2, 2);
	A2(0, 0) = 1.0f; A2(0, 1) = 2.0f;
	A2(1, 0) = 3.0f; A2(1, 1) = 4.0f;
	B2(0, 0) = 5.0f; B2(0, 1) = 6.0f;
	B2(1, 0) = 7.0f; B2(1, 1) = 8.0f;
	auto C2 = morpheus::mul_mat(A2, B2);
	std::cout << "C2(0,0): " << C2(0, 0) << "\n";
	std::cout << "C2(0,1): " << C2(0, 1) << "\n";
	std::cout << "C2(1,0): " << C2(1, 0) << "\n";
	std::cout << "C2(1,1): " << C2(1, 1) << "\n";
	std::cout << "=== Benchmarking complete ===\n";
	std::cout << "[✓] Benchmark test passed\n";
	
	Matrix<float> D(4, 4);
	D(0, 0) = 1.0f; D(0, 1) = 2.0f; D(0, 2) = 3.0f; D(0, 3) = 4.0f;
	D(1, 0) = 5.0f; D(1, 1) = 6.0f; D(1, 2) = 7.0f; D(1, 3) = 8.0f;
	D(2, 0) = 9.0f; D(2, 1) = 10.0f; D(2, 2) = 11.0f; D(2, 3) = 12.0f;
	D(3, 0) = 13.0f; D(3, 1) = 14.0f; D(3, 2) = 15.0f; D(3, 3) = 16.0f;
	Matrix<float> E(4, 4);
	E(0, 0) = 1.0f; E(0, 1) = 2.0f; E(0, 2) = 3.0f; E(0, 3) = 4.0f;
	E(1, 0) = 5.0f; E(1, 1) = 6.0f; E(1, 2) = 7.0f; E(1, 3) = 8.0f;
	E(2, 0) = 9.0f; E(2, 1) = 10.0f; E(2, 2) = 11.0f; E(2, 3) = 12.0f;
	E(3, 0) = 13.0f; E(3, 1) = 14.0f; E(3, 2) = 15.0f; E(3, 3) = 16.0f;
	
	auto F = morpheus::mul_mat(D, E);
	std::cout << "F(0,0): " << F(0, 0) << "\n";
	std::cout << "F(0,1): " << F(0, 1) << "\n";
	std::cout << "F(1,0): " << F(1, 0) << "\n";
	std::cout << "F(1,1): " << F(1, 1) << "\n";
	std::cout << "F(2,0): " << F(2, 0) << "\n";
	std::cout << "F(2,1): " << F(2, 1) << "\n";
	std::cout << "F(3,0): " << F(3, 0) << "\n";
	std::cout << "F(3,1): " << F(3, 1) << "\n";
	std::cout << "=== Benchmarking complete ===\n";

    return 0;
}

int matrix_tests() {
	using Mat = Matrix<float>;
	using Vec = Vector<float>;

	Mat A(2, 2);
	A(0, 0) = 1.0f; A(0, 1) = 2.0f;
	A(1, 0) = 3.0f; A(1, 1) = 4.0f;

	Mat B(2, 2);
	B(0, 0) = 5.0f; B(0, 1) = 6.0f;
	B(1, 0) = 7.0f; B(1, 1) = 8.0f;

	Mat C = A;
	C.add(B);
	CHECK(std::abs(C(0, 0) - 6.0f) < 1e-4);
	CHECK(std::abs(C(1, 1) - 12.0f) < 1e-4);

	C.sub(B);
	CHECK(std::abs(C(0, 0) - A(0, 0)) < 1e-4);

	C.scl(2.0f);
	CHECK(std::abs(C(0, 0) - 2.0f * A(0, 0)) < 1e-4);

	Vec x = {1, 1};
	Vec y = A.mul_vec(x);
	CHECK(std::abs(y[0] - 3.0f) < 1e-4);
	CHECK(std::abs(y[1] - 7.0f) < 1e-4);

	Vec z = A * x;
	CHECK(std::abs(z[0] - y[0]) < 1e-4);
	CHECK(std::abs(z[1] - y[1]) < 1e-4);

	Mat T = A.transpose();
	CHECK(std::abs(T(0, 1) - A(1, 0)) < 1e-4);
	CHECK(std::abs(T(1, 0) - A(0, 1)) < 1e-4);

	Mat S = A;
	S.swap_rows(0, 1);
	CHECK(std::abs(S(0, 0) - A(1, 0)) < 1e-4);
	CHECK(std::abs(S(1, 0) - A(0, 0)) < 1e-4);

	Mat tr = A.trace();
	CHECK(std::abs(tr(0, 0) - (A(0, 0) + A(1, 1))) < 1e-4);

	Mat M = A.mul_mat(B);
	CHECK(std::abs(M(0, 0) - (1*5 + 2*7)) < 1e-4);
	CHECK(std::abs(M(1, 1) - (3*6 + 4*8)) < 1e-4);
	std::cout << "Inverse test\n";
    A(0, 0) = 4.0f; A(0, 1) = 7.0f;
    A(1, 0) = 2.0f; A(1, 1) = 6.0f;

    auto A_inv = inverse_mat(A);
	CHECK(std::abs(A_inv(0, 0) - 0.6f) < 1e-3f);
	CHECK(std::abs(A_inv(0, 1) - (-0.7f)) < 1e-3f);
	CHECK(std::abs(A_inv(1, 0) - (-0.2f)) < 1e-3f);
	CHECK(std::abs(A_inv(1, 1) - 0.4f) < 1e-3f);
	CHECK(std::abs(A_inv(0, 0) * A(0, 0) + A_inv(0, 1) * A(1, 0) - 1.0f) < 1e-3f);
	CHECK(std::abs(A_inv(0, 0) * A(0, 1) + A_inv(0, 1) * A(1, 1)) < 1e-3f);
	CHECK(std::abs(A_inv(1, 0) * A(0, 0) + A_inv(1, 1) * A(1, 0)) < 1e-3f);

	std::cout << "Inverse test\n";
	std::cout << "A:\n";
	for (std::size_t i = 0; i < 2; ++i) {
		for (std::size_t j = 0; j < 2; ++j) {
			std::cout << A(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "A_inv:\n";
	for (std::size_t i = 0; i < 2; ++i) {
		for (std::size_t j = 0; j < 2; ++j) {
			std::cout << A_inv(i, j) << " ";
		}
		std::cout << "\n";
	}

    Mat Id = A.mul_mat(A_inv);
    CHECK(std::abs(Id(0, 0) - 1.0f) < 1e-3f);
    CHECK(std::abs(Id(1, 1) - 1.0f) < 1e-3f);
    CHECK(std::abs(Id(0, 1)) < 1e-3f);
    CHECK(std::abs(Id(1, 0)) < 1e-3f);
	std::cout << "Determinant test\n";
	A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    auto d = det_mat(A);
	std::cout << "Determinant: " << d << "\n";
	std::cout << "A:\n";	
	A.print();
	Mat U(3, 3);
    U(0, 0) = 1.0f; U(0, 1) = 0.0f; U(0, 2) = 0.0f;
    U(1, 0) = 0.0f; U(1, 1) = 1.0f; U(1, 2) = 0.0f;
    U(2, 0) = 0.0f; U(2, 1) = 0.0f; U(2, 2) = 1.0f;

    auto U_inv = inverse_mat(U);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            CHECK(std::abs(U_inv(i, j) - (i == j ? 1.0f : 0.0f)) < 1e-3f);
	std::cout << "U:\n";
	U.print();
	std::cout << "U_inv:\n";
	U_inv.print();

	Mat U2(3, 3);
    U2(0, 0) = 2.0f; U2(0, 1) = 0.0f; U2(0, 2) = 0.0f;
    U2(1, 0) = 0.0f; U2(1, 1) = 2.0f; U2(1, 2) = 0.0f;
    U2(2, 0) = 0.0f; U2(2, 1) = 0.0f; U2(2, 2) = 2.0f;
	auto U2_inv = inverse_mat(U2);	
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
			CHECK(std::abs(U2_inv(i, j) - (i == j ? 0.5f : 0.0f)) < 1e-3f);
	std::cout << "U2:\n";
	U2.print();
	std::cout << "U2_inv:\n";
	U2_inv.print();
	
	Mat U3(3, 3);
	U3(0, 0) = 8.0f; U3(0, 1) = 5.0f; U3(0, 2) = -2.0f;
	U3(1, 0) = 4.0f; U3(1, 1) = 7.0f; U3(1, 2) = 20.0f;
	U3(2, 0) = 7.0f; U3(2, 1) = 6.0f; U3(2, 2) = 1.0f;
	auto U3_inv = inverse_mat(U3);
	std::cout << "U3:\n";
	U3.print();
	std::cout << "U3_inv:\n";
	U3_inv.print();
    CHECK(std::abs(d - (-2.0f)) < 1e-3f);
	matrix_bench();
	std::cout << "\n✅ All Matrix tests passed.\n";
	return 0;
}


