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
    constexpr std::size_t N = 16384;
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
	matrix_bench();
	std::cout << "\n✅ All Matrix tests passed.\n";
	return 0;
}


