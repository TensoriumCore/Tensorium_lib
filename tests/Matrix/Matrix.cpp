#include "../test.hpp"
using namespace morpheus;
#include <complex>

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


    Mat Id = A.mul_mat(A_inv);
    CHECK(std::abs(Id(0, 0) - 1.0f) < 1e-3f);
    CHECK(std::abs(Id(1, 1) - 1.0f) < 1e-3f);
    CHECK(std::abs(Id(0, 1)) < 1e-3f);
    CHECK(std::abs(Id(1, 0)) < 1e-3f);
	std::cout << "Determinant test\n";
	A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;

    auto d = det_mat(A);
	Mat U(3, 3);
    U(0, 0) = 1.0f; U(0, 1) = 0.0f; U(0, 2) = 0.0f;
    U(1, 0) = 0.0f; U(1, 1) = 1.0f; U(1, 2) = 0.0f;
    U(2, 0) = 0.0f; U(2, 1) = 0.0f; U(2, 2) = 1.0f;

    auto U_inv = inverse_mat(U);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            CHECK(std::abs(U_inv(i, j) - (i == j ? 1.0f : 0.0f)) < 1e-3f);

	Mat U2(3, 3);
    U2(0, 0) = 2.0f; U2(0, 1) = 0.0f; U2(0, 2) = 0.0f;
    U2(1, 0) = 0.0f; U2(1, 1) = 2.0f; U2(1, 2) = 0.0f;
    U2(2, 0) = 0.0f; U2(2, 1) = 0.0f; U2(2, 2) = 2.0f;
	auto U2_inv = inverse_mat(U2);	
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
			CHECK(std::abs(U2_inv(i, j) - (i == j ? 0.5f : 0.0f)) < 1e-3f);
	
	Mat U3(3, 3);
	U3(0, 0) = 8.0f; U3(0, 1) = 5.0f; U3(0, 2) = -2.0f;
	U3(1, 0) = 4.0f; U3(1, 1) = 7.0f; U3(1, 2) = 20.0f;
	U3(2, 0) = 7.0f; U3(2, 1) = 6.0f; U3(2, 2) = 1.0f;
	auto U3_inv = inverse_mat(U3);
    CHECK(std::abs(d - (-2.0f)) < 1e-3f);

	Matrix<float> G(3, 4);
	G(0, 0) = 1; G(0, 1) = 2; G(0, 2) = 3; G(0, 3) = 4;
	G(1, 0) = 2; G(1, 1) = 4; G(1, 2) = 6; G(1, 3) = 8;
	G(2, 0) = 0; G(2, 1) = 0; G(2, 2) = 0; G(2, 3) = 0;

	size_t r = G.rank();
	std::cout << "Rank of A = " << r << "\n";
	CHECK(r == 1);

	Matrix<std::complex<float>> Ac(2, 2);
	Ac(0, 0) = {1.0f, 2.0f}; Ac(0, 1) = {3.0f, 4.0f};
	Ac(1, 0) = {5.0f, 6.0f}; Ac(1, 1) = {7.0f, 8.0f};

	Matrix<std::complex<float>> Bc(2, 2);
	Bc(0, 0) = {8.0f, 7.0f}; Bc(0, 1) = {6.0f, 5.0f};
	Bc(1, 0) = {4.0f, 3.0f}; Bc(1, 1) = {2.0f, 1.0f};

	Matrix<std::complex<float>> Cc = Ac;
	Cc.add(Bc);
	CHECK(std::abs(Cc(0, 0).real() - 9.0f) < 1e-4);
	CHECK(std::abs(Cc(0, 0).imag() - 9.0f) < 1e-4);
	CHECK(std::abs(Cc(1, 1).real() - 9.0f) < 1e-4);
	CHECK(std::abs(Cc(1, 1).imag() - 9.0f) < 1e-4);
	Cc.sub(Bc);
	Cc.scl(2.0f);
	Cc = morpheus::mul_mat(Ac, Bc);
	std::cout << "✅ add_mat on complex<float> passed.\n";
	matrix_bench();
	std::cout << "\n✅ All Matrix tests passed.\n";
	constexpr size_t dim = 4;

	morpheus::Vector<double> X(dim);
	X(0) = 0.0; 
	X(1) = 10.0;
	X(2) = M_PI / 2.0;
	X(3) = 0.0;

	morpheus::Tensor<double, 2> g({dim, dim});
	morpheus::Tensor<double, 2> g_inv({dim, dim});

	std::cout << "X = " << X(0) << " " << X(1) << " " << X(2) << " " << X(3) << "\n";

	morpheus_RG::Metric<double> metric("kerr", 1.0, 0.8);
	metric(X, g);

	std::cout << "Metric tensor g at X = (t=0, r=10, θ=π/2, φ=0):\n";
	g_inv = morpheus::inv_mat_tensor(g); 
	g.print_shape();
	g.print();
	std::cout << "Christoffel symbols Γ^λ_{μν} at X = (t=0, r=10, θ=π/2, φ=0):\n";
	auto gamma = morpheus::compute_christoffel(X, 1e-5, g, g_inv, metric);
	gamma.print();
	auto R = morpheus::compute_riemann_tensor<double>(X, 1e-5, morpheus_RG::Metric<double>("kerr", 1.0, 0.8));
	morpheus::print_riemann_tensor(R);
	 morpheus::contract_tensor<0, 1>(R);
	std::cout << "Riemann tensor contracted:\n";
	R.print_shape();
	std::cout << "Riemann tensor contracted to Ricci tensor:\n";
	R.print();

     constexpr std::size_t N = 1024;
    constexpr double L = 1.0; 
    constexpr double dx = L / N;
    constexpr double two_pi = 2.0 * M_PI;

    morpheus::Vector<std::complex<double>> f(N); 
    morpheus::Vector<std::complex<double>> f_hat(N);
    morpheus::Vector<std::complex<double>> df(N); 

    for (std::size_t i = 0; i < N; ++i) {
        double x = i * dx;
        f[i] = std::sin(two_pi * x);
    }

    f_hat = f;
    morpheus::forwardFFT(f_hat);

    for (std::size_t k = 0; k < N; ++k) {
        std::ptrdiff_t k_signed = (k <= N/2) ? k : k - N; 
        std::complex<double> ik = std::complex<double>(0.0, two_pi * k_signed / L);
        f_hat[k] *= ik;
    }

    df = f_hat;
    morpheus::backwardFFT(df);  

    double max_err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double x = i * dx;
        double exact = two_pi * std::cos(two_pi * x);
        double err = std::abs(df[i].real() - exact);
        if (err > max_err) max_err = err;
    }

    std::cout << "=== Dérivée spectrale de sin(2πx) ===\n";
    std::cout << "Erreur max sur f'(x) = " << max_err << "\n";

    return (max_err < 1e-12) ? 0 : 1;
}


