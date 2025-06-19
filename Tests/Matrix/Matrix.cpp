#include "../test.hpp"
using namespace tensorium;
#include <complex>

#define CHECK(expr) \
	do { \
		if (!(expr)) { \
			std::cerr << "\u274c CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(1); \
		} \
	} while (0)


template<typename K>
tensorium::Matrix<K> mul_mat_reference(const tensorium::Matrix<K>& A, const tensorium::Matrix<K>& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("Matrix dimensions do not match for reference multiplication.");

    tensorium::Matrix<K> C(A.rows, B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            K sum = K(0);
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}
int matrix_bench() {
	using namespace tensorium;
	std::vector<std::size_t> sizes = {256};

	for (std::size_t N : sizes) {
		Matrix<double> A(N, N);
		Matrix<double> B(N, N);

#pragma omp parallel
		{
			std::mt19937 rng(42 + omp_get_thread_num());
			std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
			for (std::size_t i = 0; i < N; ++i)
				for (std::size_t j = 0; j < N; ++j) {
					A(i, j) = dist(rng);
					B(i, j) = dist(rng);
				}
		}

		std::cout << "\n=== Benchmarking N = " << N << " ===\n";

		const auto t0 = std::chrono::high_resolution_clock::now();
		auto C = mul_mat(A, B);
		const auto t1 = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double>(t1 - t0).count();

		const long double flops = 2.0L * N * N * N;
		const double gflops = static_cast<double>(flops / 1e9L) / elapsed;

		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Time      : " << elapsed << "  s\n";
		std::cout << "GFLOP/s   : " << gflops  << '\n';
		std::cout << "Sample C(0,0): " << C(0, 0) << '\n';

	}

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
	std::cout << "A = " <<std::endl;	
	A.print();

	std::cout << "B = " <<std::endl;	
	B.print();
	Mat M = tensorium::mul_mat(A, B);

	std::cout << "A x B = " <<std::endl;	
	M.print();
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


    Mat Id = tensorium::mul_mat(A, A_inv);
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
	Cc = tensorium::mul_mat(Ac, Bc);
	std::cout << "✅ add_mat on complex<float> passed.\n";
	matrix_bench();

	Mat A2(2, 2);
	A2(0, 0) = 1; A2(0, 1) = 2;
	A2(1, 0) = 3; A2(1, 1) = 4;

	Mat B2(2, 2);
	B2(0, 0) = 5; B2(0, 1) = 6;
	B2(1, 0) = 7; B2(1, 1) = 8;

	Mat R2 = tensorium::mul_mat(A2, B2);
	CHECK(std::abs(R2(0, 0) - (1*5 + 2*7)) < 1e-4f);
	CHECK(std::abs(R2(0, 1) - (1*6 + 2*8)) < 1e-4f);
	CHECK(std::abs(R2(1, 0) - (3*5 + 4*7)) < 1e-4f);
	CHECK(std::abs(R2(1, 1) - (3*6 + 4*8)) < 1e-4f);

	std::cout << "✅ 2x2 matrix multiplication passed.\n";
	Mat A3(3, 3);

	A3(0, 0) = 1; A3(0, 1) = 2; A3(0, 2) = 3;
	A3(1, 0) = 4; A3(1, 1) = 5; A3(1, 2) = 6;
	A3(2, 0) = 7; A3(2, 1) = 8; A3(2, 2) = 9;

	Mat B3(3, 3);
	B3(0, 0) = 9; B3(0, 1) = 8; B3(0, 2) = 7;
	B3(1, 0) = 6; B3(1, 1) = 5; B3(1, 2) = 4;
	B3(2, 0) = 3; B3(2, 1) = 2; B3(2, 2) = 1;

	Mat R3 = tensorium::mul_mat(A3, B3);

	CHECK(std::abs(R3(0, 0) - (1*9 + 2*6 + 3*3)) < 1e-4f);
	CHECK(std::abs(R3(0, 1) - (1*8 + 2*5 + 3*2)) < 1e-4f);
	CHECK(std::abs(R3(0, 2) - (1*7 + 2*4 + 3*1)) < 1e-4f);
	CHECK(std::abs(R3(1, 0) - (4*9 + 5*6 + 6*3)) < 1e-4f);
	CHECK(std::abs(R3(2, 2) - (7*7 + 8*4 + 9*1)) < 1e-4f);

	std::cout << "✅ 3x3 matrix multiplication passed.\n";
	Mat A4(4, 4);
	Mat B4(4, 4);
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			A4(i, j) = i * 4 + j + 1;
			B4(i, j) = 16 - (i * 4 + j);
		}
	
	Mat R4 = tensorium::mul_mat(A4, B4);
	CHECK(std::abs(R4(0, 0) - (1*16 + 2*12 + 3*8 + 4*4)) < 1e-4f);
	CHECK(std::abs(R4(3, 3) - (13*13 + 14*9 + 15*5 + 16*1)) < 1e-4f);

	std::cout << "✅ 4x4 matrix multiplication passed.\n";


	Mat A8(8, 8);
	Mat B8(8, 8);
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j) {
			A8(i, j) = static_cast<float>(i * 8 + j + 1);
			B8(i, j) = static_cast<float>((i + j) % 8 + 1);
		}

	Mat R8 = tensorium::mul_mat(A8, B8);
	float expected = 0.0f;
	for (int k = 0; k < 8; ++k)
		expected += A8(0, k) * B8(k, 0);

	CHECK(std::abs(R8(0, 0) - expected) < 1e-4f);
	std::cout << "✅ 8x8 matrix multiplication passed.\n";


	Mat A16(16, 16), B16(16, 16);
	for (int i = 0; i < 16; ++i)
		for (int j = 0; j < 16; ++j)
			A16(i, j) = i + j, B16(i, j) = 16 - i + j;

	Mat R16 = tensorium::mul_mat(A16, B16);
	CHECK(std::abs(R16(0, 0) - 680.0f) < 1e-3f);
	std::cout << "✅ 16x16 matrix multiplication passed.\n";

	Mat A32(32, 32), B32(32, 32);
	for (int i = 0; i < 32; ++i)
		for (int j = 0; j < 32; ++j)
			A32(i, j) = i + j, B32(i, j) = 32 - i + j;

	Mat R32 = tensorium::mul_mat(A32, B32);
	CHECK(std::abs(R32(0, 0) - 5456.0f) < 1e-3f);
	std::cout << "✅ 32x32 matrix multiplication passed.\n";


	std::cout << "\n✅ All Matrix tests passed.\n";
	constexpr size_t dim = 4;

	tensorium::Vector<double> X(dim);
	X(0) = 0.0; 
	X(1) = 3.0;
	X(2) = M_PI / 2.0;
	X(3) = 0.0;

	tensorium::Tensor<double, 2> g({dim, dim});
	tensorium::Tensor<double, 2> g_inv({dim, dim});

	std::cout << "X = " << X(0) << " " << X(1) << " " << X(2) << " " << X(3) << "\n";

	tensorium_RG::Metric<double> metric("kerr_schild", 1.0, 0.935);
	metric(X, g);

	std::cout << "Metric tensor g at X = (t=0, r=10, θ=π/2, φ=0):\n";
	g_inv = tensorium::inv_mat_tensor(g); 
	g.print_shape();
	g.print();
	std::cout << "Christoffel symbols Γ^λ_{μν} at X = (t=0, r=10, θ=π/2, φ=0):\n";
	auto gamma = tensorium::compute_christoffel(X, 1e-5, g, g_inv, metric);
	gamma.print();
	auto R = tensorium::compute_riemann_tensor<double>(X, 1e-5, tensorium_RG::Metric<double>("kerr_schild", 1.0, 0.935));
	tensorium::print_riemann_tensor(R);
	tensorium::contract_tensor<0, 1>(R);
	std::cout << "Riemann tensor contracted:\n";
	std::cout << "Riemann tensor contracted to Ricci tensor:\n";
	double dx = 1e-4, dy = 1e-4, dz = 1e-4;

	double alpha;
	tensorium::Vector<double> beta(3);
	tensorium::Tensor<double, 2> gammaj({3, 3});
	metric.BSSN(X, alpha, beta, gammaj);
 
	tensorium::Tensor<double, 2> gammaj_inv = tensorium::inv_mat_tensor(gammaj);

	std::cout << "\n--- BSSN 3+1 Decomposition ---\n";
	std::cout << "Lapse α = " << alpha << "\n";

	std::cout << "Shift vector β^i = [";
	for (size_t i = 0; i < beta.size(); ++i)
		std::cout << beta(i) << (i + 1 < beta.size() ? ", " : "");
	std::cout << "]\n";

	std::cout << "Spatial metric γ_{ij}:\n";
	gammaj.print_shape();
	gammaj.print();

	std::cout << "Spatial metric γ^{ij}:\n";
	gammaj_inv.print();

	std::cout << "--- SETUP BSSN TEST---\n";	
	auto bssn = tensorium::setup_BSSN(X, metric, dx, dy, dz);

	return 0;
}


