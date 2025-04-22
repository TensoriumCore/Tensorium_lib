
#include "../includes/Morpheus/Morpheus.hpp"
#include <random>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>
using namespace morpheus;


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


int test_tensor_mul() {
	bool success = true;

	Tensor<float, 2> A({2, 2});
	A({0,0}) = 1.0f;  A({0,1}) = 2.0f;
	A({1,0}) = 3.0f;  A({1,1}) = 4.0f;

	Tensor<float, 2> B({2, 2});
	B({0,0}) = 10.0f; B({0,1}) = 20.0f;
	B({1,0}) = 30.0f; B({1,1}) = 40.0f;

	auto C = morpheus::mul_tensor(A, B);
	C.print_shape();

	struct {
		std::array<size_t, 4> idx;
		float expected;
	} tests[] = {
		{{0,0,0,0}, 1.0f * 10.0f},
		{{0,1,1,0}, 2.0f * 30.0f},
		{{1,0,0,1}, 3.0f * 20.0f},
		{{1,1,1,1}, 4.0f * 40.0f}
	};

	for (const auto& t : tests) {
		float val = C(t.idx);
		if (std::abs(val - t.expected) > 1e-5f) {
			std::cerr << "[×] Error at ";
			for (size_t i : t.idx) std::cerr << i << " ";
			std::cerr << ": expected " << t.expected << ", got " << val << '\n';
			success = false;
		}
	}

	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 2; ++j)
			for (size_t k = 0; k < 2; ++k)
				for (size_t l = 0; l < 2; ++l) {
					float expected = A({i,j}) * B({k,l});
					float actual = C({i,j,k,l});
					if (std::abs(expected - actual) > 1e-5f) {
						std::cerr << "[×] Mismatch at (" << i << "," << j << "," << k << "," << l << "): "
							<< "expected " << expected << ", got " << actual << '\n';
						success = false;
					}
				}

	if (success)
		std::cout << "[✓] Tensor product test passed successfully.\n";

	return success ? 0 : 1;
}

int benchmark_tensor_mul() {
	constexpr size_t N = 128;

	Tensor<float, 2> A({N, N});
	Tensor<float, 2> B({N, N});
	A.fill(1.0f);
	B.fill(2.0f);

	std::cout << "Benchmarking morpheus::mul_tensor() with size: " << N << " x " << N << " ...\n";

	auto start = std::chrono::high_resolution_clock::now();
	auto C = mul_tensor(A, B);  
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end - start;
	std::cout << "[✓] Execution time: " << elapsed.count() << " s\n";

	float err = 0;
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			for (size_t k = 0; k < N; ++k)
				for (size_t l = 0; l < N; ++l) {
					err += std::abs(C({i,j,k,l}) - 2.0f);
				}
	std::cout << "Total absolute error: " << err << "\n";
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



template<typename K, size_t Rank>
void scalar_fallback_derivative_nd(const DerivateND<K, Rank>& input, DerivateND<K, Rank>& output, size_t axis, K dx) {
	const auto& shape = input.shape;
	const size_t total = input.size();
	const K inv_2dx = K(1) / (K(2) * dx);

	std::array<size_t, Rank> strides;
	strides[Rank - 1] = 1;
	for (int i = Rank - 2; i >= 0; --i)
		strides[i] = strides[i + 1] * shape[i + 1];

	const size_t stride_axis = strides[axis];
	const size_t dim_axis = shape[axis];

	for (size_t flat = 0; flat < total; ++flat) {
		const size_t coord_axis = (flat / stride_axis) % dim_axis;

		if (coord_axis == 0 || coord_axis >= dim_axis - 1) {
			output.data[flat] = K(0);
			continue;
		}

		output.data[flat] = (input.data[flat + stride_axis] - input.data[flat - stride_axis]) * inv_2dx;
	}
}



int main() {
	dispatch_simd([](auto simd) {
			using T = decltype(simd);
			constexpr size_t W = T::width;
			constexpr size_t A = T::alignment;
			std::cout << "SIMD selected: width=" << W << ", alignment=" << A << "\n";
			});
	test_tensor_mul();
	benchmark_tensor_mul();
	//
	// std::cout << "\n=== Vector Tests ===\n";
	// Vector<float> v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	// Vector<float> v2 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
	//
	// std::cout << "\n[v1] + [v2]:\n";
	// morpheus::add_vec(v1, v2).print();
	//
	// std::cout << "\n[v1] - [v2]:\n";
	// morpheus::sub_vec(v1, v2).print();
	//
	// std::cout << "\n[v1] * 0.5:\n";
	// morpheus::scl_vec(v1, 0.5f).print();
	//
	// std::cout << "\n=== Matrix Tests ===\n";
	// Matrix<float> m1(2, 8); 
	// Matrix<float> m2(2, 8);
	//
	// for (size_t i = 0; i < m1.rows; ++i)
	// 	for (size_t j = 0; j < m1.cols; ++j) {
	// 		m1(i, j) = i * 10 + j;
	// 		m2(i, j) = 1.0f;
	// 	}
	//
	// std::cout << "\n[m1] + [m2]:\n";
	// morpheus::add_mat(m1, m2).print();
	//
	// std::cout << "\n[m1] - [m2]:\n";
	// morpheus::sub_mat(m1, m2).print();
	//
	// std::cout << "\n[m1] * 2.0:\n";
	// morpheus::scl_mat(m1, 2.0f).print();
	//
	// std::cout << "\n=== Matrix and Vector Tests ===\n";
	// comb();
	//
	// Vector<float> a = {1, 2, 3, 4};
	// Vector<float> b = {5, 6, 7, 8};
	// float result = morpheus::dot_vec(a, b);
	// std::cout << "Dot product: " << result << "\n";
	//
	// Vector<float> v = {-3, 4, -5};
	// std::cout << "Norm 1  = " << morpheus::norm1_vec(v) << "\n"; 
	// std::cout << "Norm 2  = " << morpheus::norm2_vec(v) << "\n";
	// std::cout << "Norm ∞  = " << morpheus::normInf_vec(v) << "\n";
	//
	// std::cout << "cos(angle) = " << morpheus::cosine_vec(a, b) << "\n";
	//
	// Vector<float> c = {1, 1};
	// Vector<float> d = {2, 2};
	// std::cout << "cos(angle) = " << morpheus::cosine_vec(c, d) << "\n";
	//
	// std::cout << "\n=== Test cross_product() ===\n";
	// Vector<float> u3 = {1, 0, 0};
	// Vector<float> v3 = {0, 1, 0};
	// std::cout << "u = [1, 0, 0]\n";
	// std::cout << "v = [0, 1, 0]\n";
	// std::cout << "u × v = ";
	//
	// Matrix<float> A(3, 3);
	// Matrix<float> B(3, 3);
	//
	// A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
	// A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
	// A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
	//
	// B(0, 0) = 1.0; B(0, 1) = 0.0; B(0, 2) = 0.0;
	// B(1, 0) = 0.0; B(1, 1) = 1.0; B(1, 2) = 0.0;
	// B(2, 0) = 0.0; B(2, 1) = 0.0; B(2, 2) = 1.0;
	//
	// Matrix<float> C = morpheus::mul_mat(A, B);
	// std::cout << "\n=== Matrix Multiplication (Analytic Test) ===\n";
	//
	// Matrix<float> M1(2, 2);
	// Matrix<float> M2(2, 2);
	//
	// M1(0, 0) = 1.0f; M1(0, 1) = 2.0f;
	// M1(1, 0) = 3.0f; M1(1, 1) = 4.0f;
	//
	// M2(0, 0) = 5.0f; M2(0, 1) = 6.0f;
	// M2(1, 0) = 7.0f; M2(1, 1) = 8.0f;
	//
	// Matrix<float> M3 = morpheus::mul_mat(M1, M2);
	//
	// std::cout << "Expected:\n[19 22]\n[43 50]\n";
	// std::cout << "Result:\n";
	// M3.print();
	//
	// std::cout << "Matrix C (A * B):\n";
	// C.print();
	// test_transpose_matrix();
	//
	// Matrix<float> A_solve(2, 2);
	// A_solve(0, 0) = 2.0; A_solve(0, 1) = 1.0;
	// A_solve(1, 0) = 5.0; A_solve(1, 1) = 7.0;
	// Vector<float> b_solve = { 11.0, 13.0 };
	//
	// Vector<float> x_solve = morpheus::gauss_solve(A_solve, b_solve);
	// std::cout << "Solution x:\n";
	// x_solve.print();
	//
	// Vector<float> b_check = A_solve * x_solve;
	// std::cout << "Check Ax = b:\n";
	// b_check.print();
	//
	//
	// Matrix<float> A2_solve(3, 3);
	// A2_solve(0, 0) = 10.0f; A2_solve(0, 1) = -1.0f; A2_solve(0, 2) = 2.0f;
	// A2_solve(1, 0) = -1.0f; A2_solve(1, 1) = 11.0f; A2_solve(1, 2) = -1.0f;
	// A2_solve(2, 0) = 2.0f;  A2_solve(2, 1) = -1.0f; A2_solve(2, 2) = 10.0f;
	//
	// Vector<float> b2_solve = { 6.0f, 25.0f, -11.0f };
	//
	// Vector<float> x2_solve = morpheus::jacobi_solve(A2_solve, b2_solve);
	// std::cout << "Solution x2:\n";	
	// x2_solve.print();
	// Vector<float> b2_check = morpheus::mul_vec(A2_solve, x2_solve);
	// std::cout << "Check Ax2 = b2:\n";
	// b2_check.print();

	/* std::cout << "\n=== Benchmarking ===\n"; */
	bench();
	//
	// test_flatten_index();
	// test_contract();
	// Matrix<float> expected(2, 2);
	// expected(0, 0) = 19.0f; expected(0, 1) = 22.0f;
	// expected(1, 0) = 43.0f; expected(1, 1) = 50.0f;
	//
	// bool ok = true;
	// for (size_t i = 0; i < 2; ++i)
	// 	for (size_t j = 0; j < 2; ++j)
	// 		if (std::fabs(M3(i, j) - expected(i, j)) > 1e-4f) {
	// 			ok = false;
	// 			std::cout << "Mismatch at (" << i << ", " << j << "): got " << M3(i, j) << ", expected " << expected(i, j) << "\n";
	// 		}
	//
	// if (ok)
	// 	std::cout << "[PASS] Matrix multiplication result is correct.\n";
	// else
	// 	std::cerr << "[FAIL] Matrix multiplication mismatch.\n";
	// std::cout << "\n=== Derivate 2D Test (∂/∂x) ===\n";
	// morpheus::Derivate<float> f2d(4, 4);
	// morpheus::Derivate<float> dfdx2d(4, 4);
	//
	// for (size_t i = 0; i < 4; ++i)
	// 	for (size_t j = 0; j < 4; ++j)
	// 		f2d(i, j) = static_cast<float>(i * 10 + j);
	//
	// morpheus::centered_derivative(f2d, dfdx2d, 0, 1.0f);
	//
	// std::cout << "∂f/∂x:\n";
	// for (size_t i = 0; i < 4; ++i) {
	// 	for (size_t j = 0; j < 4; ++j)
	// 		std::cout << dfdx2d(i, j) << " ";
	// 	std::cout << "\n";
	// }
	//
	// std::cout << "\n=== Derivate 2D Test (∂/∂y) ===\n";
	// morpheus::Derivate<float> f2d_y(4, 4);
	// morpheus::Derivate<float> dfdx2d_y(4, 4);
	// for (size_t i = 0; i < 4; ++i)
	// 	for (size_t j = 0; j < 4; ++j)
	// 		f2d_y(i, j) = static_cast<float>(i * 10 + j);
	// morpheus::centered_derivative(f2d_y, dfdx2d_y, 1, 1.0f);
	// std::cout << "∂f/∂y:\n";
	// for (size_t i = 0; i < 4; ++i) {
	// 	for (size_t j = 0; j < 4; ++j)
	// 		std::cout << dfdx2d_y(i, j) << " ";
	// 	std::cout << "\n";
	// }
	//
	// std::cout << "\n=== DerivateND 3D Test (∂/∂x) ===\n";
	// std::array<size_t, 3> dims_x = {4, 4, 4};
	// morpheus::DerivateND<float, 3> fnd_x(dims_x);
	// morpheus::DerivateND<float, 3> dfdxnd(dims_x);
	// for (size_t i = 0; i < 4; ++i)
	// 	for (size_t j = 0; j < 4; ++j)
	// 		for (size_t k = 0; k < 4; ++k)
	// 			fnd_x({i, j, k}) = static_cast<float>(i + j + k);
	// morpheus::centered_derivative(fnd_x, dfdxnd, 0, 1.0f);
	// std::cout << "∂f/∂x slice at i=2:\n";
	// for (size_t j = 0; j < 4; ++j) {
	// 	for (size_t k = 0; k < 4; ++k)
	// 		std::cout << dfdxnd({2, j, k}) << " ";
	// 	std::cout << "\n";
	// }
	// std::cout << "\n=== DerivateND 3D Test (∂/∂z) ===\n";
	// std::array<size_t, 3> dims = {4, 4, 4};
	// morpheus::DerivateND<float, 3> fnd(dims);
	// morpheus::DerivateND<float, 3> dfdznd(dims);
	//
	// for (size_t i = 0; i < 4; ++i)
	// 	for (size_t j = 0; j < 4; ++j)
	// 		for (size_t k = 0; k < 4; ++k)
	// 			fnd({i, j, k}) = static_cast<float>(i + j + k);
	//
	// morpheus::centered_derivative(fnd, dfdznd, 2, 1.0f);
	// 	std::cout << "∂f/∂z slice at k=2:\n";
	// for (size_t i = 0; i < 4; ++i) {
	// 	for (size_t j = 0; j < 4; ++j)
	// 		std::cout << dfdznd({i, j, 2}) << " ";
	// 	std::cout << "\n";
	// }
	//
	// morpheus::centered_derivative_order4(fnd, dfdznd, 2, 1.0f);
	//
	// std::cout << "∂f/∂z slice at k=2:\n";
	// for (size_t i = 0; i < 4; ++i) {
	// 	for (size_t j = 0; j < 4; ++j)
	// 		std::cout << dfdznd({i, j, 2}) << " ";
	// 	std::cout << "\n";
	// }
	//
	// const size_t N = 16384;
	//     const float dx = 0.01f; 
	//     const float pi = 3.14159265358979323846f;
	//
	//     morpheus::Derivate<float> f(N, 1);
	//     morpheus::Derivate<float> df_order2(N, 1);
	//     morpheus::Derivate<float> df_order4(N, 1);
	//     morpheus::Derivate<float> df_exact(N, 1);
	//
	//     for (size_t i = 0; i < N; ++i) {
	//         float x = i * dx;
	//         f(i, 0) = std::sin(x) + 0.1f * std::sin(10*x);
	//         df_exact(i, 0) = std::cos(x) + 0.1f * 10 * std::cos(10*x);
	//     }
	//     f.centered_derivative(f, df_order2, 0, dx); 
	//     f.centered_derivative_order4(f, df_order4, 0, dx);
	//
	//     float max_err_order2 = 0.0f;
	//     float max_err_order4 = 0.0f;
	//     float avg_err_order2 = 0.0f;
	//     float avg_err_order4 = 0.0f;
	//     size_t valid_points = 0;
	//
	//     for (size_t i = 2; i < N - 2; ++i) {
	//         float exact = df_exact(i, 0);
	//         float err2 = std::abs(df_order2(i, 0) - exact);
	//         float err4 = std::abs(df_order4(i, 0) - exact);
	//
	//         max_err_order2 = std::max(max_err_order2, err2);
	//         max_err_order4 = std::max(max_err_order4, err4);
	//         avg_err_order2 += err2;
	//         avg_err_order4 += err4;
	//         valid_points++;
	//     }
	//     avg_err_order2 /= valid_points;
	//     avg_err_order4 /= valid_points;
	// std::cout << "DEBUG: Formula coefficients: "
	// 	<< "1.0/" << (12*dx) << " * [-1, 8, -8, 1]" << std::endl;
	//
	// for (size_t i = 2; i < 5; ++i) {
	// 	std::cout << "x=" << i*dx << " f=" << f(i,0) 
	// 		<< " df4=" << df_order4(i,0)
	// 		<< " exact=" << df_exact(i,0) << std::endl;
	// }
	// std::cout << "=== Test dérivée d'ordre 4 ===" << std::endl;
	// std::cout << "Points valides: " << valid_points << "/" << N << std::endl;
	// std::cout << "Erreur max (ordre 2): " << max_err_order2 << std::endl;
	// std::cout << "Erreur max (ordre 4): " << max_err_order4 << std::endl;
	// std::cout << "Erreur moy (ordre 2): " << avg_err_order2 << std::endl;
	//     std::cout << "Erreur moy (ordre 4): " << avg_err_order4 << std::endl;
	//
	//     std::cout << "\nVérification des conditions aux bords:" << std::endl;
	//     std::cout << "df_order4(0,0): " << df_order4(0, 0) << " (devrait être 0)" << std::endl;
	//     std::cout << "df_order4(1,0): " << df_order4(1, 0) << " (devrait être 0)" << std::endl;
	//     std::cout << "df_order4(N-2,0): " << df_order4(N-2, 0) << " (devrait être 0)" << std::endl;
	//     std::cout << "df_order4(N-1,0): " << df_order4(N-1, 0) << " (devrait être 0)" << std::endl;
	// return 0;
}
