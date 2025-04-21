#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h>
#include "../../includes/Morpheus/Morpheus.hpp"
#include "../../includes/Morpheus/Core/LinearSolver.hpp"
#include "Morpheus/Functionnal/Functional.hpp"
#include <lapacke.h>
#include <fstream>
#include <stdexcept>


morpheus::Matrix<float> generate_diagonally_dominant_matrix(size_t n, float dominance_factor = 1.1f) {
	morpheus::Matrix<float> A(n, n);
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = i + 1; j < n; ++j) {
			float value = static_cast<float>(std::rand()) / RAND_MAX / 2.0f - 1.0f;
			value *= 10.0f;
			A(i, j) = value;
			A(j, i) = value;
		}
	}
	for (size_t i = 0; i < n; ++i) {
		float row_sum = 0.0f;
		for (size_t j = 0; j < n; ++j) {
			if (i != j) row_sum += std::abs(A(i, j));
		}
		A(i, i) = row_sum * dominance_factor;
	}
	return A;
}



void benchmark_solver(size_t n) {
    using namespace morpheus;

    Matrix<float> A = generate_diagonally_dominant_matrix(n);
    Vector<float> x_ref(n, 1.0f);                 
    Vector<float> b = morpheus::mul_vec(A, x_ref);  

    auto start = std::chrono::high_resolution_clock::now();
    Vector<float> x_jacobi = morpheus::jacobi_solve(A, b); 
    auto end = std::chrono::high_resolution_clock::now();
    float jacobi_time = std::chrono::duration<float>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    Vector<float> x_gauss = morpheus::gauss_solve(A, b); 
    end = std::chrono::high_resolution_clock::now();
    float gauss_time = std::chrono::duration<float>(end - start).count();



    Vector<float> bj = morpheus::mul_vec(A, x_jacobi); 
    Vector<float> bg = morpheus::mul_vec(A, x_gauss);


	Vector<float> diff_j = bj - b;
	Vector<float> diff_g = bg - b;
	float err_j = diff_j.norm_2();
	float err_g = diff_g.norm_2();

	std::cout << "n = " << n << " | Jacobi: " << jacobi_time << "s, err = " << err_j
		<< " | Gauss: " << gauss_time << "s, err = " << err_g << "\n";

}

template<typename K>
void benchmark_blas_vs_custom(size_t N, std::ofstream& csv) {
    using Clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    morpheus::Matrix<K> A(N, N);
    morpheus::Matrix<K> B(N, N);
    morpheus::Matrix<K> C_custom(N, N);
    morpheus::Matrix<K> C_blas(N, N);

    for (size_t i = 0; i < A.size(); ++i) A.data[i] = static_cast<K>(1.0);
    for (size_t i = 0; i < B.size(); ++i) B.data[i] = static_cast<K>(1.0);

    auto start_custom = Clock::now();
    C_custom = morpheus::mul_mat(A, B);
    auto end_custom = Clock::now();

    auto start_blas = Clock::now();
    if constexpr (std::is_same<K, float>::value) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0f,
                    A.data.data(), N,
                    B.data.data(), N,
                    0.0f,
                    C_blas.data.data(), N);
    } else if constexpr (std::is_same<K, double>::value) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0,
                    A.data.data(), N,
                    B.data.data(), N,
                    0.0,
                    C_blas.data.data(), N);
    }
    auto end_blas = Clock::now();

    double time_custom = duration(end_custom - start_custom).count();
    double time_blas   = duration(end_blas - start_blas).count();

    double gflops = 2.0 * N * N * N / 1e9;
    double perf_custom = gflops / time_custom;
    double perf_blas   = gflops / time_blas;

    K max_error = K(0);
    for (size_t i = 0; i < C_custom.size(); ++i)
        max_error = std::max(max_error, std::abs(C_custom.data[i] - C_blas.data[i]));

    csv << (std::is_same<K, float>::value ? "float" : "double") << ","
        << N << ","
        << time_custom << ","
        << time_blas << ","
        << perf_custom << ","
        << perf_blas << ","
        << max_error << "\n";

    std::cout << "N = " << N << " (" << (std::is_same<K, float>::value ? "float" : "double") << ") done.\n";
}


int main() {
    std::vector<size_t> sizes = {512, 1024, 2048, 4096, 8192};
    
    std::ofstream csv("benchmark_results.csv");
    csv << "Type,N,Time_Custom,Time_BLAS,GFLOPS_Custom,GFLOPS_BLAS,MaxAbsError\n";
    
    for (size_t N : sizes) {
        benchmark_blas_vs_custom<float>(N, csv);
        benchmark_blas_vs_custom<double>(N, csv);
    }
    
    csv.close();
    std::cout << "Benchmark results saved to benchmark_results.csv\n";
    
	std::srand(42);
	    for (size_t n : {32, 64, 128, 256, 512, 1024, 2048}) {
	        benchmark_solver(n);
	    }
    return 0;
}

