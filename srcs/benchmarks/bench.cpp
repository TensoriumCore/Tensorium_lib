#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h>
#include "../../includes/Morpheus/Morpheus.hpp"
#include <fstream>

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
    return 0;
}

