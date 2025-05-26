#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h>
#include "../../includes/Tensorium/Tensorium.hpp"
#include "../../includes/Tensorium/Core/LinearSolver.hpp"
#include "Tensorium/Functionnal/Functional.hpp"
#include <fstream>
#include <random>
#include <algorithm>
#include <vector>

// Génère une matrice symétrique, strictement diagonale dominante
tensorium::Matrix<float> generate_diagonally_dominant_matrix(size_t n, float dominance_factor = 1.1f) {
    tensorium::Matrix<float> A(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float v = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 20.f;
            A(i, j) = A(j, i) = v;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.f;
        for (size_t j = 0; j < n; ++j) if (i != j)
            sum += std::abs(A(i, j));
        A(i, i) = sum * dominance_factor;
    }
    return A;
}

// Solvers benchmark omitted…

// Benchmark BLAS vs custom matmul, écrit dans le CSV et affiche à la console
template<typename K>
void benchmark_blas_vs_custom(size_t N, std::ofstream& csv) {
    using Clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    tensorium::Matrix<K> A(N, N), B(N, N), C_custom(N, N), C_blas(N, N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<K> dist(K(0), K(1));
    for (size_t i = 0; i < N*N; ++i) {
        A.data[i] = dist(rng);
        B.data[i] = dist(rng);
    }

    auto t0 = Clock::now();
    C_custom = tensorium::mul_mat(A, B);
    auto t1 = Clock::now();
    double time_custom = duration(t1 - t0).count();

    t0 = Clock::now();
    if constexpr(std::is_same<K,float>::value) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0f,
                    A.data.data(), N,
                    B.data.data(), N,
                    0.0f,
                    C_blas.data.data(), N);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0,
                    A.data.data(), N,
                    B.data.data(), N,
                    0.0,
                    C_blas.data.data(), N);
    }
    t1 = Clock::now();
    double time_blas = duration(t1 - t0).count();

    double gflops = 2.0 * N * N * N / 1e9;
    double perf_custom = gflops / time_custom;
    double perf_blas   = gflops / time_blas;

    K max_error = K(0);
    for (size_t i = 0; i < N*N; ++i)
        max_error = std::max(max_error, std::abs(C_custom.data[i] - C_blas.data[i]));

    // Écriture CSV
    csv << N << "," 
        << time_custom << "," 
        << time_blas   << "," 
        << perf_custom << "," 
        << perf_blas   << "," 
        << max_error   << "\n";

    // Affichage console
    std::cout << "N=" << N
              << " | Custom: " << time_custom << " s (" << perf_custom << " GFLOP/s)"
              << " | BLAS: "   << time_blas   << " s (" << perf_blas   << " GFLOP/s)"
              << " | MaxErr: " << max_error   << "\n";
}

// Benchmark sur une plage de tailles
template<typename K>
void benchmark_matmul_range(
    size_t startn,
    size_t maxn,
    size_t step,
    size_t loops,
    const std::string& out_csv = "benchmark_range.csv"
) {
    using Clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    std::ofstream csv(out_csv);
    csv << "Size,Loop,Time(s),GFLOP/s\n";

    for (size_t n = startn; n <= maxn; n += step) {
        std::vector<double> times, gflops_list;
        tensorium::Matrix<K> A(n,n), B(n,n);

        std::mt19937_64 rng(123);
        std::uniform_real_distribution<K> dist(K(0), K(1));
        for (size_t i = 0; i < n*n; ++i) {
            A.data[i] = dist(rng);
            B.data[i] = dist(rng);
        }

        for (size_t loop = 0; loop < loops; ++loop) {
            auto t0 = Clock::now();
            auto C = tensorium::mul_mat(A, B);
            auto t1 = Clock::now();

            double elapsed = duration(t1 - t0).count();
            double ops = 2.0 * n * n * n;
            double gflops = ops / (elapsed * 1e9);

            times.push_back(elapsed);
            gflops_list.push_back(gflops);

            csv << n << "," << loop << "," << elapsed << "," << gflops << "\n";
        }

        double avg  = std::accumulate(gflops_list.begin(), gflops_list.end(), 0.0) / loops;
        double peak = *std::max_element(gflops_list.begin(), gflops_list.end());
        double last = times.back();

        std::cout << "Range N=" << n
                  << " | Avg="  << avg  << " GFLOP/s"
                  << " | Peak=" << peak << " GFLOP/s"
                  << " | Last=" << last << " s\n";
    }
    std::cout << "Range benchmark saved to " << out_csv << "\n";
}

int main() {
    std::cout << "\n=== BLAS vs Custom Matrix×Matrix ===\n";
    std::ofstream csv1("blas_vs_custom.csv");
    csv1 << "N,Time_Custom,Time_BLAS,GFLOPS_Custom,GFLOPS_BLAS,MaxAbsError\n";
    for (size_t N : {512, 1024, 2048, 4096, 8192}) {
        benchmark_blas_vs_custom<float>(N, csv1);
    }
    csv1.close();

    std::cout << "\n=== Plage de tailles pour Custom Matrix×Matrix ===\n";
    benchmark_matmul_range<float>(250, 2000, 250, 5, "range_benchmark.csv");

    return 0;
}
