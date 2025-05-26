#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "includes/Tensorium/Tensorium.hpp"
#include "includes/Tensorium/Core/GemmKernel.hpp"

using namespace tensorium;
using namespace std;
using namespace chrono;

template<typename T>
void print_matrix(const T* mat, int rows, int cols, const string& name) {
    cout << "\nMatrix " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < min(10, rows); ++i) {
        for (int j = 0; j < min(10, cols); ++j) {
            cout << setw(10) << setprecision(4) << mat[i * cols + j] << " ";
        }
        if (cols > 10) cout << "...";
        cout << endl;
    }
    if (rows > 10) cout << "...\n";
}

template<typename T>
void random_matrix(T* mat, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<T> dist(-1.0, 1.0);
    
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * cols + j] = dist(gen);
        }
    }
}

void gemm_ref(float* A, float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[j * K + k];
            }
            C[j * M + i] = sum;
        }
    }
}

void verify_results(float* C_ref, float* C_opt, int M, int N) {
    const float epsilon = 1e-3f;
    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_i = 0, max_diff_j = 0;

    #pragma omp parallel for reduction(+:error_count)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float ref = C_ref[i * N + j];
            const float opt = C_opt[i * N + j];
            const float diff = fabs(ref - opt);
            const float rel_diff = diff / (fabs(ref) + 1e-9f);

            if (diff > epsilon && rel_diff > epsilon) {
                #pragma omp critical
                {
                    error_count++;
                    if (diff > max_diff) {
                        max_diff = diff;
                        max_diff_i = i;
                        max_diff_j = j;
                    }
                    if (error_count <= 5) {
                        cout << "DIFF at (" << i << "," << j << "): "
                             << "Ref=" << ref << " Opt=" << opt 
                             << " Diff=" << diff << " Rel=" << rel_diff << endl;
                    }
                }
            }
        }
    }

    if (error_count > 0) {
        cout << "\nVERIFICATION FAILED: " << error_count << " errors found\n";
        cout << "Max difference at (" << max_diff_i << "," << max_diff_j << "): "
             << "Ref=" << C_ref[max_diff_i * N + max_diff_j] 
             << " Opt=" << C_opt[max_diff_i * N + max_diff_j]
             << " Diff=" << max_diff << endl;
    } else {
        cout << "\nVERIFICATION PASSED - All values match within tolerance\n";
    }
}

int main() {
    const int M = 168;
    const int N = 168;
    const int K = 168;
    
    float *A, *B, *C_ref, *C_opt;
    posix_memalign((void**)&A, 64, M * K * sizeof(float));
    posix_memalign((void**)&B, 64, K * N * sizeof(float));
    posix_memalign((void**)&C_ref, 64, M * N * sizeof(float));
    posix_memalign((void**)&C_opt, 64, M * N * sizeof(float));

    for (int i = 0; i < M * N; ++i) C_opt[i] = 0.0f;
    cout << "Generating random matrices..." << endl;
    random_matrix(A, M, K);
    random_matrix(B, K, N);
    memset(C_ref, 0, M * N * sizeof(float));
    memset(C_opt, 0, M * N * sizeof(float));

    // Print sample of input matrices
    print_matrix(A, M, K, "A");
    print_matrix(B, K, N, "B");

    // Reference calculation
    cout << "\nRunning reference GEMM..." << endl;
    auto start = high_resolution_clock::now();
    gemm_ref(A, B, C_ref, M, N, K);
    auto duration_ref = duration_cast<microseconds>(high_resolution_clock::now() - start);
    print_matrix(C_ref, M, N, "C_ref");
    
    // Optimized calculation
    cout << "\nRunning Tensorium GEMM..." << endl;
    start = high_resolution_clock::now();
    GemmKernel<float> kernel;
    kernel.matmul(A, B, C_opt, M, N, K);
    auto duration_opt = duration_cast<microseconds>(high_resolution_clock::now() - start);
    print_matrix(C_opt, M, N, "C_opt");

    // Verification
    cout << "\nVerifying results..." << endl;
    verify_results(C_ref, C_opt, M, N);

    // Performance metrics
    double flops = 2.0 * M * N * K;
    double gflops_ref = flops / (duration_ref.count() * 1e3);
    double gflops_opt = flops / (duration_opt.count() * 1e3);

    cout << "\n=== Performance Results ===" << endl;
    cout << fixed << setprecision(2);
    cout << "Reference time: " << duration_ref.count() / 1000.0 << " ms (" 
         << gflops_ref << " GFLOPS)" << endl;
    cout << "Tensorium time: " << duration_opt.count() / 1000.0 << " ms (" 
         << gflops_opt << " GFLOPS)" << endl;
    cout << "Speedup: " << duration_ref.count() / (double)duration_opt.count() << "x" << endl;

    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_opt);

    return 0;
}
