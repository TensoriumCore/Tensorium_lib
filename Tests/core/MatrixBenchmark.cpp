#include "../../includes/Tensorium/Tensorium.hpp"

#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace tensorium;

#ifdef TENSORIUM_USE_CBLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif
#ifdef TENSORIUM_USE_FFTW
#    include <fftw3.h>
#endif

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            std::cerr << "\u274c CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ \
                      << "\n";                                                                     \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

template <typename K>
tensorium::Matrix<K> mul_mat_reference(const tensorium::Matrix<K> &A,
                                       const tensorium::Matrix<K> &B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("Matrix dimensions do not match for reference multiplication.");

    tensorium::Matrix<K> C(A.rows, B.cols);

    for (size_t i = 0; i < A.rows; ++i)
        for (size_t j = 0; j < B.cols; ++j) {
            K sum = K(0);
            for (size_t k = 0; k < A.cols; ++k)
                sum += A(i, k) * B(k, j);
            C(i, j) = sum;
        }
    return C;
}

int run_matrix_benchmark() {
    std::vector<std::size_t> sizes = {1024, 2048, 4096};
    const std::string        csv_path = "matrix_bench_results.csv";

    std::ofstream csv(csv_path);

#ifdef TENSORIUM_USE_CBLAS
    csv << "N,Tensorium_GFLOPs,OpenBLAS_GFLOPs,Tensorium_Time(s),OpenBLAS_Time(s),Speedup\n";
#else
    csv << "N,Tensorium_GFLOPs,Tensorium_Time(s)\n";
#endif

    std::cout << "Benchmarking GEMM performance...\n";

    for (std::size_t N : sizes) {
        Matrix<float> A(N, N);
        Matrix<float> B(N, N);
        Matrix<float> C_our(N, N);
#ifdef TENSORIUM_USE_CBLAS
        Matrix<float> C_ref(N, N);
#endif

#pragma omp parallel
        {
            std::mt19937                          rng(42 + omp_get_thread_num());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j) {
                    A(i, j) = dist(rng);
                    B(i, j) = dist(rng);
                }
        }

        std::cout << "\n=== Benchmarking N = " << N << " ===\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        C_our = mul_mat(A, B);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double      elapsed_our = std::chrono::duration<double>(t1 - t0).count();
        const long double flops = 2.0L * N * N * N;
        const double      gflops_our = static_cast<double>(flops / 1e9L) / elapsed_our;

        std::cout << "[Tensorium GEMM]\n";
        std::cout << "Time      : " << elapsed_our << " s\n";
        std::cout << "GFLOP/s   : " << gflops_our << "\n";
        std::cout << "Sample C(0,0): " << C_our(0, 0) << "\n";

#ifdef TENSORIUM_USE_CBLAS
        auto t2 = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)N, (int)N, 1.0f,
                    A.data.data(), (int)N, B.data.data(), (int)N, 0.0f, C_ref.data.data(), (int)N);
        auto t3 = std::chrono::high_resolution_clock::now();

        const double elapsed_blas = std::chrono::duration<double>(t3 - t2).count();
        const double gflops_blas = static_cast<double>(flops / 1e9L) / elapsed_blas;

        std::cout << "\n[OpenBLAS SGEMM]\n";
        std::cout << "Time      : " << elapsed_blas << " s\n";
        std::cout << "GFLOP/s   : " << gflops_blas << "\n";
        std::cout << "Sample C(0,0): " << C_ref(0, 0) << "\n";

        double max_abs_diff = 0.0;
        double sq_norm = 0.0;
        size_t mismatch_printed = 0;
        constexpr size_t max_to_show = 5;
        for (size_t col = 0; col < N; ++col) {
            for (size_t row = 0; row < N; ++row) {
                const size_t idx = col * N + row;
                const double diff = double(C_ref.data[idx] - C_our.data[idx]);
                max_abs_diff = std::max(max_abs_diff, std::abs(diff));
                sq_norm += diff * diff;
                if (std::abs(diff) > 0 && mismatch_printed < max_to_show) {
                    std::cout << "  diff(" << row << "," << col << ") = " << diff << '\n';
                    ++mismatch_printed;
                }
            }
        }
        const double frob = std::sqrt(sq_norm);
        std::cout << "Max |C_tensorium - C_ref| : " << max_abs_diff << "\n";
        std::cout << "||C_tensorium - C_ref||_F = " << frob << "\n";
        if (mismatch_printed == 0)
            std::cout << "(Matrices identiques à la précision machine)\n";

        const double speedup = gflops_our / gflops_blas;
        std::cout << "\n[Summary]\n";
        std::cout << "Tensorium : " << gflops_our << " GFLOP/s\n";
        std::cout << "BLAS      : " << gflops_blas << " GFLOP/s\n";
        std::cout << "Speedup   : x" << speedup << "\n";

        csv << N << "," << gflops_our << "," << gflops_blas << "," << elapsed_our << ","
            << elapsed_blas << "," << speedup << "," << max_abs_diff << "," << frob << "\n";
#else
        csv << N << "," << gflops_our << "," << elapsed_our << "\n";
#endif
    }

    csv.close();
    std::cout << "\nResults written to: " << csv_path << "\n";
    return 0;
}

#ifdef TENSORIUM_USE_FFTW
std::vector<std::complex<double>> make_signal(size_t N) {
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> data(N);
    for (size_t i = 0; i < N; ++i)
        data[i] = {dist(rng), dist(rng)};
    return data;
}

struct FFTBenchmarkResult {
    double tensorium_time;
    double fftw_time;
    double max_error;
};

FFTBenchmarkResult benchmark_fft_size(size_t N, size_t iters) {
    FFTBenchmarkResult result{0.0, 0.0, 0.0};
    Vector<std::complex<double>> tensorium_data(N);
    auto base = make_signal(N);
    for (size_t i = 0; i < N; ++i)
        tensorium_data[i] = base[i];
    double total_tensorium = 0.0;
    for (size_t iter = 0; iter < iters; ++iter) {
        for (size_t i = 0; i < N; ++i)
            tensorium_data[i] = base[i];
        auto start = std::chrono::high_resolution_clock::now();
        SpectralFFT<double>::forward(tensorium_data);
        SpectralFFT<double>::backward(tensorium_data);
        auto end = std::chrono::high_resolution_clock::now();
        total_tensorium += std::chrono::duration<double>(end - start).count();
    }
    result.tensorium_time = total_tensorium / iters;

    std::vector<std::complex<double>> fftw_in = base;
    std::vector<std::complex<double>> fftw_out(N);
    fftw_plan plan_fwd =
        fftw_plan_dft_1d(static_cast<int>(N), reinterpret_cast<fftw_complex *>(fftw_in.data()),
                         reinterpret_cast<fftw_complex *>(fftw_out.data()), FFTW_FORWARD,
                         FFTW_MEASURE);
    fftw_plan plan_bwd =
        fftw_plan_dft_1d(static_cast<int>(N), reinterpret_cast<fftw_complex *>(fftw_out.data()),
                         reinterpret_cast<fftw_complex *>(fftw_in.data()), FFTW_BACKWARD,
                         FFTW_MEASURE);
    double total_fftw = 0.0;
    for (size_t iter = 0; iter < iters; ++iter) {
        fftw_in = base;
        auto start = std::chrono::high_resolution_clock::now();
        fftw_execute(plan_fwd);
        fftw_execute(plan_bwd);
        auto end = std::chrono::high_resolution_clock::now();
        total_fftw += std::chrono::duration<double>(end - start).count();
    }
    result.fftw_time = total_fftw / iters;

    const double scale = 1.0 / static_cast<double>(N);
    for (size_t i = 0; i < N; ++i) {
        std::complex<double> fftw_val = fftw_in[i] * scale;
        std::complex<double> tensorium_val = tensorium_data[i];
        result.max_error = std::max(result.max_error,
                                    std::abs(fftw_val.real() - tensorium_val.real()));
        result.max_error = std::max(result.max_error,
                                    std::abs(fftw_val.imag() - tensorium_val.imag()));
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);

    return result;
}
#endif

void run_fft_benchmark() {
#ifdef TENSORIUM_USE_FFTW
    std::vector<size_t> sizes = {1024, 4096, 16384};
    const size_t        iterations = 5;
    std::cout << "\nBenchmarking Tensorium FFT vs FFTW\n";
    for (size_t N : sizes) {
        auto res = benchmark_fft_size(N, iterations);
        std::cout << "\nSize N = " << N << "\n";
        std::cout << "Tensorium avg time : " << res.tensorium_time << " s\n";
        std::cout << "FFTW avg time      : " << res.fftw_time << " s\n";
        std::cout << "Speedup (Tensorium/FFTW): " << (res.tensorium_time / res.fftw_time)
                  << "\n";
        std::cout << "Max |difference|   : " << res.max_error << "\n";
    }
#else
    std::cout << "\n[FFT Benchmark] FFTW not available. Configure with FFTW to enable." << std::endl;
#endif
}

} // namespace

int main() {
    int status = run_matrix_benchmark();
    run_fft_benchmark();
    return status;
}
