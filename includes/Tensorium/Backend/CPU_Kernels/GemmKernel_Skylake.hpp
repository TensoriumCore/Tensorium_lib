#pragma once

#include <Tensorium/Backend/OpenMP/OpenMPCompat.hpp>
#include <Tensorium/Backend/SIMD/SIMD.hpp>

#include <algorithm>
#include <thread>

namespace tensorium {

#if defined(TENSORIUM_X86) && defined(__AVX512F__)

template <typename T> class GemmKernelSkylake {
  public:
    using Simd = simd::SimdTraits<T, DefaultISA>;
    using reg = typename Simd::reg;

    static constexpr int VecWidth = static_cast<int>(Simd::width);
    static constexpr int MR = VecWidth * 2;
    static constexpr int NR = 8;
    static constexpr int BlockM = 192;
    static constexpr int BlockN = 128;
    static constexpr int BlockK = 256;

    static int thread_count() {
#    ifdef _OPENMP
        return omp_get_max_threads();
#    else
        unsigned int count = std::thread::hardware_concurrency();
        return count == 0 ? 1 : static_cast<int>(count);
#    endif
    }

    static inline void store_vector(T *dst, reg value, int rows_left, bool accumulate) {
        if (rows_left >= VecWidth) {
            if (accumulate)
                Simd::storeu(dst, Simd::add(Simd::loadu(dst), value));
            else
                Simd::storeu(dst, value);
            return;
        }

        alignas(Simd::alignment) T tmp[VecWidth];
        Simd::storeu(tmp, value);
        for (int i = 0; i < rows_left; ++i)
            if (accumulate)
                dst[i] += tmp[i];
            else
                dst[i] = tmp[i];
    }

    static inline void microkernel(const T *A, const T *B, T *C, int lda, int ldb, int ldc, int mr,
                                   int nr, int kc, bool accumulate) {
        reg c00 = Simd::zero(), c01 = Simd::zero();
        reg c10 = Simd::zero(), c11 = Simd::zero();
        reg c20 = Simd::zero(), c21 = Simd::zero();
        reg c30 = Simd::zero(), c31 = Simd::zero();
        reg c40 = Simd::zero(), c41 = Simd::zero();
        reg c50 = Simd::zero(), c51 = Simd::zero();
        reg c60 = Simd::zero(), c61 = Simd::zero();
        reg c70 = Simd::zero(), c71 = Simd::zero();

        for (int p = 0; p < kc; ++p) {
            const reg a0 = Simd::loadu(A + p * lda);
            const reg a1 = Simd::loadu(A + p * lda + VecWidth);

            if (nr >= 1) {
                const reg b = Simd::broadcast(B + p);
                c00 = Simd::fmadd(a0, b, c00);
                c01 = Simd::fmadd(a1, b, c01);
            }
            if (nr >= 2) {
                const reg b = Simd::broadcast(B + p + ldb);
                c10 = Simd::fmadd(a0, b, c10);
                c11 = Simd::fmadd(a1, b, c11);
            }
            if (nr >= 3) {
                const reg b = Simd::broadcast(B + p + 2 * ldb);
                c20 = Simd::fmadd(a0, b, c20);
                c21 = Simd::fmadd(a1, b, c21);
            }
            if (nr >= 4) {
                const reg b = Simd::broadcast(B + p + 3 * ldb);
                c30 = Simd::fmadd(a0, b, c30);
                c31 = Simd::fmadd(a1, b, c31);
            }
            if (nr >= 5) {
                const reg b = Simd::broadcast(B + p + 4 * ldb);
                c40 = Simd::fmadd(a0, b, c40);
                c41 = Simd::fmadd(a1, b, c41);
            }
            if (nr >= 6) {
                const reg b = Simd::broadcast(B + p + 5 * ldb);
                c50 = Simd::fmadd(a0, b, c50);
                c51 = Simd::fmadd(a1, b, c51);
            }
            if (nr >= 7) {
                const reg b = Simd::broadcast(B + p + 6 * ldb);
                c60 = Simd::fmadd(a0, b, c60);
                c61 = Simd::fmadd(a1, b, c61);
            }
            if (nr >= 8) {
                const reg b = Simd::broadcast(B + p + 7 * ldb);
                c70 = Simd::fmadd(a0, b, c70);
                c71 = Simd::fmadd(a1, b, c71);
            }
        }

        const int rows0 = std::min(mr, VecWidth);
        const int rows1 = std::max(0, mr - VecWidth);

        if (nr >= 1) {
            store_vector(C, c00, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + VecWidth, c01, rows1, accumulate);
        }
        if (nr >= 2) {
            store_vector(C + ldc, c10, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + ldc + VecWidth, c11, rows1, accumulate);
        }
        if (nr >= 3) {
            store_vector(C + 2 * ldc, c20, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 2 * ldc + VecWidth, c21, rows1, accumulate);
        }
        if (nr >= 4) {
            store_vector(C + 3 * ldc, c30, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 3 * ldc + VecWidth, c31, rows1, accumulate);
        }
        if (nr >= 5) {
            store_vector(C + 4 * ldc, c40, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 4 * ldc + VecWidth, c41, rows1, accumulate);
        }
        if (nr >= 6) {
            store_vector(C + 5 * ldc, c50, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 5 * ldc + VecWidth, c51, rows1, accumulate);
        }
        if (nr >= 7) {
            store_vector(C + 6 * ldc, c60, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 6 * ldc + VecWidth, c61, rows1, accumulate);
        }
        if (nr >= 8) {
            store_vector(C + 7 * ldc, c70, rows0, accumulate);
            if (rows1 > 0)
                store_vector(C + 7 * ldc + VecWidth, c71, rows1, accumulate);
        }
    }

    inline void matmul(T *A, T *B, T *C, int M, int N, int K) {
        const int threads = thread_count();

        for (int pb = 0; pb < K; pb += BlockK) {
            const int kc = std::min(BlockK, K - pb);

#    ifdef _OPENMP
#        pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
#    endif
            for (int jb = 0; jb < N; jb += BlockN) {
                for (int ib = 0; ib < M; ib += BlockM) {
                    const int nc = std::min(BlockN, N - jb);
                    const int mc = std::min(BlockM, M - ib);

                    for (int j = 0; j < nc; j += NR) {
                        const int nr = std::min(NR, nc - j);
                        for (int i = 0; i < mc; i += MR) {
                            const int mr = std::min(MR, mc - i);
                            microkernel(A + (ib + i) + pb * M, B + pb + (jb + j) * K,
                                        C + (ib + i) + (jb + j) * M, M, K, M, mr, nr, kc,
                                        pb != 0);
                        }
                    }
                }
            }
        }
    }
};

#endif

} // namespace tensorium
