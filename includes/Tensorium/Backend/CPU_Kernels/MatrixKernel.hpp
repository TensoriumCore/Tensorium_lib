#pragma once

<<<<<<< HEAD:includes/Tensorium/Core/MatrixKernels/MatrixKernel.hpp
#include "../Matrix.hpp"
#define TENSORIUM_DUMP     __attribute__((noinline, used, annotate("tensorium_dump")))
#define TENSORIUM_ANNOTATE __attribute__((annotate("tensorium_dump")))
=======
#include "../../Core/Matrix.hpp"
>>>>>>> 2226d95657c81e6fb423aba1d20316dc597cb1a6:includes/Tensorium/Backend/CPU_Kernels/MatrixKernel.hpp

namespace tensorium {

/**
 * @brief MatrixKernel provides specialized SIMD-accelerated matrix multiplication routines
 *        for statically-sized square matrices.
 *
 * This class inherits from a column-major `Matrix<K, true>` and offers optimized
 * kernels for specific square sizes (2x2, 3x3, ..., 64x64), using AVX/SIMD intrinsics
 * for high performance. The kernels exploit register-level blocking and FMADD chaining.
 *
 * @tparam K Scalar type (float, double, etc.)
 */
template <typename K> class MatrixKernel : public Matrix<K, true> {
  public:
    using Matrix<K, true>::rows;
    using Matrix<K, true>::cols;
    using Matrix<K, true>::data;
    using Matrix<K, true>::operator();

    using Simd = simd::SimdTraits<K, DefaultISA>;
    using reg = typename Simd::reg;
    /**
     * @brief Construct a MatrixKernel from a column-major matrix.
     * @param m Source matrix.
     */
    MatrixKernel(const Matrix<K, true> &m) : Matrix<K, true>(m) {}
    /**
     * @brief Construct a MatrixKernel from a row-major matrix by copying elements.
     * @param m Source matrix.
     */
    MatrixKernel(const Matrix<K, false> &m) : Matrix<K, true>(m.rows, m.cols) {
        for (size_t i = 0; i < m.rows; ++i)
            for (size_t j = 0; j < m.cols; ++j)
                (*this)(i, j) = m(i, j);
    }

    /**
     * @brief Construct an empty column-major matrix kernel of size (r × c).
     */
    MatrixKernel(size_t r, size_t c) : Matrix<K, true>(r, c) {}
    /**
     * @brief Multiply two 2×2 matrices using SIMD.
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
	TENSORIUM_DUMP
	__attribute__((noinline))
    inline Matrix<K> mul_mat2x2(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> C(2, 2);

        reg b_col0 = Simd::loadu(&B.data[0]);
        reg b_col1 = Simd::loadu(&B.data[2]);

        K b00 = Simd::extract(b_col0, 0);
        K b10 = Simd::extract(b_col0, 1);
        K b01 = Simd::extract(b_col1, 0);
        K b11 = Simd::extract(b_col1, 1);

        C(0, 0) = (*this)(0, 0) * b00 + (*this)(0, 1) * b10;
        C(1, 0) = (*this)(1, 0) * b00 + (*this)(1, 1) * b10;
        C(0, 1) = (*this)(0, 0) * b01 + (*this)(0, 1) * b11;
        C(1, 1) = (*this)(1, 0) * b01 + (*this)(1, 1) * b11;

        return C;
    }
    /**
     * @brief Multiply two 3×3 matrices using SIMD.
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat3x3(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(3, 3);

        for (int i = 0; i < 3; ++i) {
            reg ai0 = Simd::set1((*this)(i, 0));
            reg ai1 = Simd::set1((*this)(i, 1));
            reg ai2 = Simd::set1((*this)(i, 2));

            alignas(32) K brow0[4] = {B(0, 0), B(0, 1), B(0, 2), K(0)};
            alignas(32) K brow1[4] = {B(1, 0), B(1, 1), B(1, 2), K(0)};
            alignas(32) K brow2[4] = {B(2, 0), B(2, 1), B(2, 2), K(0)};

            reg b0 = Simd::loadu(brow0);
            reg b1 = Simd::loadu(brow1);
            reg b2 = Simd::loadu(brow2);

            reg acc = Simd::mul(ai0, b0);
            acc = Simd::fmadd(ai1, b1, acc);
            acc = Simd::fmadd(ai2, b2, acc);

            result(i, 0) = Simd::extract(acc, 0);
            result(i, 1) = Simd::extract(acc, 1);
            result(i, 2) = Simd::extract(acc, 2);
        }

        return result;
    }

    /**
     * @brief Multiply two 4×4 matrices using SIMD.
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat4x4(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(4, 4);

        reg brow0 = Simd::loadu(&B.data[0 * 4 + 0]);
        reg brow1 = Simd::loadu(&B.data[1 * 4 + 0]);
        reg brow2 = Simd::loadu(&B.data[2 * 4 + 0]);
        reg brow3 = Simd::loadu(&B.data[3 * 4 + 0]);

        {
            const K *a = &data[0 * 4];
            reg      a0 = Simd::set1(a[0]);
            reg      a1 = Simd::set1(a[1]);
            reg      a2 = Simd::set1(a[2]);
            reg      a3 = Simd::set1(a[3]);

            reg acc0 = Simd::mul(a0, brow0);
            acc0 = Simd::fmadd(a1, brow1, acc0);
            acc0 = Simd::fmadd(a2, brow2, acc0);
            acc0 = Simd::fmadd(a3, brow3, acc0);

            Simd::storeu(&result.data[0 * 4 + 0], acc0);
        }

        {
            const K *a = &data[1 * 4];
            reg      a0 = Simd::set1(a[0]);
            reg      a1 = Simd::set1(a[1]);
            reg      a2 = Simd::set1(a[2]);
            reg      a3 = Simd::set1(a[3]);

            reg acc1 = Simd::mul(a0, brow0);
            acc1 = Simd::fmadd(a1, brow1, acc1);
            acc1 = Simd::fmadd(a2, brow2, acc1);
            acc1 = Simd::fmadd(a3, brow3, acc1);

            Simd::storeu(&result.data[1 * 4 + 0], acc1);
        }

        {
            const K *a = &data[2 * 4];
            reg      a0 = Simd::set1(a[0]);
            reg      a1 = Simd::set1(a[1]);
            reg      a2 = Simd::set1(a[2]);
            reg      a3 = Simd::set1(a[3]);

            reg acc2 = Simd::mul(a0, brow0);
            acc2 = Simd::fmadd(a1, brow1, acc2);
            acc2 = Simd::fmadd(a2, brow2, acc2);
            acc2 = Simd::fmadd(a3, brow3, acc2);

            Simd::storeu(&result.data[2 * 4 + 0], acc2);
        }

        {
            const K *a = &data[3 * 4];
            reg      a0 = Simd::set1(a[0]);
            reg      a1 = Simd::set1(a[1]);
            reg      a2 = Simd::set1(a[2]);
            reg      a3 = Simd::set1(a[3]);

            reg acc3 = Simd::mul(a0, brow0);
            acc3 = Simd::fmadd(a1, brow1, acc3);
            acc3 = Simd::fmadd(a2, brow2, acc3);
            acc3 = Simd::fmadd(a3, brow3, acc3);

            Simd::storeu(&result.data[3 * 4 + 0], acc3);
        }

        return result;
    }

    /**
     * @brief Multiply two 8×8 matrices using SIMD.
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat8x8(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(8, 8);

        reg col[8];
        for (int j = 0; j < 8; ++j)
            col[j] = Simd::loadu(&B.data[j * 8]);

        for (int i = 0; i < 8; ++i) {
            const K *a = &data[i * 8];
            reg      a0 = Simd::set1(a[0]);
            reg      a1 = Simd::set1(a[1]);
            reg      a2 = Simd::set1(a[2]);
            reg      a3 = Simd::set1(a[3]);
            reg      a4 = Simd::set1(a[4]);
            reg      a5 = Simd::set1(a[5]);
            reg      a6 = Simd::set1(a[6]);
            reg      a7 = Simd::set1(a[7]);

            reg acc = Simd::mul(a0, col[0]);
            acc = Simd::fmadd(a1, col[1], acc);
            acc = Simd::fmadd(a2, col[2], acc);
            acc = Simd::fmadd(a3, col[3], acc);
            acc = Simd::fmadd(a4, col[4], acc);
            acc = Simd::fmadd(a5, col[5], acc);
            acc = Simd::fmadd(a6, col[6], acc);
            acc = Simd::fmadd(a7, col[7], acc);

            Simd::storeu(&result.data[i * 8], acc);
        }
        return result;
    }

    /**
     * @brief Multiply two 16×16 matrices using SIMD with FMADD accumulation.
     *        This function splits each row into two registers (low/high).
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat16x16(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(16, 16);

        reg row_lo[16], row_hi[16];
        for (int k = 0; k < 16; ++k) {
            row_lo[k] = Simd::loadu(&B.data[k * 16 + 0]);
            row_hi[k] = Simd::loadu(&B.data[k * 16 + 8]);
        }

        for (int i = 0; i < 16; ++i) {
            const K *a = &data[i * 16];

            reg acc_lo = Simd::mul(Simd::set1(a[0]), row_lo[0]);
            reg acc_hi = Simd::mul(Simd::set1(a[0]), row_hi[0]);

            for (int k = 1; k < 16; ++k) {
                reg ak = Simd::set1(a[k]);
                acc_lo = Simd::fmadd(ak, row_lo[k], acc_lo);
                acc_hi = Simd::fmadd(ak, row_hi[k], acc_hi);
            }

            Simd::storeu(&result.data[i * 16 + 0], acc_lo);
            Simd::storeu(&result.data[i * 16 + 8], acc_hi);
        }

        return result;
    }
    /**
     * @brief Multiply two 32×32 matrices using SIMD.
     *        Each row is split into two registers (16 elements each).
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat32x32(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(32, 32);

        reg brow[32][2];
#pragma unroll(2)
        for (int k = 0; k < 32; ++k) {
            brow[k][0] = Simd::loadu(&B.data[k * 32 + 0]);
            brow[k][1] = Simd::loadu(&B.data[k * 32 + 16]);
        }
#pragma unroll(2)
        for (int i = 0; i < 32; ++i) {
            const K *a = &data[i * 32];
            reg      acc0 = Simd::mul(Simd::set1(a[0]), brow[0][0]);
            reg      acc1 = Simd::mul(Simd::set1(a[0]), brow[0][1]);

            for (int k = 1; k < 32; ++k) {
                reg ak = Simd::set1(a[k]);
                acc0 = Simd::fmadd(ak, brow[k][0], acc0);
                acc1 = Simd::fmadd(ak, brow[k][1], acc1);
            }

            Simd::storeu(&result.data[i * 32 + 0], acc0);
            Simd::storeu(&result.data[i * 32 + 16], acc1);
        }

        return result;
    }

    /**
     * @brief Multiply two 64×64 matrices using SIMD.
     *        Each row is split into 4 SIMD registers (4×16 elements).
     *        Vectorized FMADD chaining is used for performance.
     * @param B Right-hand matrix.
     * @return Result of multiplication.
     */
    inline Matrix<K> mul_mat64x64(const MatrixKernel<K> &B) const {
        using Simd = simd::SimdTraits<K, DefaultISA>;
        using reg = typename Simd::reg;

        Matrix<K> result(64, 64);

        reg brow_lo[64], brow_hi[64], brow_32[64], brow_48[64];

        for (int k = 0; k < 64; ++k) {
            const K *b = &B.data[k * 64];
            brow_lo[k] = Simd::loadu(b + 0);
            brow_hi[k] = Simd::loadu(b + 8);
            brow_32[k] = Simd::loadu(b + 16);
            brow_48[k] = Simd::loadu(b + 24);
        }

        for (int i = 0; i < 64; ++i) {
            const K *a = &data[i * 64];

            reg acc0 = Simd::mul(Simd::set1(a[0]), brow_lo[0]);
            reg acc1 = Simd::mul(Simd::set1(a[0]), brow_hi[0]);
            reg acc2 = Simd::mul(Simd::set1(a[0]), brow_32[0]);
            reg acc3 = Simd::mul(Simd::set1(a[0]), brow_48[0]);

            for (int k = 1; k < 64; ++k) {
                reg ak = Simd::set1(a[k]);
                acc0 = Simd::fmadd(ak, brow_lo[k], acc0);
                acc1 = Simd::fmadd(ak, brow_hi[k], acc1);
                acc2 = Simd::fmadd(ak, brow_32[k], acc2);
                acc3 = Simd::fmadd(ak, brow_48[k], acc3);
            }

            K *r = &result.data[i * 64];
            Simd::storeu(r + 0, acc0);
            Simd::storeu(r + 8, acc1);
            Simd::storeu(r + 16, acc2);
            Simd::storeu(r + 24, acc3);
        }

        return result;
    }
};
} // namespace tensorium
