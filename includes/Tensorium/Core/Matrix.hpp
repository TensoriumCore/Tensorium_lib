#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <cassert>
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"
#include "Vector.hpp"
#include "../MathUtils/MathsUtils.hpp"
#include "MatrixKernels/GemmKernel_bigger.hpp"

namespace tensorium {
	/**
	 * @brief High-performance aligned matrix class with SIMD support
	 *
	 * This class provides a dense matrix container with:
	 * - Aligned memory allocation for SIMD operations
	 * - Fast matrix arithmetic (add, sub, scale)
	 * - Optimized matrix-vector and matrix-matrix multiplication
	 * - Support for inversion, determinant, transpose, and rank
	 *
	 * @tparam K Scalar type (float, double, etc.)
	 */
	template<typename K>
		 class Matrix {
			 public:
				 size_t rows, cols;
				 aligned_vector<K> data;
				 size_t block_size;
				 /**
				  * @brief Construct a matrix of size r × c, initialized with zeros
				  */
				 Matrix(size_t r, size_t c) 
					 : rows(r), cols(c), data(r * c, K()), block_size(detect_optimal_block_size()) {
					 }

				 using Simd = simd::SimdTraits<K, DefaultISA>;
				 using reg = typename Simd::reg;
				 size_t simd_width = Simd::width;
				 /** @brief Return the total number of elements */
				 size_t size() const { 
					 return rows * cols; 
				 }
				 /** @brief Element access (mutable) */

				 K& operator()(size_t i, size_t j) {
					 return data[j * rows + i];
				 }

				 const K& operator()(size_t i, size_t j) const {
					 return data[j * rows + i];
				 }

				 /** @brief Print the matrix to stdout */
				 void print() const {
					 for (size_t i = 0; i < rows; ++i) {
						 std::cout << "[ ";
						 for (size_t j = 0; j < cols; ++j)
							 std::cout << operator()(i, j) << " ";
						 std::cout << "]\n";
					 }
				 }
				 /** @brief Swap two rows of the matrix */
				 void swap_rows(size_t i, size_t j) {
					 assert(i < rows && j < rows);
					 for (size_t k = 0; k < cols; ++k) {
						 MathsUtils::_swap((*this)(i, k), (*this)(j, k));
					 }
				 }
				 /**
				  * @brief Multiply matrix by a vector (naïve fallback)
				  *
				  * \f[
				  * (Ax)_i = \sum_j A_{ij} x_j
				  * \f]
				  */
				 template <typename T>
					 Vector<T> operator*(const Vector<T>& v) const {
						 assert(cols == v.size() && "Matrix-Vector size mismatch");
						 Vector<T> result(rows);
						 for (auto& x : result) x = T(0);

						 for (size_t i = 0; i < rows; ++i) {
							 for (size_t j = 0; j < cols; ++j) {
								 result[i] += (*this)(i, j) * v[j];
							 }
						 }
						 return result;
					 }
				 /** @brief In-place matrix addition: this += m */
				 inline void add(const Matrix &m) {
					 if (rows != m.rows || cols != m.cols)
						 throw std::invalid_argument("Matrix sizes do not match");

					 using Simd = simd::SimdTraits<K, DefaultISA>;
					 using reg = typename Simd::reg;
					 const size_t simd_width = Simd::width;

					 size_t n = size();
					 size_t i = 0;

					 _mm_prefetch((const char *)&m.data[0], _MM_HINT_T0);

					 for (; i + 2 * simd_width - 1 < n; i += 2 * simd_width) {
						 reg a0 = Simd::load(&data[i]);
						 reg b0 = Simd::load(&m.data[i]);
						 a0 = Simd::add(a0, b0);
						 Simd::store(&data[i], a0);

						 reg a1 = Simd::load(&data[i + simd_width]);
						 reg b1 = Simd::load(&m.data[i + simd_width]);
						 a1 = Simd::add(a1, b1);
						 Simd::store(&data[i + simd_width], a1);
					 }

					 for (; i < n; ++i)
						 data[i] += m.data[i];
				 }
				 /** @brief In-place matrix subtraction: this -= m */
				 inline void sub(const Matrix &m) {
					 if (rows != m.rows || cols != m.cols) 
						 throw std::invalid_argument("Matrix sizes do not match");
					 using Simd = simd::SimdTraits<K, DefaultISA>;
					 using reg = typename Simd::reg;
					 const size_t simd_width = Simd::width;

					 size_t n = size();
					 size_t i = 0;

					 _mm_prefetch((const char *)&m.data[0], _MM_HINT_T0);
					 for (; i + 15 < n; i += 16) {
						 reg a0 = Simd::load(&data[i]);
						 reg b0 = Simd::load(&m.data[i]);
						 a0 = Simd::sub(a0, b0);
						 Simd::store(&data[i], a0);

						 reg a1 = Simd::load(&data[i + simd_width]);
						 reg b1 = Simd::load(&m.data[i + simd_width]);
						 a1 = Simd::sub(a1, b1);
						 Simd::store(&data[i + simd_width], a1);
					 }
					 for (; i < size(); ++i) {
						 data[i] -= m.data[i];
					 }
				 }
				 /** @brief In-place scalar multiplication: this *= a */
				 inline void scl(K a) {
					 size_t n = size();
					 size_t i = 0;
					 using Simd = simd::SimdTraits<K, DefaultISA>;
					 using reg = typename Simd::reg;
					 const size_t simd_width = Simd::width;
					 _mm_prefetch((const char *)&data[0], _MM_HINT_T0);
					 reg scalar = Simd::set1(a);

					 for (; i + 15 < n; i += 16) {
						 reg v0 = Simd::load(&data[i]);
						 v0 = Simd::mul(v0, scalar);
						 Simd::store(&data[i], v0);

						 reg v1 = Simd::load(&data[i + simd_width]);
						 v1 = Simd::mul(v1, scalar);
						 Simd::store(&data[i + simd_width], v1);
					 }

					 for (; i < n; ++i)
						 data[i] *= a;
				 }
				
				 /** @brief Linearly interpolate between two matrices: this = (1 - α) * A + α * B */
				 inline void lerp(const Matrix<K>& A, const Matrix<K>& B, K alpha) {
					 if (A.rows != B.rows || A.cols != B.cols || rows != A.rows || cols != A.cols)
						 throw std::invalid_argument("Matrix size mismatch for lerp");

					 using Simd = simd::SimdTraits<K, DefaultISA>;
					 using reg = typename Simd::reg;
					 const size_t simd_width = Simd::width;

					 size_t n = size();
					 size_t i = 0;

					 reg alpha_vec = Simd::set1(alpha);
					 reg one_minus_alpha_vec = Simd::set1(K(1) - alpha);

					 for (; i + 2 * simd_width - 1 < n; i += 2 * simd_width) {
						 reg a0 = Simd::load(&A.data[i]);
						 reg b0 = Simd::load(&B.data[i]);
						 reg r0 = Simd::fmadd(one_minus_alpha_vec, a0, Simd::mul(alpha_vec, b0));
						 Simd::store(&data[i], r0);

						 reg a1 = Simd::load(&A.data[i + simd_width]);
						 reg b1 = Simd::load(&B.data[i + simd_width]);
						 reg r1 = Simd::fmadd(one_minus_alpha_vec, a1, Simd::mul(alpha_vec, b1));
						 Simd::store(&data[i + simd_width], r1);
					 }

					 for (; i < n; ++i) {
						 data[i] = (K(1) - alpha) * A.data[i] + alpha * B.data[i];
					 }
				 }
				 /**
				  * @brief Multiply matrix by another matrix using optimized SIMD path
				  * 
				  * Uses blocking and micro-kernels to avoid cache bottleneck with FMA/AVX units repartition. Fast-paths exist for 4×4, 8×8, and 16×16.
				  */
				 inline Matrix _mul_mat(const Matrix<K>& mat) const {
					 if (cols != mat.rows)
						 throw std::invalid_argument("Matrix dimensions do not match for multiplication");

					 Matrix<K> result(rows, mat.cols);

					 const K* A = data.data();               // Already column-major (this)
					 const K* B = mat.data.data();           // Already column-major (rhs)
					 K* C       = result.data.data();        // Output (also column-major)

					 tensorium::GemmKernelBigger<K> kernel;
					 kernel.matmul(
							 const_cast<K*>(A),
							 const_cast<K*>(B),
							 C,
							 static_cast<int>(rows),             // M
							 static_cast<int>(mat.cols),         // N
							 static_cast<int>(cols)              // K
							 );

					 return result;
				 }
				 /**
				  * @brief Multiply matrix by a vector using SIMD
				  *
				  * This is a faster alternative to operator* when available.
				  */
				 template<typename T>
					 inline Vector<T> mul_vec(const Vector<T>& x) const {
						 using Simd = simd::SimdTraits<T, DefaultISA>;
						 using reg  = typename Simd::reg;
						 constexpr size_t W = Simd::width;

						 assert(cols == x.size());

						 Vector<T> result(rows, T(0));

						 alignas(64) T buffer[W];

						 for (size_t i = 0; i < rows; ++i) {
							 reg acc = Simd::zero();
							 size_t j = 0;

							 for (; j + W <= cols; j += W) {
								 for (size_t w = 0; w < W; ++w)
									 buffer[w] = (*this)(i, j + w);  

								 reg A_vec = Simd::load(buffer);
								 reg x_vec = Simd::load(&x[j]);
								 acc = Simd::fmadd(A_vec, x_vec, acc);
							 }

							 T sum = Simd::horizontal_add(acc);

							 for (; j < cols; ++j)
								 sum += (*this)(i, j) * x[j];

							 result[i] = sum;
						 }

						 return result;
					 }

				 /** @brief Returns the transpose \f$ A^T \f$ of the matrix (column-major layout) */
				 inline Matrix<K> transpose() const {
					 Matrix<K> result(cols, rows); 

					 for (size_t i = 0; i < rows; ++i)
						 for (size_t j = 0; j < cols; ++j)
							 result(j, i) = (*this)(i, j);

					 return result;
				 }

				 /** @brief Returns the trace of a square matrix as a 1×1 matrix */
				 inline Matrix<K> trace() const {
					 if (rows != cols) {
						 throw std::invalid_argument("Matrix is not square");
					 }

					 Matrix<K> result(1, 1);
					 result(0, 0) = K(0);

					 for (size_t i = 0; i < rows; ++i) {
						 result(0, 0) += operator()(i, i);
					 }

					 return result;
				 }
				 /**
				  * @brief Compute the inverse of the matrix using Gauss–Jordan elimination
				  *
				  * Throws if the matrix is singular or not square.
				  * need to add a fallback for unsquare matrix if possible
				  */
				 inline Matrix<K> inverse() const {
					 if (rows != cols)
						 throw std::invalid_argument("Matrix must be square");

					 const auto n = rows;
					 Matrix<K> M(n, n);
					 Matrix<K> Inv(n, n);

					 for (auto i = decltype(n)(0); i < n; ++i) {
						 for (auto j = decltype(n)(0); j < n; ++j) {
							 M(i, j) = operator()(i, j);
							 Inv(i, j) = (i == j) ? K(1) : K(0);
						 }
					 }

					 using SimdT = simd::SimdTraits<K, DefaultISA>;
					 for (auto i = decltype(n)(0); i < n; ++i) {
						 auto piv = i;
						 auto maxv = MathsUtils::_abs(M(i, i));
						 for (auto r = i + 1; r < n; ++r) {
							 auto v = MathsUtils::_abs(M(r, i));
							 if (v > maxv) { maxv = v; piv = r; }
						 }
						 if (maxv < static_cast<K>(1e-6))
							 throw std::runtime_error("Matrix is singular or nearly singular.");

						 if (piv != i) {
							 M.swap_rows(i, piv);
							 Inv.swap_rows(i, piv);
						 }

						 auto diag = M(i, i);
						 auto diag_inv = K(1) / diag;
						 for (auto j = 0u; j < n; ++j) {
							 M(i, j) *= diag_inv;
							 Inv(i, j) *= diag_inv;
						 }

#pragma omp parallel for schedule(dynamic, UNROLL)
						 for (auto j = 0u; j < n; ++j) {
							 if (j != i) {
								 auto f = M(j, i);
								 for (auto k = 0u; k < n; ++k) {
									 M(j, k) -= f * M(i, k);
									 Inv(j, k) -= f * Inv(i, k);
								 }
							 }
						 }
					 }

					 return Inv;
				 }
				 /**
				  * @brief Compute the determinant using Gaussian elimination
				  *
				  * \f[
				  * \det A = \prod_{i=1}^n U_{ii} \quad \text{with } A = LU
				  * \f]
				  */
				 inline K det() const {
					 if (rows != cols)
						 throw std::invalid_argument("Matrix must be square");

					 const size_t n = rows;
					 Matrix<K> M(n, n);
					 using SimdT = simd::SimdTraits<K, DefaultISA>;
					 const size_t simd_width = SimdT::width;

					 for (size_t i = 0; i < n; ++i)
						 for (size_t j = 0; j < n; ++j)
							 M(i, j) = operator()(i, j);

					 K det_sign = K(1);  

					 for (size_t i = 0; i < n; ++i) {
						 size_t piv = i;
						 auto maxv = MathsUtils::_abs(M(i, i));
						 for (size_t r = i + 1; r < n; ++r) {
							 auto v = MathsUtils::_abs(M(r, i));
							 if (v > maxv) { maxv = v; piv = r; }
						 }
						 if (maxv < static_cast<K>(1e-12))
							 return K(0); 

						 if (piv != i) {
							 M.swap_rows(i, piv);
							 det_sign = -det_sign;
						 }

						 for (size_t j = i + 1; j < n; ++j) {
							 auto f = M(j, i) / M(i, i);
							 M(j, i) = K(0);

							 auto f_vec = SimdT::set1(-f);
							 size_t k = i + 1;

							 for (; k + simd_width - 1 < n; k += simd_width) {
								 auto mjk = SimdT::load(&M(j, k));
								 auto mik = SimdT::load(&M(i, k));
								 mjk = SimdT::fmadd(f_vec, mik, mjk);  
								 SimdT::store(&M(j, k), mjk);
							 }
							 for (; k < n; ++k) {
								 M(j, k) -= f * M(i, k);
							 }
						 }
					 }

					 K det = det_sign;
					 for (size_t i = 0; i < n; ++i)
						 det *= M(i, i);

					 return det;
				 }
				 /**
				  * @brief Compute the numerical rank of the matrix
				  *
				  * Performs row echelon reduction and counts non-zero pivots.
				  */
				 inline size_t rank(K eps = K(1e-6)) const {
					 Matrix<K> M(*this); 
					 const size_t m = rows;
					 const size_t n = cols;
					 size_t r = 0;

					 for (size_t col = 0; col < n; ++col) {
						 size_t pivot_row = r;
						 for (size_t i = r; i < m; ++i) {
							 if (MathsUtils::_abs(M(i, col)) > MathsUtils::_abs(M(pivot_row, col)))
								 pivot_row = i;
						 }

						 if (MathsUtils::_abs(M(pivot_row, col)) <= eps)
							 continue;

						 if (pivot_row != r)
							 M.swap_rows(pivot_row, r);

						 for (size_t i = r + 1; i < m; ++i) {
							 auto f = M(i, col) / M(r, col);
							 M(i, col) = 0;
							 for (size_t j = col + 1; j < n; ++j)
								 M(i, j) -= f * M(r, j);
						 }

						 ++r;
					 }

					 return r;
				 }
		 };
}

