#pragma once
#include "Matrix.hpp"
#include "Vector.hpp"
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"

namespace morpheus::solver {
	template<typename K>
		class Jacobi;

	template<typename K>
		class Gauss {
			public:
				size_t rows() const;
				size_t block_size;
				aligned_vector<K> data;

				__attribute__((always_inline, hot, flatten))
					static inline Vector<K> solve(const Matrix<K>& A, const Vector<K>& b) {
						static_assert(std::is_floating_point<K>::value, "Gauss solver only supports floating point types.");
						assert(A.rows == A.cols && "Matrix A must be square");
						assert(A.rows == b.size() && "Matrix/vector size mismatch");

						if (A.rows > 2048 || A.cols > 2048) {
							std::cerr << "[warning] Switching to iterative Jacobi solver for large matrix (" << A.rows << "x" << A.cols << ")\n";
							return Jacobi<K>::solve(A, b);
						}

						size_t n = A.rows;
						Matrix<K> M = A;
						Vector<K> B = b;
						Vector<K> x(n);

						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;

						for (size_t i = 0; i < n; i++) {
							size_t max_row = i;
							K max_val = std::abs(M(i, i));
							for (size_t j = i + 1; j < n; ++j) {
								K abs_val = std::abs(M(j, i));
								if (abs_val > max_val) {
									max_val = abs_val;
									max_row = j;
								}
							}

							if (max_row != i) {
								M.swap_rows(i, max_row);
								std::swap(B[i], B[max_row]);
							}

							K pivot = M(i, i);
							if (pivot == K(0)) {
								throw std::runtime_error("Matrix is singular.");
							}

#pragma omp parallel for schedule(dynamic, 4)
							for (size_t j = i + 1; j < n; ++j) {
								K factor = M(j, i) / pivot;
								reg factor_vec = Simd::set1(factor);

								size_t k = i;
								for (; k + 2 * simd_width <= n; k += 2 * simd_width) {
									reg rj1 = Simd::loadu(&M(j, k));
									reg ri1 = Simd::loadu(&M(i, k));
									rj1 = Simd::sub(rj1, Simd::mul(factor_vec, ri1));
									Simd::storeu(&M(j, k), rj1);

									reg rj2 = Simd::loadu(&M(j, k + simd_width));
									reg ri2 = Simd::loadu(&M(i, k + simd_width));
									rj2 = Simd::sub(rj2, Simd::mul(factor_vec, ri2));
									Simd::storeu(&M(j, k + simd_width), rj2);
								}

								for (; k < n; ++k) {
									M(j, k) -= factor * M(i, k);
								}

								B[j] -= factor * B[i];
							}
						}

						for (ssize_t i = static_cast<ssize_t>(n) - 1; i >= 0; --i) {
							reg acc = Simd::setzero();
							size_t j = i + 1;
							for (; j + simd_width - 1 < n; j += simd_width) {
								reg mvec = Simd::loadu(&M(i, j));
								reg xvec = Simd::loadu(&x[j]);
								acc = Simd::fmadd(mvec, xvec, acc);
							}

							K sum = Simd::horizontal_add(acc);
							for (; j < n; ++j) {
								sum += M(i, j) * x[j];
							}

							x[i] = (B[i] - sum) / M(i, i);
						}

						return x;
					}
		};

	template<typename K>
		class Jacobi {
			public:
				aligned_vector<K> data;
				static inline Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-10, int max_iter = 5000) {
					static_assert(std::is_floating_point<K>::value, "Jacobi solver requires floating-point type.");
					assert(A.rows == A.cols && "Matrix A must be square");
					assert(A.rows == b.size() && "Matrix/vector size mismatch");
					const size_t n = A.rows;
					Vector<K> x(n, K(0)); 
					Vector<K> x_new(n, K(0));

					using Simd = simd::SimdTraits<K, DefaultISA>;
					using reg = typename Simd::reg;
					const size_t simd_width = Simd::width;

					for (int iter = 0; iter < max_iter; ++iter) {
#pragma omp parallel for schedule(dynamic, 4) 
						for (size_t i = 0; i < n; ++i) {
							if (std::abs(A(i, i)) < 1e-10)
								throw std::runtime_error("Jacobi: division by near-zero on diagonal, matrix likely not diagonally dominant.");

							reg sum_vec = Simd::setzero();
							size_t j = 0;

							for (; j + simd_width <= n; j += simd_width) {
								reg a_vec = Simd::loadu(&A(i, j));
								reg x_vec = Simd::loadu(&x[j]);

								if (i >= j && i < j + simd_width) {
									std::array<K, simd_width> mask_arr;
									Simd::storeu(mask_arr.data(), a_vec);
									mask_arr[i - j] = K(0);
									a_vec = Simd::loadu(mask_arr.data());
								}

								sum_vec = Simd::fma(a_vec, x_vec, sum_vec); 
							}

							K sigma = Simd::horizontal_add(sum_vec);

							for (; j < n; ++j) {
								if (j != i)
									sigma += A(i, j) * x[j];
							}

							x_new[i] = (b[i] - sigma) / A(i, i);
						}

						K err = K(0);
#pragma omp parallel for reduction(+:err) schedule(dynamic)
						for (size_t i = 0; i < n; ++i) {
							K diff = x_new[i] - x[i];
							err += diff * diff;
						}
						err = std::sqrt(err);

						if (err < tol)
							break;

						x = x_new;
					}

					return x;
				}
		};

	template<typename K>
		class GaussSeidel {
			public:
				aligned_vector<K> data;
				static Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-10, int max_iter = 5000);
		};

}

