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
					static inline Vector<K> solve(const Matrix<K>& A_in, const Vector<K>& b_in) {
						static_assert(std::is_floating_point<K>::value, "");
						const auto n = A_in.rows;

						assert(n == A_in.cols && n == b_in.size());
						if (A_in.rows >= 1024 || A_in.cols >= 1024)
							return Jacobi<K>::solve(A_in, b_in);
						
						Matrix<K> M(n, n);
						Vector<K> B(n);
						for (auto i = decltype(n)(0); i < n; ++i) {
							B[i] = b_in[i];
							for (auto j = decltype(n)(0); j < n; ++j)
								M(i, j) = A_in(i, j);
						}

						Vector<K> x(n);
						using SimdT = simd::SimdTraits<K, DefaultISA>;
						using regT  = typename SimdT::reg;
						const auto W = SimdT::width;

						for (auto i = decltype(n)(0); i < n; ++i) {
							auto piv  = i;
							auto maxv = std::abs(M(i, i));
							for (auto r = i + 1; r < n; ++r) {
								auto v = std::abs(M(r, i));
								if (v > maxv) { maxv = v; piv = r; }
							}
							if (maxv < static_cast<K>(1e-12))
								throw std::runtime_error("Gauss: matrix is singular or nearly singular.");
							if (piv != i) {
								M.swap_rows(i, piv);
								std::swap(B[i], B[piv]);
							}

							constexpr auto TILE = UNROLL;
#pragma omp parallel for schedule(dynamic, TILE)
							for (auto j = i + 1; j < n; ++j) {
								auto* __restrict rowj = &M(j, 0);
								auto* __restrict rowi = &M(i, 0);
								auto f                = rowj[i] / rowi[i];
								rowj[i]               = K(0);

								const auto fv = SimdT::set1(f);
								auto k        = i + 1;
								for (; k + 4 * W <= n; k += 4 * W) {
									for (int t = 0; t < 4; ++t) {
										auto off = k + t * W;
										auto vj  = SimdT::loadu(rowj + off);
										auto vi  = SimdT::loadu(rowi + off);
										vj        = SimdT::sub(vj, SimdT::mul(fv, vi));
										SimdT::storeu(rowj + off, vj);
									}
								}
								for (; k + W <= n; k += W) {
									auto vj = SimdT::loadu(rowj + k);
									auto vi = SimdT::loadu(rowi + k);
									vj        = SimdT::sub(vj, SimdT::mul(fv, vi));
									SimdT::storeu(rowj + k, vj);
								}
								for (; k + 1 < n; k += 2) {
									rowj[k]     -= f * rowi[k];
									rowj[k + 1] -= f * rowi[k + 1];
								}
								if (k < n)
									rowj[k] -= f * rowi[k];

								B[j] -= f * B[i];
							}
						}

						for (auto ii = n; ii-- > 0;) {
							auto* __restrict rowi = &M(ii, 0);
							auto acc              = SimdT::setzero();
							auto j                = ii + 1;
							for (; j + W <= n; j += W) {
								auto u  = SimdT::loadu(rowi + j);
								auto xv = SimdT::loadu(&x[j]);
								acc     = SimdT::fmadd(u, xv, acc);
							}
							auto sum = SimdT::horizontal_add(acc);
							for (; j < n; ++j)
								sum += rowi[j] * x[j];

							x[ii] = (B[ii] - sum) / rowi[ii];
						}

						return x;
					}
		};

	template<typename K>
		class Jacobi {
			public:
				aligned_vector<K> data;
				static inline Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-10, int max_iter = 2000) {
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
				static Vector<K> solve(const Matrix<K>& A, const Vector<K>& b, K tol = 1e-8, int max_iter = 2000);
		};

}

