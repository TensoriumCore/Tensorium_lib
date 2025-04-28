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

namespace morpheus {
	template<typename K>
		class Matrix {
			public:
				size_t rows, cols;
				aligned_vector<K> data;
				size_t block_size;
				Matrix(size_t r, size_t c) 
					: rows(r), cols(c), data(r * c, K()), block_size(detect_optimal_block_size()) {
					}

				using Simd = simd::SimdTraits<K, DefaultISA>;
				using reg = typename Simd::reg;
				size_t simd_width = Simd::width;
				size_t size() const { 
					return rows * cols; 
				}

				K& operator()(size_t i, size_t j) { return data[i * cols + j]; }
				const K& operator()(size_t i, size_t j) const { return data[i * cols + j]; }

				void print() const {
					for (size_t i = 0; i < rows; ++i) {
						std::cout << "[ ";
						for (size_t j = 0; j < cols; ++j)
							std::cout << operator()(i, j) << " ";
						std::cout << "]\n";
					}
				}

				void swap_rows(size_t i, size_t j) {
					assert(i < rows && j < rows);
					for (size_t k = 0; k < cols; ++k) {
						std::swap((*this)(i, k), (*this)(j, k));
					}
				}
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

				__attribute__((always_inline, hot, flatten))
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



				__attribute__((always_inline, hot, flatten))
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


				__attribute__((always_inline, hot, flatten))
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



				template<int UN>
					static inline __attribute__((always_inline))
					void microkernel4(size_t nCols,   
							const K* __restrict a,
							const K* const __restrict* b,
							reg* sum)
					{
						constexpr size_t W = Simd::width;

						for (size_t k = 0; k + W - 1 < nCols; k += W) {
							reg av = Simd::load(a + k);

#pragma unroll(UN)
							for (int x = 0; x < UN; ++x)
								sum[x] = Simd::fmadd(av, Simd::set1(b[x][k]), sum[x]);
						}
					}



				__attribute__((always_inline, hot, flatten))
					inline Matrix mul_mat(const Matrix<K>& mat) const {
						if (cols != mat.rows)
							throw std::invalid_argument("Matrix dimensions do not match for multiplication");
						if (cols == 4 && mat.rows == 4 && mat.cols == 4)
							return mul_mat_NxN<4>(mat);
						if (cols == 8 && mat.rows == 8 && mat.cols == 8)
							return mul_mat_NxN<8>(mat);
						if (cols == 16 && mat.rows == 16 && mat.cols == 16)
							return mul_mat_NxN<16>(mat);


						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg  = typename Simd::reg;
						constexpr size_t simd_width = Simd::width;
						constexpr size_t unroll     = UNROLL;   

						Matrix<K> result(rows, mat.cols);
						Matrix<K> mat_transposed(mat.cols, mat.rows);

#pragma omp parallel for schedule(static)
						for (size_t i = 0; i < mat.rows; ++i)
							for (size_t j = 0; j < mat.cols; ++j)
								mat_transposed(j, i) = mat(i, j);

#pragma omp parallel for collapse(2) schedule(dynamic)
						for (size_t ii = 0; ii < rows; ii += block_size) {
							for (size_t jj = 0; jj < mat.cols; jj += block_size) {
								const size_t i_end = std::min(ii + block_size, rows);
								const size_t j_end = std::min(jj + block_size, mat.cols);

								for (size_t i = ii; i < i_end; ++i) {
									for (size_t j = jj; j + unroll - 1 < j_end; j += unroll) {

										const K* __restrict__ a_ptr  = &data[i * cols];
									
										reg sum0 = Simd::zero(), sum1 = Simd::zero();
										reg sum2 = Simd::zero(), sum3 = Simd::zero();
										reg* sum_arr[4] = { &sum0, &sum1, &sum2, &sum3 };

										const K* __restrict__ b_ptr[4] = {
											&mat_transposed.data[(j + 0) * mat.rows],
											&mat_transposed.data[(j + 1) * mat.rows],
											&mat_transposed.data[(j + 2) * mat.rows],
											&mat_transposed.data[(j + 3) * mat.rows]
										};
#define UN 4
										reg sum[UN];
										microkernel4<UN>(cols, a_ptr, b_ptr, sum);


										K total0 = Simd::horizontal_add(sum0);
										K total1 = Simd::horizontal_add(sum1);
										K total2 = Simd::horizontal_add(sum2);
										K total3 = Simd::horizontal_add(sum3);

										for (size_t k = (cols & ~(simd_width-1)); k < cols; ++k) {
											K a_val = a_ptr[k];
											total0 += a_val * b_ptr[0][k];
											total1 += a_val * b_ptr[1][k];
											total2 += a_val * b_ptr[2][k];
											total3 += a_val * b_ptr[3][k];
										}


										reg res_vec = Simd::set(total3, total2, total1, total0);
										if (((uintptr_t)&result.data[i*cols+j] & 31) == 0)
											Simd::stream(&result.data[i * result.cols + j], res_vec);
										else
											Simd::store(&result.data[i * result.cols + j], res_vec);
									}

									for (size_t j = j_end - (j_end - jj) % unroll; j < j_end; ++j) {
										reg sum = Simd::zero();
										const K* __restrict__ a_ptr = &data[i * cols];
										const K* __restrict__ b_ptr = &mat_transposed.data[j * mat.rows];

										size_t k = 0;
										for (; k + simd_width - 1 < cols; k += simd_width) {
											reg a = Simd::load(a_ptr + k);
											reg b = Simd::load(b_ptr + k);
											sum   = Simd::fmadd(a, b, sum);
										}
										K total = Simd::horizontal_add(sum);
										for (; k < cols; ++k) total += a_ptr[k] * b_ptr[k];
										result(i, j) = total;
									}
								}
							}
						}
						return result;
					}

				template<size_t N>
					__attribute__((always_inline, flatten, hot))
					inline Matrix<K> mul_mat_NxN(const Matrix<K>& mat) const {
						static_assert(N == 4 || N == 8 || N == 16, "Only small NxN matrices supported");

						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg  = typename Simd::reg;
						constexpr size_t simd_width = Simd::width;

						Matrix<K> result(N, N);

						reg a[N];
						for (size_t i = 0; i < N; ++i) {
							if ((uintptr_t)&this->data[i * N] & 31) {
								Simd::load(&this->data[i * N]);
							}
							a[i] = Simd::loadu(&this->data[i * N]);
						}

						reg b[N];
						for (size_t j = 0; j < N; ++j) {
							b[j] = Simd::set(
									mat.data[j + 0 * N],
									mat.data[j + 1 * N],
									mat.data[j + 2 * N],
									mat.data[j + 3 * N]  
									);
						}

						for (size_t i = 0; i < N; ++i) {
							for (size_t j = 0; j < N; ++j) {
								result.data[i*N + j] = Simd::horizontal_add(Simd::mul(a[i], b[j]));
							}
						}

						return result;
					}

				template<typename T>
					__attribute__((always_inline, hot, flatten))
					inline Vector<T> mul_vec(const Vector<T>& x) const {
						using Simd = simd::SimdTraits<T, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;

						assert(cols == x.size());

						Vector<T> result(rows, T(0));

						for (size_t i = 0; i < rows; ++i) {
							reg acc = Simd::zero();

							size_t j = 0;
							for (; j + simd_width - 1 < cols; j += simd_width) {
								reg A_vec = Simd::load(&(*this)(i, j));
								reg x_vec = Simd::load(&x[j]);
								acc = Simd::fmadd(A_vec, x_vec, acc);
							}

							T sum = Simd::horizontal_add(acc);

							for (; j < cols; ++j) {
								sum += (*this)(i, j) * x[j];
							}

							result[i] = sum;
						}

						return result;
					}


				__attribute__((always_inline, hot, flatten))
					inline Matrix<K> transpose() const {
						Matrix<K> result(cols, rows); 
						if constexpr (std::is_same_v<K, float> || std::is_same_v<K, double>) {
							using ISA = DefaultISA;
							using Simd = simd::SimdTraits<K, ISA>;

							const size_t W = Simd::width;

							if (rows % W == 0 && cols % W == 0) {
								for (size_t i = 0; i < rows; i += W) {
									for (size_t j = 0; j < cols; j += W) {
										typename Simd::reg block[W];
										for (size_t k = 0; k < W; ++k)
											block[k] = Simd::load(&data[(i + k) * cols + j]);

										alignas(ALIGN) K tmp[W][W];
										for (size_t k = 0; k < W; ++k)
											Simd::store(tmp[k], block[k]);

										for (size_t x = 0; x < W; ++x)
											for (size_t y = 0; y < W; ++y)
												result(j + x, i + y) = tmp[y][x];
									}
								}
								return result;
							}
						}

						for (size_t i = 0; i < rows; ++i)
							for (size_t j = 0; j < cols; ++j)
								result(j, i) = operator()(i, j);

						return result;
					}

				__attribute__((always_inline, hot, flatten))
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

				__attribute__((always_inline, hot, flatten))
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
						using regT  = typename SimdT::reg;
						const auto W = SimdT::width;

						for (auto i = decltype(n)(0); i < n; ++i) {
							auto piv = i;
							auto maxv = std::abs(M(i, i));
							for (auto r = i + 1; r < n; ++r) {
								auto v = std::abs(M(r, i));
								if (v > maxv) { maxv = v; piv = r; }
							}
							if (maxv < static_cast<K>(1e-12))
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
			
				__attribute__((always_inline, hot, flatten))
					inline K det() const {
						if (rows != cols)
							throw std::invalid_argument("Matrix must be square");

						const size_t n = rows;
						Matrix<K> M(n, n);
						using SimdT = simd::SimdTraits<K, DefaultISA>;
						using reg = typename SimdT::reg;
						const size_t simd_width = SimdT::width;

						for (size_t i = 0; i < n; ++i)
							for (size_t j = 0; j < n; ++j)
								M(i, j) = operator()(i, j);

						K det_sign = K(1);  

						for (size_t i = 0; i < n; ++i) {
							size_t piv = i;
							auto maxv = std::abs(M(i, i));
							for (size_t r = i + 1; r < n; ++r) {
								auto v = std::abs(M(r, i));
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
		};
}

