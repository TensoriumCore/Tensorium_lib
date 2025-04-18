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
						std::cout << "Auto-selected BLOCK_SIZE = " << block_size << std::endl;
					}


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


				__attribute__((always_inline, hot, flatten))
					inline Matrix mul_mat(const Matrix<K>& mat) const {
						if (cols != mat.rows) {
							throw std::invalid_argument("Matrix dimensions do not match for multiplication");
						}

						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						constexpr size_t unroll = UNROLL;

						Matrix<K> result(rows, mat.cols);
						Matrix<K> mat_transposed(mat.cols, mat.rows);
#pragma omp parallel for collapse(2)
						for (size_t i = 0; i < mat.rows; ++i) {
							for (size_t j = 0; j < mat.cols; ++j) {
								mat_transposed(j, i) = mat(i, j);
							}
						}

#pragma omp parallel for collapse(2) schedule(static) shared(result)
						for (size_t ii = 0; ii < rows; ii += block_size) {
							for (size_t jj = 0; jj < mat.cols; jj += block_size) {
								const size_t i_end = std::min(ii + block_size, rows);
								const size_t j_end = std::min(jj + block_size, mat.cols);
								_mm_prefetch((const char *)&mat_transposed.data[jj * mat.rows], _MM_HINT_T0);
#pragma omp simd
								for (size_t i = ii; i < i_end; ++i) {
									for (size_t j = jj; j + unroll - 1 < j_end; j += unroll) {
										reg sum0 = Simd::zero();
										reg sum1 = Simd::zero();
										reg sum2 = Simd::zero();
										reg sum3 = Simd::zero();

										const K* a_ptr = &data[i * cols];
										const K* b_ptr0 = &mat_transposed.data[j * mat.rows];
										const K* b_ptr1 = b_ptr0 + mat.rows;
										const K* b_ptr2 = b_ptr1 + mat.rows;
										const K* b_ptr3 = b_ptr2 + mat.rows;

										size_t k = 0;
										for (; k + simd_width - 1 < cols; k += simd_width) {
											reg a = Simd::load(a_ptr + k);

											reg b0 = Simd::set1(b_ptr0[k]);
											reg b1 = Simd::set1(b_ptr1[k]);
											reg b2 = Simd::set1(b_ptr2[k]);
											reg b3 = Simd::set1(b_ptr3[k]);

											sum0 = Simd::fmadd(a, b0, sum0);
											sum1 = Simd::fmadd(a, b1, sum1);
											sum2 = Simd::fmadd(a, b2, sum2);
											sum3 = Simd::fmadd(a, b3, sum3);
										}

										K sum_array0[simd_width], sum_array1[simd_width],
										sum_array2[simd_width], sum_array3[simd_width];
										Simd::store(sum_array0, sum0);
										Simd::store(sum_array1, sum1);
										Simd::store(sum_array2, sum2);
										Simd::store(sum_array3, sum3);

										K total0 = K(0), total1 = K(0), total2 = K(0), total3 = K(0);		
#pragma omp simd reduction(+:total0,total1,total2,total3)
										for (size_t s = 0; s < simd_width; ++s) {
											total0 += sum_array0[s];
											total1 += sum_array1[s];
											total2 += sum_array2[s];
											total3 += sum_array3[s];
										}

										for (; k < cols; ++k) {
											total0 += a_ptr[k] * b_ptr0[k];
											total1 += a_ptr[k] * b_ptr1[k];
											total2 += a_ptr[k] * b_ptr2[k];
											total3 += a_ptr[k] * b_ptr3[k];
										}

										result(i, j)     = total0;
										result(i, j + 1) = total1;
										result(i, j + 2) = total2;
										result(i, j + 3) = total3;
									}

									for (size_t j = j_end - (j_end - jj) % unroll; j < j_end; ++j) {
										K sum = K(0);
										for (size_t k = 0; k < cols; ++k) {
											sum += data[i * cols + k] * mat_transposed(j, k);
										}
										result(i, j) = sum;
									}
								}
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
		};
}

