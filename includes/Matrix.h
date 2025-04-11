#pragma once

#include "SIMD.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <stdexcept>
#include <mutex>


template<typename K>
class Matrix {
	public:
		size_t rows, cols;
		aligned_vector<K> data;
		constexpr static size_t BLOCK_SIZE = 64;
		Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, K()) {}
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


		__attribute__((always_inline, hot, flatten))
			inline void add(const Matrix &m) {
				if (rows != m.rows || cols != m.cols) 
					throw std::invalid_argument("Matrix sizes do not match");

				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&m.data[0], _MM_HINT_T0);

				for (; i + 15 < n; i += 16) {
					__m256 a0 = _mm256_load_ps(&data[i]);
					__m256 b0 = _mm256_load_ps(&m.data[i]);
					a0 = _mm256_add_ps(a0, b0);
					_mm256_store_ps(&data[i], a0);

					__m256 a1 = _mm256_load_ps(&data[i + 8]);
					__m256 b1 = _mm256_load_ps(&m.data[i + 8]);
					a1 = _mm256_add_ps(a1, b1);
					_mm256_store_ps(&data[i + 8], a1);
				}

				for (; i < n; ++i)
					data[i] += m.data[i];
			}


		__attribute__((always_inline, hot, flatten))
			inline void sub(const Matrix &m) {
				if (rows != m.rows || cols != m.cols) 
					throw std::invalid_argument("Matrix sizes do not match");

				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&m.data[0], _MM_HINT_T0);
				for (; i + 15 < n; i += 16) {
					__m256 a0 = _mm256_load_ps(&data[i]);
					__m256 b0 = _mm256_load_ps(&m.data[i]);
					a0 = _mm256_sub_ps(a0, b0);
					_mm256_store_ps(&data[i], a0);

					__m256 a1 = _mm256_load_ps(&data[i + 8]);
					__m256 b1 = _mm256_load_ps(&m.data[i + 8]);
					a1 = _mm256_sub_ps(a1, b1);
					_mm256_store_ps(&data[i + 8], a1);
				}
				for (; i < size(); ++i) {
					data[i] -= m.data[i];
				}
			}


		__attribute__((always_inline, hot, flatten))
			inline void scl(float a) {
				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
				__m256 scalar = _mm256_set1_ps(a);

				for (; i + 15 < n; i += 16) {
					__m256 v0 = _mm256_load_ps(&data[i]);
					v0 = _mm256_mul_ps(v0, scalar);
					_mm256_store_ps(&data[i], v0);

					__m256 v1 = _mm256_load_ps(&data[i + 8]);
					v1 = _mm256_mul_ps(v1, scalar);
					_mm256_store_ps(&data[i + 8], v1);
				}

				for (; i < n; ++i)
					data[i] *= a;
			}


		__attribute__((always_inline, hot, flatten))
			inline Matrix<float> mul_mat(const Matrix<float>& mat) const {
				if (cols != mat.rows) {
					throw std::invalid_argument("Matrix dimensions do not match for multiplication");
				}

				constexpr size_t BLOCK_SIZE = 16; 
				constexpr size_t UNROLL = 16; 

				Matrix<K> result(rows, mat.cols);
				Matrix<K> mat_transposed(mat.cols, mat.rows);

				for (size_t i = 0; i < mat.rows; ++i) {
					for (size_t j = 0; j < mat.cols; ++j) {
						mat_transposed(j, i) = mat(i, j);
					}
				}
#pragma omp parallel for schedule(dynamic)
				for (size_t ii = 0; ii < rows; ii += BLOCK_SIZE) {
					for (size_t jj = 0; jj < mat.cols; jj += BLOCK_SIZE) {
						size_t i_end = std::min(ii + BLOCK_SIZE, rows);
						size_t j_end = std::min(jj + BLOCK_SIZE, mat.cols);

						for (size_t i = ii; i < i_end; ++i) {
							for (size_t j = jj; j + (UNROLL-1) < j_end; j += UNROLL) {
								__m256 sum0 = _mm256_setzero_ps();
								__m256 sum1 = _mm256_setzero_ps();
								__m256 sum2 = _mm256_setzero_ps();
								__m256 sum3 = _mm256_setzero_ps();

								const float* a_ptr = &data[i * cols];
								const float* b_ptr0 = &mat_transposed.data[j * mat.rows];
								const float* b_ptr1 = b_ptr0 + mat.rows;
								const float* b_ptr2 = b_ptr1 + mat.rows;
								const float* b_ptr3 = b_ptr2 + mat.rows;

								size_t k = 0;
								for (; k + 7 < cols; k += 8) {
									__m256 a = _mm256_loadu_ps(a_ptr + k);

									__m256 b0 = _mm256_set1_ps(b_ptr0[k]);
									__m256 b1 = _mm256_set1_ps(b_ptr1[k]);
									__m256 b2 = _mm256_set1_ps(b_ptr2[k]);
									__m256 b3 = _mm256_set1_ps(b_ptr3[k]);

									sum0 = _mm256_fmadd_ps(a, b0, sum0);
									sum1 = _mm256_fmadd_ps(a, b1, sum1);
									sum2 = _mm256_fmadd_ps(a, b2, sum2);
									sum3 = _mm256_fmadd_ps(a, b3, sum3);
								}

								float sum_array0[8], sum_array1[8], sum_array2[8], sum_array3[8];
								_mm256_storeu_ps(sum_array0, sum0);
								_mm256_storeu_ps(sum_array1, sum1);
								_mm256_storeu_ps(sum_array2, sum2);
								_mm256_storeu_ps(sum_array3, sum3);

								float total0 = sum_array0[0] + sum_array0[1] + sum_array0[2] + sum_array0[3] +
									sum_array0[4] + sum_array0[5] + sum_array0[6] + sum_array0[7];
								float total1 = sum_array1[0] + sum_array1[1] + sum_array1[2] + sum_array1[3] +
									sum_array1[4] + sum_array1[5] + sum_array1[6] + sum_array1[7];
								float total2 = sum_array2[0] + sum_array2[1] + sum_array2[2] + sum_array2[3] +
									sum_array2[4] + sum_array2[5] + sum_array2[6] + sum_array2[7];
								float total3 = sum_array3[0] + sum_array3[1] + sum_array3[2] + sum_array3[3] +
									sum_array3[4] + sum_array3[5] + sum_array3[6] + sum_array3[7];

								for (; k < cols; ++k) {
									total0 += a_ptr[k] * b_ptr0[k];
									total1 += a_ptr[k] * b_ptr1[k];
									total2 += a_ptr[k] * b_ptr2[k];
									total3 += a_ptr[k] * b_ptr3[k];
								}

								result(i, j)   = total0;
								result(i, j+1) = total1;
								result(i, j+2) = total2;
								result(i, j+3) = total3;
							}

							for (size_t j = j_end - (j_end-jj)%UNROLL; j < j_end; ++j) {
								float sum = 0.0f;
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
};
