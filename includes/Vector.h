#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include "SIMD.h"
#include <mutex>



template<typename K>
class Vector {
	public:
		aligned_vector<K> data;

		Vector(size_t n) : data(n, K()) {}
		Vector(std::initializer_list<K> init) : data(init) {}

		size_t size() const {
			return data.size();
		}

		void print() const {
			std::cout << "Vector size: " << size() << "\n";
			for (float f : data)
				std::cout << "[" << f << "]\n";
		}

		__attribute__((always_inline, hot, flatten))
			inline void add(const Vector &v) {
				if (v.size() != size())
					throw std::invalid_argument("Vector sizes do not match");

				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);

				for (; i + 15 < n; i += 16) {
					__m256 a0 = _mm256_load_ps(&data[i]);
					__m256 b0 = _mm256_load_ps(&v.data[i]);
					a0 = _mm256_add_ps(a0, b0);
					_mm256_store_ps(&data[i], a0);

					__m256 a1 = _mm256_load_ps(&data[i + 8]);
					__m256 b1 = _mm256_load_ps(&v.data[i + 8]);
					a1 = _mm256_add_ps(a1, b1);
					_mm256_store_ps(&data[i + 8], a1);
				}

				for (; i < n; ++i)
					data[i] += v.data[i];
			}


		__attribute__((always_inline, hot, flatten))
			inline void sub(const Vector &v) {
				if (v.size() != size()) 
					throw std::invalid_argument("Vector sizes do not match");
				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
				for (; i + 15 < n; i += 16) {
					__m256 a0 = _mm256_load_ps(&data[i]);
					__m256 b0 = _mm256_load_ps(&v.data[i]);
					a0 = _mm256_sub_ps(a0, b0);
					_mm256_store_ps(&data[i], a0);

					__m256 a1 = _mm256_load_ps(&data[i + 8]);
					__m256 b1 = _mm256_load_ps(&v.data[i + 8]);
					a1 = _mm256_sub_ps(a1, b1);
					_mm256_store_ps(&data[i + 8], a1);
				}

				for (; i < n; ++i)
					data[i] -= v.data[i];
			}



		__attribute__((always_inline, hot, flatten))
			inline void scl(float a) {
				size_t n = size();
				size_t i = 0;

				_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
				__m256 scalar = _mm256_set1_ps(a);
				float* __restrict out = &data[0];

				for (; i + 15 < n; i += 16) {
					__m256 v0 = _mm256_load_ps(out + i);
					v0 = _mm256_mul_ps(v0, scalar);
					_mm256_store_ps(out + i, v0);

					__m256 v1 = _mm256_load_ps(out + i + 8);
					v1 = _mm256_mul_ps(v1, scalar);
					_mm256_store_ps(out + i + 8, v1);
				}

				for (; i < n; ++i)
					out[i] *= a;
			}



		__attribute__((always_inline, hot, flatten))
			static inline Vector<float> linear_combination(const std::vector<Vector<float>> &u, 
					const std::vector<float> &coefs) {
				if (u.size() != coefs.size())
					throw std::invalid_argument("Mismatched number of vectors and coefficients");
				if (u.empty())
					return Vector<float>(0);

				const size_t n = u[0].size();
				for (const auto& v : u)
					if (v.size() != n)
						throw std::invalid_argument("Vector sizes do not match");

				Vector<float> result(n);

				size_t i = 0;
				constexpr size_t W = 8;
				for (; i + W - 1 < n; i += W) {
					__m256 acc = _mm256_setzero_ps();
					for (size_t j = 0; j < u.size(); ++j) {
						__m256 v = _mm256_loadu_ps(&u[j].data[i]);
						__m256 c = _mm256_set1_ps(coefs[j]);
						acc = _mm256_fmadd_ps(v, c, acc); 
					}
					_mm256_stream_ps(&result.data[i], acc);
				}

				for (; i < n; ++i) {
					float acc = 0.f;
					for (size_t j = 0; j < u.size(); ++j)
						acc += coefs[j] * u[j].data[i];
					result.data[i] = acc;
				}

				return result;
			}

		__attribute__((always_inline, hot, flatten))
			static inline Vector<float> lerp(const Vector<float>& a, const Vector<float>& b, float t) {
				if (a.size() != b.size())
					throw std::invalid_argument("Vector sizes do not match");

				const size_t n = a.size();
				Vector<float> result(n);

				const __m256 vt = _mm256_set1_ps(t);
				const __m256 vt1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), vt);

				size_t i = 0;
				_mm_prefetch((const char *)&a.data[0], _MM_HINT_T0);
				for (; i + 7 < n; i += 8) {
					__m256 va = _mm256_load_ps(&a.data[i]);
					__m256 vb = _mm256_load_ps(&b.data[i]);

					__m256 r = _mm256_fmadd_ps(vb, vt, _mm256_mul_ps(va, vt1));

					_mm256_stream_ps(&result.data[i], r);	
				}

				for (; i < n; ++i)
					result.data[i] = (1.0f - t) * a.data[i] + t * b.data[i];

				return result;
			}


		__attribute__((always_inline, hot, flatten))
			inline float dot(const Vector<float>& v) const {
				if (v.size() != size())
					throw std::invalid_argument("Vector sizes do not match");

				const size_t n = size();
				size_t i = 0;
				__m256 acc = _mm256_setzero_ps();

				const float* __restrict a_ptr = &data[0];
				const float* __restrict b_ptr = &v.data[0];
				_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
				for (; i + 7 < n; i += 8) {
					__m256 a = _mm256_load_ps(a_ptr + i);
					__m256 b = _mm256_load_ps(b_ptr + i);
					acc = _mm256_fmadd_ps(a, b, acc);
				}

				float result = detail::reduce_sum(acc);

				for (; i < n; ++i)
					result += a_ptr[i] * b_ptr[i];

				return result;
			}


		__attribute__((always_inline, hot, flatten))
			inline float norm_1() const {
				size_t n = size();
				size_t i = 0;

				__m256 acc = _mm256_setzero_ps();
				__m256 sign_mask = _mm256_set1_ps(-0.0f);
				const float* __restrict v_ptr = &data[0];
				_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
				for (; i + 7 < n; i += 8) {
					__m256 v = _mm256_load_ps(v_ptr + i);
					__m256 abs_v = _mm256_andnot_ps(sign_mask, v);
					acc = _mm256_add_ps(acc, abs_v);
				}

				float result = detail::reduce_sum(acc);

				for (; i < n; ++i)
					result += std::fabs(v_ptr[i]);

				return result;
			}


		__attribute__((always_inline, hot, flatten))
			inline float norm_2() const {
				size_t n = size();
				size_t i = 0;

				__m256 acc = _mm256_setzero_ps();
				const float* __restrict v_ptr = &data[0];
				_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
				for (; i + 7 < n; i += 8) {
					__m256 v = _mm256_load_ps(v_ptr + i);
					acc = _mm256_fmadd_ps(v, v, acc);
				}

				float result = detail::reduce_sum(acc);

				for (; i < n; ++i)
					result += v_ptr[i] * v_ptr[i];

				return std::sqrt(result);
			}



		__attribute__((always_inline, hot, flatten))
			inline float norm_inf() const {
				size_t n = size();
				size_t i = 0;

				__m256 max_v = _mm256_setzero_ps();
				__m256 sign_mask = _mm256_set1_ps(-0.0f);
				const float* __restrict v_ptr = &data[0];
				_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
				for (; i + 7 < n; i += 8) {
					__m256 v = _mm256_load_ps(v_ptr + i);
					__m256 abs_v = _mm256_andnot_ps(sign_mask, v);
					max_v = _mm256_max_ps(max_v, abs_v);
				}

				float result = detail::reduce_sum(max_v);

				for (; i < n; ++i)
					result = std::max(result, std::fabs(v_ptr[i]));

				return result;
			}
		

		__attribute__((always_inline, hot, flatten))
			static inline float angle_cos(const Vector<float>& u, const Vector<float>& v) {
				if (u.size() != v.size())
					throw std::invalid_argument("Vector sizes do not match");

				const float dot = u.dot(v);
				const float norm_u = u.norm_2();
				const float norm_v = v.norm_2();

				return dot / (norm_u * norm_v);
			}
			
		__attribute__((always_inline, hot, flatten))
			static inline Vector<float> cross_product(const Vector<float>& u, const Vector<float>& v) {
				if (u.size() != 3 || v.size() != 3)
					throw std::invalid_argument("Cross product is only defined for 3D vectors.");

				Vector<float> r(3);

				__m128 uxy = _mm_set_ps(0.0f, u.data[0], u.data[2], u.data[1]); 
				__m128 vxy = _mm_set_ps(0.0f, v.data[0], v.data[2], v.data[1]); 

				r.data[0] = std::fma(u.data[1], v.data[2], -u.data[2] * v.data[1]);
				r.data[1] = std::fma(u.data[2], v.data[0], -u.data[0] * v.data[2]);
				r.data[2] = std::fma(u.data[0], v.data[1], -u.data[1] * v.data[0]);

				return r;
			}


};



