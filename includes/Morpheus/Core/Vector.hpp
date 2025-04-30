#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "../SIMD/SIMD.hpp" 
#include "../SIMD/CPU_id.hpp" 
#include "../SIMD/Allocator.hpp"
#include "../MathUtils/MathsUtils.hpp"

namespace morpheus {
	template<typename K>
		class Vector {
			public:
				aligned_vector<K> data;

				Vector(const std::vector<K>& vec) : data(vec.begin(), vec.end()) {}
				K& operator[](size_t i) { return data[i]; }
				const K& operator[](size_t i) const { return data[i]; }
				Vector(size_t n) : data(n, K()) {}
				Vector(std::initializer_list<K> init) : data(init) {}

				Vector(size_t n, K value) : data(n, value) {}
				auto begin() { return data.begin(); }
				auto end()   { return data.end(); }
				auto begin() const { return data.begin(); }
				auto end()   const { return data.end(); }
				size_t size() const {
					return data.size();
				}
				
				__attribute__((always_inline, hot, flatten))
					Vector<K> operator-(const Vector<K>& other) const {
						assert(data.size() == other.data.size());
						Vector<K> result(data.size());
						for (size_t i = 0; i < data.size(); ++i)
							result[i] = data[i] - other[i];
						return result;
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
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						size_t n = size();
						size_t i = 0;

						_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);

						for (; i + 15 < n; i += 16) {
							reg a0 = Simd::load(&data[i]);
							reg b0 = Simd::load(&v.data[i]);
							a0 = Simd::add(a0, b0);
							Simd::store(&data[i], a0);

							reg a1 = Simd::load(&data[i + simd_width]);
							reg b1 = Simd::load(&v.data[i + simd_width]);
							a1 = Simd::add(a1, b1);
							Simd::store(&data[i + simd_width], a1);
						}

						for (; i < n; ++i)
							data[i] += v.data[i];
					}

				__attribute__((always_inline, hot, flatten))
					inline void sub(const Vector &v) {
						if (v.size() != size()) 
							throw std::invalid_argument("Vector sizes do not match");
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						size_t n = size();
						size_t i = 0;

						_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
						for (; i + 15 < n; i += 16) {
							reg a0 = Simd::load(&data[i]);
							reg b0 = Simd::load(&v.data[i]);
							a0 = Simd::sub(a0, b0);
							Simd::store(&data[i], a0);

							reg a1 = Simd::load(&data[i + simd_width]);
							reg b1 = Simd::load(&v.data[i + simd_width]);
							a1 = Simd::sub(a1, b1);
							Simd::store(&data[i + simd_width], a1);
						}

						for (; i < n; ++i)
							data[i] -= v.data[i];
					}

				__attribute__((always_inline, hot, flatten))
					inline void scl(float a) {
						size_t n = size();
						size_t i = 0;
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
						reg scalar = Simd::set1(a);
						float* __restrict out = &data[0];

						for (; i + 15 < n; i += 16) {
							reg v0 = Simd::load(out + i);
							v0 = Simd::mul(v0, scalar);
							Simd::store(out + i, v0);

							reg v1 = Simd::load(out + i + simd_width);
							v1 = Simd::mul(v1, scalar);
							Simd::store(out + i + simd_width, v1);
						}

						for (; i < n; ++i)
							out[i] *= a;
					}

				__attribute__((always_inline, hot, flatten))
					static inline Vector<float> linear_combination(const std::vector<Vector<float>> &u, 
							const std::vector<float> &coefs) {
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;

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
						constexpr size_t W = simd_width;
						for (; i + W - 1 < n; i += W) {
							reg acc = Simd::zero();
							for (size_t j = 0; j < u.size(); ++j) {
								reg v = Simd::load(&u[j].data[i]);
								reg c = Simd::set1(coefs[j]);
								acc = Simd::fmadd(v, c, acc); 
							}
							Simd::store(&result.data[i], acc);
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
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;

						const size_t n = a.size();
						Vector<float> result(n);

						const reg vt = Simd::set1(t);
						const reg vt1 = Simd::sub(Simd::set1(1.0f), vt);

						size_t i = 0;
						_mm_prefetch((const char *)&a.data[0], _MM_HINT_T0);
						for (; i + 7 < n; i += simd_width) {
							reg va = Simd::load(&a.data[i]);
							reg vb = Simd::load(&b.data[i]);

							reg r = Simd::fmadd(vb, vt, Simd::mul(va, vt1));

							Simd::store_stream(&result.data[i], r);	
						}

						for (; i < n; ++i)
							result.data[i] = (1.0f - t) * a.data[i] + t * b.data[i];

						return result;
					}

				__attribute__((always_inline, hot, flatten))
					inline float dot(const Vector<float>& v) const {
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						if (v.size() != size())
							throw std::invalid_argument("Vector sizes do not match");
						const size_t n = size();
						size_t i = 0;
						reg acc = Simd::zero();

						const float* __restrict a_ptr = &data[0];
						const float* __restrict b_ptr = &v.data[0];
						_mm_prefetch((const char *)&v.data[0], _MM_HINT_T0);
						for (; i + 7 < n; i += simd_width) {
							reg a = Simd::load(a_ptr + i);
							reg b = Simd::load(b_ptr + i);
							acc = Simd::fmadd(a, b, acc);
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
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						reg acc = Simd::zero();
						reg sign_mask = Simd::set1(-0.0f);
						const float* __restrict v_ptr = &data[0];
						_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
						for (; i + 7 < n; i += simd_width) {
							reg v = Simd::load(v_ptr + i);
							reg abs_v = Simd::andnot(sign_mask, v);
							acc = Simd::add(acc, abs_v);
						}

						float result = detail::reduce_sum(acc);

						for (; i < n; ++i)
							result += MathsUtils::_fabs(v_ptr[i]);

						return result;
					}

				__attribute__((always_inline, hot, flatten))
					inline float norm_2() const {
						size_t n = size();
						size_t i = 0;
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						reg acc = Simd::zero();
						const float* __restrict v_ptr = &data[0];
						_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
						for (; i + 7 < n; i += simd_width) {
							reg v = Simd::load(v_ptr + i);
							acc = Simd::fmadd(v, v, acc);
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
						using Simd = simd::SimdTraits<K, DefaultISA>;
						using reg = typename Simd::reg;
						const size_t simd_width = Simd::width;
						reg max_v = Simd::zero();
						reg sign_mask = Simd::set1(-0.0f);
						const float* __restrict v_ptr = &data[0];
						_mm_prefetch((const char *)&data[0], _MM_HINT_T0);
						for (; i + 7 < n; i += simd_width) {
							reg v = Simd::load(v_ptr + i);
							reg abs_v = Simd::andnot(sign_mask, v);
							max_v = Simd::max(max_v, abs_v);
						}

						float result = detail::reduce_sum(max_v);

						for (; i < n; ++i)
							result = MathsUtils::_max(result, MathsUtils::_fabs(v_ptr[i]));

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
}
