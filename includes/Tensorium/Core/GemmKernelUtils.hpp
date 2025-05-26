#pragma once

#include <immintrin.h>

template<typename K> struct AVX;
template<> struct AVX<float> {
	using reg = __m256;
	static inline reg load(const float* p) { return _mm256_loadu_ps(p); }
	static inline void store(float* ptr, reg x)	{ _mm256_storeu_ps(ptr, x); }
	static inline reg broadcast(const float* p) { return _mm256_broadcast_ss(p); }
	static inline reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b, c); }
	static inline reg setzero() { return _mm256_setzero_ps(); }
};

template<> struct AVX<double> {
	using reg = __m256d;
	static inline reg load(const double* p) { return _mm256_loadu_pd(p); }
	static inline void store(const double *ptr, reg x) { _mm256_storeu_pd(ptr, x); }
	static inline reg broadcast(const double* p) { return _mm256_broadcast_sd(p); }
	static inline reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b, c); }
	static inline reg setzero() { return _mm256_setzero_pd(); }
};
