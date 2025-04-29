#pragma once
#include <immintrin.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cpuid.h>
#include <memory>
#include <stdexcept>
#include <complex>

struct sse_t    { static constexpr size_t width = 4;  using reg = __m128;  static constexpr size_t alignment = 16; };
struct avx2_t   { static constexpr size_t width = 8;  using reg = __m256;  static constexpr size_t alignment = 32; };
struct avx512_t { static constexpr size_t width = 16; using reg = __m512;  static constexpr size_t alignment = 64; };
#ifdef __AVX512F__
using DefaultISA = avx512_t;
#define ALIGN 64
#define SIMD_WIDTH 8
#define UNROLL 256
#elif defined(__AVX2__)
using DefaultISA = avx2_t;
#define ALIGN 32
#define SIMD_WIDTH 8
#define UNROLL 128
#else
using DefaultISA = sse_t;
#define ALIGN 16
#define SIMD_WIDTH 4
#define UNROLL 64
#endif

namespace morpheus {

	struct avx2_t {
		static constexpr size_t width = SIMD_WIDTH;
		using reg = __m256;
		static constexpr size_t alignment = ALIGN;
	};
}

inline bool supports_avx512() {
	int regs[4];
	__cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
	return (regs[1] & (1 << 16));
}

inline bool supports_avx2() {
	int regs[4];
	__cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
	return (regs[1] & (1 << 5));
}

inline bool supports_sse() {
	int regs[4];
	__cpuid_count(1, 0, regs[0], regs[1], regs[2], regs[3]);
	return (regs[3] & (1 << 25));
}




#include <iostream>
template<typename F>
	void dispatch_simd(F&& f) {
		if (supports_avx512()) {
			std::cout << "[dispatch] Detected AVX512\n";
			f(avx512_t{});
		} else if (supports_avx2()) {
			std::cout << "[dispatch] Detected AVX2\n";
			f(avx2_t{});
		} else if (supports_sse()) {
			std::cout << "[dispatch] Detected SSE\n";
			f(sse_t{});
		} else {
			throw std::runtime_error("No supported SIMD ISA (SSE/AVX2/AVX512).");
		}
	}

inline __m256 extractf32x8_ps_fallback(__m512 v, int imm8) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    if (imm8 == 0)
        return _mm256_load_ps(&tmp[0]);   
    else
        return _mm256_load_ps(&tmp[8]); 
}


/*
 * REAL NUMBERS
 */

namespace detail {
	__attribute__((always_inline, hot, flatten))
		inline float reduce_sum(__m256 acc) {
			__m128 low  = _mm256_castps256_ps128(acc);
			__m128 high = _mm256_extractf128_ps(acc, 1);
			__m128 sum = _mm_add_ps(low, high);
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			return _mm_cvtss_f32(sum);
		}
	__attribute__((always_inline, hot, flatten))
		inline double reduce_sum(__m256d acc) {
			__m128d low  = _mm256_castpd256_pd128(acc);
			__m128d high = _mm256_extractf128_pd(acc, 1);
			__m128d sum = _mm_add_pd(low, high);
			double r[2];
			_mm_store_pd(r, sum);
			return r[0] + r[1];
		}
	__attribute__((always_inline, hot, flatten))
		inline uint64_t reduce_sum(__m256i acc) {
			__m128i low  = _mm256_castsi256_si128(acc);
			__m128i high = _mm256_extractf128_si256(acc, 1);
			__m128i sum = _mm_add_epi64(low, high);
			uint64_t r[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(r), sum);
			return r[0] + r[1];
		}
	__attribute__((always_inline, hot, flatten))
		inline float reduce_sum(__m512 acc) {
			__m256 low  = _mm512_castps512_ps256(acc);
#if defined(__AVX512DQ__)
			__m256 high = _mm512_extractf32x8_ps(acc, 1);
#else
			__m256 high = extractf32x8_ps_fallback(acc, 1);
#endif
			__m256 sum = _mm256_add_ps(low, high);
			return reduce_sum(sum);
		}
	__attribute__((always_inline, hot, flatten))
		inline double reduce_sum(__m512d acc) {
			__m256d low  = _mm512_castpd512_pd256(acc);
			__m256d high = _mm512_extractf64x4_pd(acc, 1);
			__m256d sum = _mm256_add_pd(low, high);
			return reduce_sum(sum);
		}
	__attribute__((always_inline, hot, flatten))
	inline uint64_t reduce_sum(__m512i acc) {
		__m256i low  = _mm512_castsi512_si256(acc);
		__m256i high = _mm512_extracti64x4_epi64(acc, 1);  
		__m256i sum = _mm256_add_epi64(low, high);
		return reduce_sum(sum);  
	}
	__attribute__((always_inline, hot, flatten))
		inline float reduce_sum(__m128 acc) {
			__m128 sum = _mm_hadd_ps(acc, acc);
			sum = _mm_hadd_ps(sum, sum);
			return _mm_cvtss_f32(sum);
		}
}


static inline __m512 andnot_fallback(__m512 a, __m512 b) {
    __m512i a_bits = _mm512_castps_si512(a);
    __m512i not_a_bits = _mm512_xor_si512(a_bits, _mm512_set1_epi32(-1)); 
    __m512i b_bits = _mm512_castps_si512(b);
    __m512i result_bits = _mm512_and_si512(not_a_bits, b_bits);
    return _mm512_castsi512_ps(result_bits);
}


namespace simd {


	template<typename T, typename ISA = DefaultISA>
		struct SimdTraits;
	template<>
		struct SimdTraits<float, sse_t> {
			using reg = __m128;
			static constexpr size_t width = 4;
			static inline reg set1(float x)				{ return _mm_set1_ps(x); }
			static inline reg set(float a, float b, float c, float d) {
				return _mm_set_ps(a, b, c, d);
			}
			static inline reg set4(
					float a0, float a1, float a2, float a3
					) {
				return _mm_set_ps(a3, a2, a1, a0);
			}
			static inline float extract(reg x, size_t index) {
				alignas(16) float values[4];
				_mm_storeu_ps(values, x);
				return values[index];
			}
			static inline void stream(float* ptr, reg x) { _mm_stream_ps(ptr, x); }
			static inline reg setzero() { return _mm_setzero_ps(); }
			static inline reg fma(reg a, reg b, reg c)   { return _mm_fmadd_ps(a, b, c); }
			static inline float horizontal_add(reg v)    { 
				alignas(16) float values[4];
				_mm_store_ps(values, v);
				return values[0] + values[1] + values[2] + values[3];
			}
			static inline reg load(const float* ptr)	 { return _mm_load_ps(ptr); }
			static inline reg loadu(const float* ptr)	 { return _mm_loadu_ps(ptr); }
			static inline void store(float* ptr, reg x)	 { _mm_store_ps(ptr, x); }
			static inline void storeu(float* ptr, reg x) { _mm_storeu_ps(ptr, x); }
			static inline reg zero()					 { return _mm_setzero_ps(); }
			static inline reg fmadd(reg a, reg b, reg c) { return _mm_fmadd_ps(a, b, c); }
			static inline reg add(reg a, reg b)			 { return _mm_add_ps(a, b); }
			static inline reg mul(reg a, reg b)			 { return _mm_mul_ps(a, b); }
			static inline reg sub(reg a, reg b)			 { return _mm_sub_ps(a, b); }
			static inline reg andnot(reg a, reg b)		 { return _mm_andnot_ps(a, b); }
			static inline void store_stream(float* ptr, reg x) { _mm_stream_ps(ptr, x); }
			static inline reg max(reg a, reg b)			 { return _mm_max_ps(a, b); }
			static inline reg min(reg a, reg b)			 { return _mm_min_ps(a, b); }
		};
	
	template<>
		struct SimdTraits<double, sse_t> {
			using reg = __m128d;
			static constexpr size_t width = 2;
			static inline reg set1(double x)            { return _mm_set1_pd(x); } 
			static inline reg set(double a, double b)	{ return _mm_set_pd(b, a); }
			static inline double extract(reg x, size_t index) {
				alignas(16) double values[2];
				_mm_storeu_pd(values, x);
				return values[index];
			}
			static inline void stream(double* ptr, reg x)	{ _mm_stream_pd(ptr, x); }
			static inline reg setzero()						{ return _mm_setzero_pd(); }
			static inline double horizontal_add(reg v)		{ 
				alignas(16) double values[2];
				_mm_store_pd(values, v);
				return values[0] + values[1];
			}
			static inline reg load(const double* ptr)		{ return _mm_load_pd(ptr); }
			static inline reg loadu(const double* ptr)		{ return _mm_loadu_pd(ptr); }
			static inline void store(double* ptr, reg x)	{ _mm_store_pd(ptr, x); }
			static inline void storeu(double* ptr, reg x)	{ _mm_storeu_pd(ptr, x); }
			static inline reg zero()						{ return _mm_setzero_pd(); }
			static inline reg fmadd(reg a, reg b, reg c)	{ return _mm_fmadd_pd(a, b, c); }
			static inline reg add(reg a, reg b)				{ return _mm_add_pd(a, b); }
			static inline reg mul(reg a, reg b)				{ return _mm_mul_pd(a, b); }
			static inline reg sub(reg a, reg b)				{ return _mm_sub_pd(a, b); }
			static inline reg andnot(reg a, reg b)			{ return _mm_andnot_pd(a, b); }
			static inline void store_stream(double* ptr, reg x) { _mm_stream_pd(ptr, x); }
			static inline reg max(reg a, reg b)				{ return _mm_max_pd(a, b); }
			static inline reg min(reg a, reg b)				{ return _mm_min_pd(a, b); }
		};

	template<>
		struct SimdTraits<size_t, sse_t> {
			using reg = __m128i;
			static constexpr size_t width = 2;
			static inline reg set1(uint64_t x)				{ return _mm_set1_epi64x(x); }
			static inline reg set(uint64_t a, uint64_t b)	{ return _mm_set_epi64x(b, a); }
			static inline uint64_t extract(reg x, size_t index) {
				alignas(16) uint64_t values[2];
				_mm_storeu_si128(reinterpret_cast<__m128i*>(values), x);
				return values[index];
			}
			static inline void stream(uint64_t* ptr, reg x) { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), x); }
			static inline reg setzero()						{ return _mm_setzero_si128(); }
			static inline float horizontal_add(reg v)		{ 
				alignas(16) uint64_t values[2];
				_mm_store_si128(reinterpret_cast<__m128i*>(values), v);
				return values[0] + values[1];
			}
			static inline reg load(const uint64_t* ptr)		{ return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr)); }
			static inline reg loadu(const uint64_t* ptr)	{ return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)); }
			static inline void store(uint64_t* ptr, reg x)	{ _mm_store_si128(reinterpret_cast<__m128i*>(ptr), x); }
			static inline void storeu(uint64_t* ptr, reg x) { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), x); }
			static inline reg zero()						{ return _mm_setzero_si128(); }
			static inline reg fmadd(reg a, reg b, reg c)	{ return _mm_add_epi64(mul(a, b), c); }
			static inline reg mul(reg a, reg b) {
				alignas(16) size_t lhs[2], rhs[2], out[2];
				_mm_store_si128((__m128i*)lhs, a);
				_mm_store_si128((__m128i*)rhs, b);
				for (size_t i = 0; i < 2; ++i)
					out[i] = lhs[i] * rhs[i];
				return _mm_load_si128((__m128i*)out);
			}
			static inline reg add(reg a, reg b)				{ return _mm_add_epi64(a, b); }
			static inline reg sub(reg a, reg b)				{ return _mm_sub_epi64(a, b); }
			static inline reg andnot(reg a, reg b)			{ return _mm_andnot_si128(a, b); }
			static inline void store_stream(uint64_t* ptr, reg x) { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), x); }
			static inline reg set_epi64(int64_t a, int64_t b) { return _mm_set_epi64x(b, a); }
			static inline reg max(reg a, reg b)				{ return _mm_max_epi64(a, b); }
			static inline reg min(reg a, reg b)				{ return _mm_min_epi64(a, b); }
		};

	template<>
		struct SimdTraits<float, avx2_t> {
			using reg = __m256;
			static constexpr size_t width = 8;
			static inline reg set1(float x)				{ return _mm256_set1_ps(x); }
			static inline reg set(float a, float b, float c, float d) {
				return _mm256_set_ps(a, b, c, d, a, b, c, d);
			}
			static inline reg set8(
					float a0, float a1, float a2, float a3,
					float a4, float a5, float a6, float a7
					) {
				return _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0);
			}
			static inline float extract(reg x, size_t index) {
				alignas(32) float values[8];
				_mm256_storeu_ps(values, x);
				return values[index];
			}
			static inline void stream(float* ptr, reg x)	{ _mm256_stream_ps(ptr, x); }
			static inline reg setzero()						{ return _mm256_setzero_ps(); }
			static inline reg fma(reg a, reg b, reg c)		{ return _mm256_fmadd_ps(a, b, c); }
			static inline float horizontal_add(reg v)		{ return detail::reduce_sum(v); }
			static inline reg load(const float* ptr)		{ return _mm256_load_ps(ptr); }
			static inline reg loadu(const float* ptr)		{ return _mm256_loadu_ps(ptr); }
			static inline void store(float* ptr, reg x)		{ _mm256_store_ps(ptr, x); }
			static inline void storeu(float* ptr, reg x)	{ _mm256_storeu_ps(ptr, x); }
			static inline reg zero()						{ return _mm256_setzero_ps(); }
			static inline reg fmadd(reg a, reg b, reg c)	{ return _mm256_fmadd_ps(a, b, c); }
			static inline reg add(reg a, reg b)				{ return _mm256_add_ps(a, b); }
			static inline reg mul(reg a, reg b)				{ return _mm256_mul_ps(a, b); }
			static inline reg sub(reg a, reg b)				{ return _mm256_sub_ps(a, b); }
			static inline reg andnot(reg a, reg b)			{ return _mm256_andnot_ps(a, b); }
			static inline void store_stream(float* ptr, reg x) { _mm256_stream_ps(ptr, x); }
			static inline reg max(reg a, reg b)				{ return _mm256_max_ps(a, b); }
			static inline reg min(reg a, reg b)				{ return _mm256_min_ps(a, b); }

		};

	template<>
		struct SimdTraits<double, avx2_t> {
			using reg = __m256d;
			static constexpr size_t width = 4;
			static inline reg set1(double x)            { return _mm256_set1_pd(x); }
			static inline reg set(double a, double b, double c, double d) {
				return _mm256_set_pd(a, b, c, d);
			}
			static inline double extract(reg x, size_t index) {
				alignas(32) double values[4];
				_mm256_storeu_pd(values, x);
				return values[index];
			}
			static inline void stream(double* ptr, reg x) { _mm256_stream_pd(ptr, x); }
			static inline reg setzero()					{ return _mm256_setzero_pd(); }
			static inline float horizontal_add(reg v)	{ return detail::reduce_sum(v); }
			static inline reg load(const double* ptr)   { return _mm256_load_pd(ptr); }
			static inline reg loadu(const double* ptr)  { return _mm256_loadu_pd(ptr); }
			static inline void store(double* ptr, reg x){ _mm256_store_pd(ptr, x); }
			static inline void storeu(double* ptr, reg x){ _mm256_storeu_pd(ptr, x); }
			static inline reg zero()                    { return _mm256_setzero_pd(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm256_fmadd_pd(a, b, c); }
			static inline reg add(reg a, reg b)         { return _mm256_add_pd(a, b); }
			static inline reg mul(reg a, reg b)         { return _mm256_mul_pd(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm256_sub_pd(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm256_andnot_pd(a, b); }
			static inline void store_stream(double* ptr, reg x) { _mm256_stream_pd(ptr, x); }
			static inline reg max(reg a, reg b)		{ return _mm256_max_pd(a, b); }
		};

	template<>
		struct SimdTraits<size_t, avx2_t> {
			using reg = __m256i;
			static constexpr size_t width = 4;
			static inline reg set1(uint64_t x)			{ return _mm256_set1_epi64x(x); }
			static inline reg set(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
				return _mm256_set_epi64x(a, b, c, d);
			}
			static inline uint64_t extract(reg x, size_t index) {
				alignas(32) uint64_t values[4];
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(values), x);
				return values[index];
			}
			static inline void stream(uint64_t* ptr, reg x) { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), x); }
			static inline reg setzero()					{ return _mm256_setzero_si256(); }
			static inline float horizontal_add(reg v)	{ return detail::reduce_sum(v); }
			static inline reg load(const uint64_t* ptr)	{ return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr)); }
			static inline reg loadu(const uint64_t* ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }
			static inline void store(uint64_t* ptr, reg x) { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), x); }
			static inline void storeu(uint64_t* ptr, reg x) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), x); }
			static inline reg mul(reg a, reg b) {
				alignas(32) size_t lhs[4], rhs[4], out[4];
				_mm256_store_si256((__m256i*)lhs, a);
				_mm256_store_si256((__m256i*)rhs, b);
				for (size_t i = 0; i < 4; ++i)
					out[i] = lhs[i] * rhs[i];
				return _mm256_load_si256((__m256i*)out);
			}
			static inline reg zero()					{ return _mm256_setzero_si256(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm256_add_epi64(mul(a, b), c); }
			static inline reg add(reg a, reg b)			{ return _mm256_add_epi64(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm256_sub_epi64(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm256_andnot_si256(a, b); }
			static inline void store_stream(uint64_t* ptr, reg x) { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), x); }
			static inline reg set_epi64(int64_t a, int64_t b, int64_t c, int64_t d) {
				return _mm256_set_epi64x(a, b, c, d);
			}
			static inline reg max(reg a, reg b)		{ return _mm256_max_epi64(a, b); }
		};


	template<>
		struct SimdTraits<float, avx512_t> {
			using reg = __m512;
			static constexpr size_t width = 16;
			static inline reg set1(float x)				{ return _mm512_set1_ps(x); }
			static inline reg set(float a, float b, float c, float d,
								  float e, float f, float g, float h,
								  float i, float j, float k, float l,
								  float m, float n, float o, float p) {
				return _mm512_set_ps(a, b, c, d, e, f, g, h,
									i, j, k, l, m, n, o, p);
			}
			static inline float extract(reg x, size_t index) {
				alignas(64) float values[16];
				_mm512_storeu_ps(values, x);
				return values[index];
			}
			static inline void stream(float* ptr, reg x)	{ _mm512_stream_ps(ptr, x); }
			static inline reg setzero()					{ return _mm512_setzero_ps(); }
			static inline float horizontal_add(reg v)	{ return detail::reduce_sum(v); }
			static inline reg load(const float* ptr)	{ return _mm512_load_ps(ptr); }
			static inline reg loadu(const float* ptr)	{ return _mm512_loadu_ps(ptr); }
			static inline void store(float* ptr, reg x)	{ _mm512_store_ps(ptr, x); }
			static inline reg loadu_stream(const float* ptr) { return _mm512_loadu_ps(ptr); }
			static inline reg zero()					{ return _mm512_setzero_ps(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm512_fmadd_ps(a, b, c); }
			static inline reg add(reg a, reg b)			{ return _mm512_add_ps(a, b); }
			static inline reg mul(reg a, reg b)			{ return _mm512_mul_ps(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm512_sub_ps(a, b); }
#if defined(__AVX512DQ__)
			static inline reg andnot(reg a, reg b)		{ return _mm512_andnot_ps(a, b); }
#else
			static inline reg andnot(reg a, reg b)		{ return andnot_fallback(a, b); }
#endif
			static inline void store_stream(float* ptr, reg x) { _mm512_stream_ps(ptr, x); }
			static inline reg max(reg a, reg b)		{ return _mm512_max_ps(a, b); }
		};

	template<>
		struct SimdTraits<double, avx512_t> {
			using reg = __m512d;
			static constexpr size_t width = 8;
			static inline reg set1(double x)            { return _mm512_set1_pd(x); }
			static inline reg set(double a, double b, double c, double d,
								  double e, double f, double g, double h) {
				return _mm512_set_pd(a, b, c, d, e, f, g, h);
			}
			static inline double extract(reg x, size_t index) {
				alignas(64) double values[8];
				_mm512_storeu_pd(values, x);
				return values[index];
			}
			static inline void stream(double* ptr, reg x) { _mm512_stream_pd(ptr, x); }
			static inline reg setzero()					{ return _mm512_setzero_pd(); }
			static inline float horizontal_add(reg v)	{ return detail::reduce_sum(v); }
			static inline reg load(const double* ptr)   { return _mm512_load_pd(ptr); }
			static inline reg loadu(const double* ptr)  { return _mm512_loadu_pd(ptr); }
			static inline void store(double* ptr, reg x){ _mm512_store_pd(ptr, x); }
			static inline reg loadu_stream(const double* ptr) { return _mm512_loadu_pd(ptr); }
			static inline reg zero()                    { return _mm512_setzero_pd(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm512_fmadd_pd(a, b, c); }
			static inline reg add(reg a, reg b)         { return _mm512_add_pd(a, b); }
			static inline reg mul(reg a, reg b)         { return _mm512_mul_pd(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm512_sub_pd(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm512_andnot_pd(a, b); }
			static inline void store_stream(double* ptr, reg x) { _mm512_stream_pd(ptr, x); }
			static inline reg max(reg a, reg b)		{ return _mm512_max_pd(a, b); }
		};

	template<>
		struct SimdTraits<size_t, avx512_t> {
			using reg = __m512i;
			static constexpr size_t width = 8;
			static inline reg set1(size_t x)			{ return _mm512_set1_epi64(x); }
			static inline reg set(size_t a, size_t b, size_t c, size_t d,
								  size_t e, size_t f, size_t g, size_t h) {
				return _mm512_set_epi64(a, b, c, d, e, f, g, h);
			}
			static inline size_t extract(reg x, size_t index) {
				alignas(64) size_t values[8];
				_mm512_storeu_si512(reinterpret_cast<__m512i*>(values), x);
				return values[index];
			}
			static inline reg setzero()					{ return _mm512_setzero_si512(); }
			static inline float horizontal_add(reg v)	{ return detail::reduce_sum(v); }
			static inline reg load(const size_t* ptr)	{ return _mm512_load_si512(reinterpret_cast<const __m512i*>(ptr)); }
			static inline reg loadu(const size_t* ptr)	{ return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)); }
			static inline void store(size_t* ptr, reg x) { _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), x); }
			static inline void storeu(size_t* ptr, reg x) { _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), x); }
			static inline reg zero()					{ return _mm512_setzero_si512(); }
			static inline reg add(reg a, reg b)			{ return _mm512_add_epi64(a, b); }
#if defined(__AVX512DQ__)
			static inline reg mul(reg a, reg b) {
				return _mm512_mullo_epi64(a, b);
			}
#else
			static inline reg mul(reg a, reg b) {
				alignas(64) uint64_t A[8], B[8], R[8];
				_mm512_store_epi64(A, a);
				_mm512_store_epi64(B, b);
				for (int i = 0; i < 8; ++i) R[i] = A[i] * B[i];
				return _mm512_load_epi64(R);
			}
#endif

			static inline reg fmadd(reg a, reg b, reg c){ return _mm512_add_epi64(mul(a, b), c); }
			static inline reg sub(reg a, reg b)			{ return _mm512_sub_epi64(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm512_andnot_si512(a, b); }
			static inline void store_stream(size_t* ptr, reg x) { _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), x); }
			static inline reg set_epi64(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f, int64_t g, int64_t h) {
				return _mm512_set_epi64(a, b, c, d, e, f, g, h);
			}
			static inline reg max(reg a, reg b)		{ return _mm512_max_epi64(a, b); }
		};

/*
 * COMPLEX NUMBERS
 */

	template<>
		struct SimdTraits<std::complex<float>, sse_t> {
			using reg = __m128;
			static constexpr size_t width = 2;
			static inline reg set(std::complex<float> a, std::complex<float> b) {
				return _mm_set_ps(b.imag(), a.real(), b.real(), a.imag());
			}
			static inline reg set1(std::complex<float> x) {
				return _mm_set_ps(x.imag(), x.real(), x.imag(), x.real());
			}
			static inline reg load(const std::complex<float>* ptr) {
				return _mm_loadu_ps(reinterpret_cast<const float*>(ptr));
			}
			static inline reg loadu(const std::complex<float>* ptr) {
				return _mm_loadu_ps(reinterpret_cast<const float*>(ptr));
			}
			static inline void store(std::complex<float>* ptr, reg x) {
				_mm_storeu_ps(reinterpret_cast<float*>(ptr), x);
			}
			static inline void storeu(std::complex<float>* ptr, reg x) {
				_mm_storeu_ps(reinterpret_cast<float*>(ptr), x);
			}
			static inline reg add(reg a, reg b)		{ return _mm_add_ps(a, b); }
			static inline reg sub(reg a, reg b)		{ return _mm_sub_ps(a, b); }
			static inline reg mul(reg a, reg b) {
				__m128 a_real = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,2,0,0));
				__m128 a_imag = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,3,1,1));
				__m128 b_real = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2,2,0,0));
				__m128 b_imag = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,3,1,1));

				__m128 real = _mm_sub_ps(_mm_mul_ps(a_real, b_real), _mm_mul_ps(a_imag, b_imag));
				__m128 imag = _mm_add_ps(_mm_mul_ps(a_real, b_imag), _mm_mul_ps(a_imag, b_real));

				return _mm_unpacklo_ps(real, imag); 
			}	
			static inline reg andnot(reg a, reg b)	{ return _mm_andnot_ps(a, b); }
			static inline reg max(reg a, reg b)		{ return _mm_max_ps(a, b); }
			static inline reg min(reg a, reg b)		{ return _mm_min_ps(a, b); }
			static inline reg setzero()				{ return _mm_setzero_ps(); }
			static inline reg fma(reg a, reg b, reg c) { return _mm_fmadd_ps(a, b, c); }
			static inline std::complex<float> horizontal_add(reg v) {
				alignas(16) float values[4];
				_mm_storeu_ps(values, v);
				return std::complex<float>(values[0] + values[2], values[1] + values[3]);
			}
			static inline void stream(std::complex<float>* ptr, reg x) {
				_mm_stream_ps(reinterpret_cast<float*>(ptr), x); 
			}
			static inline void store_stream(std::complex<float>* ptr, reg x) {
				_mm_stream_ps(reinterpret_cast<float*>(ptr), x);
			}
		};

	template<>
		struct SimdTraits<std::complex<double>, sse_t> {
			using reg = __m128d;
			static constexpr size_t width = 1;
			static inline reg set(std::complex<double> a, std::complex<double> b) {
				return _mm_set_pd(b.imag(), a.real());
			}
			static inline reg set1(std::complex<double> x) {
				return _mm_set_pd(x.imag(), x.real());
			}
			static inline reg load(const std::complex<double>* ptr) {
				return _mm_loadu_pd(reinterpret_cast<const double*>(ptr));
			}
			static inline reg loadu(const std::complex<double>* ptr) {
				return _mm_loadu_pd(reinterpret_cast<const double*>(ptr));
			}
			static inline void store(std::complex<double>* ptr, reg x) {
				_mm_storeu_pd(reinterpret_cast<double*>(ptr), x);
			}
			static inline void storeu(std::complex<double>* ptr, reg x) {
				_mm_storeu_pd(reinterpret_cast<double*>(ptr), x);
			}
			static inline reg add(reg a, reg b)		{ return _mm_add_pd(a, b); }
			static inline reg sub(reg a, reg b)		{ return _mm_sub_pd(a, b); }
			static inline reg mul(reg a, reg b) {
				__m128d a_real = _mm_unpacklo_pd(a, a);
				__m128d a_imag = _mm_unpackhi_pd(a, a);
				__m128d b_real = _mm_unpacklo_pd(b, b);
				__m128d b_imag = _mm_unpackhi_pd(b, b);

				__m128d real = _mm_sub_pd(_mm_mul_pd(a_real, b_real), _mm_mul_pd(a_imag, b_imag));
				__m128d imag = _mm_add_pd(_mm_mul_pd(a_real, b_imag), _mm_mul_pd(a_imag, b_real));

				return _mm_unpacklo_pd(real, imag);
			}
			static inline reg fmadd(reg a, reg b, reg c) {
				reg result = mul(a, b);
				return _mm_add_pd(result, c);
			}
			static inline reg andnot(reg a, reg b)	{ return _mm_andnot_pd(a, b); }
			static inline reg max(reg a, reg b)		{ return _mm_max_pd(a, b); }
			static inline reg min(reg a, reg b)		{ return _mm_min_pd(a, b); }
			static inline reg setzero()				{ return _mm_setzero_pd(); }
		};

	template<>
		struct SimdTraits<std::complex<float>, avx2_t> {
			using reg = __m256;
			static constexpr size_t width = 8;
			static inline reg set(std::complex<float> a, std::complex<float> b,
					std::complex<float> c, std::complex<float> d) {
				return _mm256_set_ps(
						d.imag(), d.real(),
						c.imag(), c.real(),
						b.imag(), b.real(),
						a.imag(), a.real()
						);
			}
			static inline reg set1(std::complex<float> x) {
				return _mm256_set_ps(x.imag(), x.real(), x.imag(), x.real(),
						x.imag(), x.real(), x.imag(), x.real());
			}
			static inline reg load(const std::complex<float>* ptr) {
				return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
			}
			static inline reg loadu(const std::complex<float>* ptr) {
				return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
			}
			static inline void store(std::complex<float>* ptr, reg x) {
				_mm256_store_ps(reinterpret_cast<float*>(ptr), x);
			}
			static inline void storeu(std::complex<float>* ptr, reg x) {
				_mm256_storeu_ps(reinterpret_cast<float*>(ptr), x);
			}
			static inline void stream(std::complex<float>* ptr, reg x) {
				_mm256_stream_ps(reinterpret_cast<float*>(ptr), x);
			}
			static inline reg add(reg a, reg b)		{ return _mm256_add_ps(a, b); }
			static inline reg sub(reg a, reg b)		{ return _mm256_sub_ps(a, b); }
			static inline reg mul(reg a, reg b) {
				__m256 a_real = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2,0,2,0));
				__m256 a_imag = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3,1,3,1));
				__m256 b_real = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2,0,2,0));
				__m256 b_imag = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3,1,3,1));

				__m256 real = _mm256_sub_ps(_mm256_mul_ps(a_real, b_real), _mm256_mul_ps(a_imag, b_imag));
				__m256 imag = _mm256_add_ps(_mm256_mul_ps(a_real, b_imag), _mm256_mul_ps(a_imag, b_real));

				__m256 result = _mm256_unpacklo_ps(real, imag);
				__m256 result_high = _mm256_unpackhi_ps(real, imag);

				return _mm256_permute2f128_ps(result, result_high, 0x20);
			}
			static inline reg fmadd(reg a, reg b , reg c) {
				reg result = mul(a, b);
				return _mm256_add_ps(result, c);
			}
			static inline reg andnot(reg a, reg b)	{ return _mm256_andnot_ps(a, b); }
			static inline reg max(reg a, reg b)		{ return _mm256_max_ps(a, b); }
			static inline reg min(reg a, reg b)		{ return _mm256_min_ps(a, b); }
			static inline reg setzero()				{ return _mm256_setzero_ps(); }
			static inline reg zero()				{ return _mm256_setzero_ps(); }
			static inline reg fma(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b, c); }
			static inline std::complex<float> horizontal_add(reg v) {
				alignas(32) float values[8];
				_mm256_storeu_ps(values, v);
				return std::complex<float>(
						values[0] + values[2] + values[4] + values[6],
						values[1] + values[3] + values[5] + values[7]
						);
			}
		};

	template<>
		struct SimdTraits<std::complex<double>, avx2_t> {
			using reg = __m256d;
			static constexpr size_t width = 2;
			static inline reg set(std::complex<double> a, std::complex<double> b) {
				return _mm256_set_pd(
						b.imag(), b.real(),
						a.imag(), a.real()
						);
			}
			static inline reg set1(std::complex<double> x) {
				return _mm256_set_pd(x.imag(), x.real(), x.imag(), x.real());
			}
			static inline reg load(const std::complex<double>* ptr) {
				return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));
			}
			static inline reg loadu(const std::complex<double>* ptr) {
				return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));
			}
			static inline void store(std::complex<double>* ptr, reg x) {
				_mm256_storeu_pd(reinterpret_cast<double*>(ptr), x);
			}
			static inline void storeu(std::complex<double>* ptr, reg x) {
				_mm256_storeu_pd(reinterpret_cast<double*>(ptr), x);
			}
			static inline reg add(reg a, reg b)		{ return _mm256_add_pd(a, b); }
			static inline reg sub(reg a, reg b)		{ return _mm256_sub_pd(a, b); }
			static inline reg mul(reg a, reg b) {
				__m128d a_lo = _mm256_castpd256_pd128(a);
				__m128d a_hi = _mm256_extractf128_pd(a, 1); 
				__m128d b_lo = _mm256_castpd256_pd128(b);
				__m128d b_hi = _mm256_extractf128_pd(b, 1);

				__m128d a_lo_real = _mm_unpacklo_pd(a_lo, a_lo);
				__m128d a_lo_imag = _mm_unpackhi_pd(a_lo, a_lo);
				__m128d b_lo_real = _mm_unpacklo_pd(b_lo, b_lo);
				__m128d b_lo_imag = _mm_unpackhi_pd(b_lo, b_lo);

				__m128d real_lo = _mm_sub_pd(_mm_mul_pd(a_lo_real, b_lo_real), _mm_mul_pd(a_lo_imag, b_lo_imag));
				__m128d imag_lo = _mm_add_pd(_mm_mul_pd(a_lo_real, b_lo_imag), _mm_mul_pd(a_lo_imag, b_lo_real));
				__m128d result_lo = _mm_unpacklo_pd(real_lo, imag_lo);

				__m128d a_hi_real = _mm_unpacklo_pd(a_hi, a_hi);
				__m128d a_hi_imag = _mm_unpackhi_pd(a_hi, a_hi);
				__m128d b_hi_real = _mm_unpacklo_pd(b_hi, b_hi);
				__m128d b_hi_imag = _mm_unpackhi_pd(b_hi, b_hi);

				__m128d real_hi = _mm_sub_pd(_mm_mul_pd(a_hi_real, b_hi_real), _mm_mul_pd(a_hi_imag, b_hi_imag));
				__m128d imag_hi = _mm_add_pd(_mm_mul_pd(a_hi_real, b_hi_imag), _mm_mul_pd(a_hi_imag, b_hi_real));
				__m128d result_hi = _mm_unpacklo_pd(real_hi, imag_hi);

				return _mm256_insertf128_pd(_mm256_castpd128_pd256(result_lo), result_hi, 1);
			}
			static inline reg fmadd(reg a, reg b , reg c) {
				reg result = mul(a, b);
				return _mm256_add_pd(result, c);
			}
			static inline reg andnot(reg a, reg b)	{ return _mm256_andnot_pd(a, b); }
			static inline reg max(reg a, reg b)		{ return _mm256_max_pd(a, b); }
			static inline reg min(reg a, reg b)		{ return _mm256_min_pd(a, b); }
			static inline reg setzero()				{ return _mm256_setzero_pd(); }
			static inline reg fma(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b, c); }
			static inline std::complex<double> horizontal_add(reg v) {
				alignas(32) double values[4];
				_mm256_storeu_pd(values, v);
				return std::complex<double>(
						values[0] + values[2],
						values[1] + values[3]
						);
			}
		};
	template<>
		struct SimdTraits<std::complex<float>, avx512_t> {
			using reg = __m512;
			static constexpr size_t width = 8; 

			static inline reg set(std::complex<float> a, std::complex<float> b,
					std::complex<float> c, std::complex<float> d,
					std::complex<float> e, std::complex<float> f,
					std::complex<float> g, std::complex<float> h) {
				return _mm512_set_ps(
						h.imag(), h.real(),
						g.imag(), g.real(),
						f.imag(), f.real(),
						e.imag(), e.real(),
						d.imag(), d.real(),
						c.imag(), c.real(),
						b.imag(), b.real(),
						a.imag(), a.real()
						);
			}

			static inline reg set1(std::complex<float> x) {
				return _mm512_set_ps(
						x.imag(), x.real(), x.imag(), x.real(),
						x.imag(), x.real(), x.imag(), x.real(),
						x.imag(), x.real(), x.imag(), x.real(),
						x.imag(), x.real(), x.imag(), x.real()
						);
			}

			static inline reg load(const std::complex<float>* ptr) {
				return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));
			}

			static inline reg loadu(const std::complex<float>* ptr) {
				return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));
			}

			static inline void store(std::complex<float>* ptr, reg x) {
				_mm512_store_ps(reinterpret_cast<float*>(ptr), x);
			}

			static inline void storeu(std::complex<float>* ptr, reg x) {
				_mm512_storeu_ps(reinterpret_cast<float*>(ptr), x);
			}

			static inline void stream(std::complex<float>* ptr, reg x) {
				_mm512_stream_ps(reinterpret_cast<float*>(ptr), x);
			}

			static inline reg add(reg a, reg b) { return _mm512_add_ps(a, b); }
			static inline reg sub(reg a, reg b) { return _mm512_sub_ps(a, b); }

			static inline reg mul(reg a, reg b) {
				__m512 a_real = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2,0,2,0));
				__m512 a_imag = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(3,1,3,1));
				__m512 b_real = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(2,0,2,0));
				__m512 b_imag = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(3,1,3,1));

				__m512 real = _mm512_sub_ps(_mm512_mul_ps(a_real, b_real), _mm512_mul_ps(a_imag, b_imag));
				__m512 imag = _mm512_add_ps(_mm512_mul_ps(a_real, b_imag), _mm512_mul_ps(a_imag, b_real));

				return _mm512_unpacklo_ps(real, imag); 
			}

			static inline reg fma(reg a, reg b, reg c) {
#if defined(__AVX512F__) && defined(__FMA__)
				return _mm512_fmadd_ps(a, b, c);
#else
				reg result = mul(a, b);
				return _mm512_add_ps(result, c);
#endif
			}

			static inline reg setzero() { return _mm512_setzero_ps(); }
			static inline reg zero() { return _mm512_setzero_ps(); }
			static inline reg andnot(reg a, reg b) { return _mm512_andnot_ps(a, b); }
			static inline reg max(reg a, reg b) { return _mm512_max_ps(a, b); }
			static inline reg min(reg a, reg b) { return _mm512_min_ps(a, b); }

			static inline std::complex<float> horizontal_add(reg v) {
				alignas(64) float values[16];
				_mm512_storeu_ps(values, v);
				return std::complex<float>(
						values[0] + values[2] + values[4] + values[6] +
						values[8] + values[10] + values[12] + values[14],
						values[1] + values[3] + values[5] + values[7] +
						values[9] + values[11] + values[13] + values[15]
						);
			}
		};
	
	template<>
		struct SimdTraits<std::complex<double>, avx512_t> {
			using reg = __m512d;
			static constexpr size_t width = 4;

			static inline reg set(std::complex<double> a, std::complex<double> b,
					std::complex<double> c, std::complex<double> d) {
				return _mm512_set_pd(
						d.imag(), d.real(),
						c.imag(), c.real(),
						b.imag(), b.real(),
						a.imag(), a.real()
						);
			}

			static inline reg set1(std::complex<double> x) {
				return _mm512_set_pd(
						x.imag(), x.real(), x.imag(), x.real(),
						x.imag(), x.real(), x.imag(), x.real()
						);
			}

			static inline reg load(const std::complex<double>* ptr) {
				return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));
			}

			static inline reg loadu(const std::complex<double>* ptr) {
				return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));
			}

			static inline void store(std::complex<double>* ptr, reg x) {
				_mm512_store_pd(reinterpret_cast<double*>(ptr), x);
			}

			static inline void storeu(std::complex<double>* ptr, reg x) {
				_mm512_storeu_pd(reinterpret_cast<double*>(ptr), x);
			}

			static inline void stream(std::complex<double>* ptr, reg x) {
				_mm512_stream_pd(reinterpret_cast<double*>(ptr), x);
			}

			static inline reg add(reg a, reg b) { return _mm512_add_pd(a, b); }
			static inline reg sub(reg a, reg b) { return _mm512_sub_pd(a, b); }

			static inline reg mul(reg a, reg b) {
				// shuffle les réels et imaginaires
				__m512d a_real = _mm512_shuffle_pd(a, a, 0b00000000);
				__m512d a_imag = _mm512_shuffle_pd(a, a, 0b11111111);
				__m512d b_real = _mm512_shuffle_pd(b, b, 0b00000000);
				__m512d b_imag = _mm512_shuffle_pd(b, b, 0b11111111);

				__m512d real = _mm512_sub_pd(_mm512_mul_pd(a_real, b_real), _mm512_mul_pd(a_imag, b_imag));
				__m512d imag = _mm512_add_pd(_mm512_mul_pd(a_real, b_imag), _mm512_mul_pd(a_imag, b_real));

				return _mm512_unpacklo_pd(real, imag);
			}

			static inline reg fmadd(reg a, reg b, reg c) {
#if defined(__AVX512F__) && defined(__FMA__)
				return _mm512_fmadd_pd(a, b, c);
#else
				reg result = mul(a, b);
				return _mm512_add_pd(result, c);
#endif
			}

			static inline reg setzero() { return _mm512_setzero_pd(); }
			static inline reg zero() { return _mm512_setzero_pd(); }
			static inline reg andnot(reg a, reg b) { return _mm512_andnot_pd(a, b); }
			static inline reg max(reg a, reg b) { return _mm512_max_pd(a, b); }
			static inline reg min(reg a, reg b) { return _mm512_min_pd(a, b); }

			static inline std::complex<double> horizontal_add(reg v) {
				alignas(64) double values[8];
				_mm512_storeu_pd(values, v);
				return std::complex<double>(
						values[0] + values[2] + values[4] + values[6],
						values[1] + values[3] + values[5] + values[7]
						);
			}
		};
} 
