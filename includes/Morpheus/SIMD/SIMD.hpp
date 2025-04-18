#pragma once
#include <immintrin.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cpuid.h>
#include <memory>
#include <stdexcept>

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
		struct SimdTraits<float, avx2_t> {
			using reg = __m256;
			static constexpr size_t width = 8;
			static inline reg set1(float x)				{ return _mm256_set1_ps(x); }
			static inline reg setzero() { return _mm256_setzero_ps(); }
			static inline reg fma(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b, c); }
			static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
			static inline reg load(const float* ptr)	{ return _mm256_load_ps(ptr); }
			static inline reg loadu(const float* ptr) { return _mm256_loadu_ps(ptr); }
			static inline void store(float* ptr, reg x)	{ _mm256_store_ps(ptr, x); }
			static inline void storeu(float* ptr, reg x) { _mm256_storeu_ps(ptr, x); }
			static inline reg zero()					{ return _mm256_setzero_ps(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm256_fmadd_ps(a, b, c); }
			static inline reg add(reg a, reg b)			{ return _mm256_add_ps(a, b); }
			static inline reg mul(reg a, reg b)			{ return _mm256_mul_ps(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm256_sub_ps(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm256_andnot_ps(a, b); }
			static inline void store_stream(float* ptr, reg x) { _mm256_stream_ps(ptr, x); }
			static inline reg max(reg a, reg b)		{ return _mm256_max_ps(a, b); }
		};

	template<>
		struct SimdTraits<double, avx2_t> {
			using reg = __m256d;
			static constexpr size_t width = 4;
			static inline reg set1(double x)            { return _mm256_set1_pd(x); }
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

} 
