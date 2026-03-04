
#pragma once
#include <complex>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include "Analyzer.hpp"
#include <Tensorium/Backend/OpenMP/OpenMPCompat.hpp>

#if defined(__x86_64__) || defined(_M_X64)
#    include <cpuid.h>
#    include <immintrin.h>
#    define TENSORIUM_X86 1
#elif defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
#    include <arm_neon.h>
#    define TENSORIUM_ARM 1
#else
#    define TENSORIUM_FALLBACK 1
#endif

#if defined(TENSORIUM_X86)
#elif defined(TENSORIUM_ARM)
#    include <arm_neon.h>
#    define TENSORIUM_ARM 1
#    define ALIGN 16
#    define SIMD_WIDTH 4
#    define UNROLL 32
#else
#    define ALIGN 8
#endif

// #ifndef UNROLL
// #    define UNROLL 4
// #endif
// #ifndef SIMD_WIDTH
// #    define SIMD_WIDTH 4
// #endif
// #ifndef ALIGN
// #    define ALIGN 16
// #endif

#if !defined(__x86_64__)
#    define _MM_HINT_T0 0
#    define _mm_prefetch(PTR, HINT) ((void)0)
#endif

namespace simd {

template <typename T, typename ISA> struct SimdTraits;

} // namespace simd

#ifdef TENSORIUM_X86

struct sse_t {
    static constexpr size_t width = 4;
    using reg = __m128;
    static constexpr size_t alignment = 16;
};
struct avx_t {
    static constexpr size_t width = 8;
    using reg = __m256;
    static constexpr size_t alignment = 32;
};
struct avx2_t {
    static constexpr size_t width = 8;
    using reg = __m256;
    static constexpr size_t alignment = 32;
};
struct avx512_t {
    static constexpr size_t width = 16;
    using reg = __m512;
    static constexpr size_t alignment = 64;
};
#    ifdef __AVX512F__
using DefaultISA = avx512_t;
#        define ALIGN 64
#        define SIMD_WIDTH 8
#        define UNROLL 256
#    elif defined(__AVX2__)
using DefaultISA = avx2_t;
#        define ALIGN 32
#        define SIMD_WIDTH 8
#        define UNROLL 128
#    elif defined(__AVX__)
using DefaultISA = avx_t;
#        define ALIGN 32
#        define SIMD_WIDTH 8
#        define UNROLL 128
#    else
using DefaultISA = sse_t;
#        define ALIGN 16
#        define SIMD_WIDTH 4
#        define UNROLL 64
#    endif

template <typename T, std::size_t Align> struct alignas(Align) aligned_reg {
    T value;
};
namespace tensorium {

struct avx2_t {
    static constexpr size_t width = SIMD_WIDTH;
    using reg = __m256;
    static constexpr size_t alignment = ALIGN;
    using reg_aligned = aligned_reg<reg, alignment>;
};

} // namespace tensorium

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

inline bool supports_avx() {
    int regs[4];
    __cpuid_count(1, 0, regs[0], regs[1], regs[2], regs[3]);
    return (regs[2] & (1 << 28));
}

inline bool supports_sse() {
    int regs[4];
    __cpuid_count(1, 0, regs[0], regs[1], regs[2], regs[3]);
    return (regs[3] & (1 << 25));
}

#    include <iostream>
template <typename F> void dispatch_simd(F &&f) {
    if (supports_avx512()) {
        std::cout << "[dispatch] Detected AVX512\n";
        f(avx512_t{});
    } else if (supports_avx2()) {
        std::cout << "[dispatch] Detected AVX2\n";
        f(avx2_t{});
    } else if (supports_avx()) {
        std::cout << "[dispatch] Detected AVX\n";
        f(avx_t{});
    } else if (supports_sse()) {
        std::cout << "[dispatch] Detected SSE\n";
        f(sse_t{});
    } else {
        throw std::runtime_error("No supported SIMD ISA (SSE/AVX/AVX2/AVX512).");
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

inline __m256d extractf64x4_pd_fallback(__m512d v, int imm8) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, v);
    if (imm8 == 0)
        return _mm256_load_pd(&tmp[0]);
    else
        return _mm256_load_pd(&tmp[4]);
}

inline __m256i extracti64x4_epi64_fallback(__m512i v, int imm8) {
    alignas(64) std::uint64_t tmp[8];
    _mm512_store_si512(reinterpret_cast<__m512i *>(tmp), v);
    if (imm8 == 0)
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(&tmp[0]));
    else
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(&tmp[4]));
}

/*
 * REAL NUMBERS
 */

namespace detail {
__attribute__((always_inline, hot, flatten)) inline float reduce_sum(__m256 acc) {
    __m128 low = _mm256_castps256_ps128(acc);
    __m128 high = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
__attribute__((always_inline, hot, flatten)) inline double reduce_sum(__m256d acc) {
    __m128d low = _mm256_castpd256_pd128(acc);
    __m128d high = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(low, high);
    double  r[2];
    _mm_store_pd(r, sum);
    return r[0] + r[1];
}
__attribute__((always_inline, hot, flatten)) inline uint64_t reduce_sum(__m256i acc) {
#    if defined(__AVX2__)
    __m128i  low = _mm256_castsi256_si128(acc);
    __m128i  high = _mm256_extractf128_si256(acc, 1);
    __m128i  sum = _mm_add_epi64(low, high);
    uint64_t r[2];
    _mm_store_si128(reinterpret_cast<__m128i *>(r), sum);
    return r[0] + r[1];
#    else
    alignas(32) uint64_t r[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(r), acc);
    return r[0] + r[1] + r[2] + r[3];
#    endif
}
__attribute__((always_inline, hot, flatten)) inline float reduce_sum(__m512 acc) {
    __m256 low = _mm512_castps512_ps256(acc);
#    if defined(__AVX512DQ__)
    __m256 high = _mm512_extractf32x8_ps(acc, 1);
#    else
    __m256 high = extractf32x8_ps_fallback(acc, 1);
#    endif
    __m256 sum = _mm256_add_ps(low, high);
    return reduce_sum(sum);
}
__attribute__((always_inline, hot, flatten)) inline double reduce_sum(__m512d acc) {
    __m256d low = _mm512_castpd512_pd256(acc);
    #if defined(__AVX512DQ__)
    __m256d high = _mm512_extractf64x4_pd(acc, 1);
    #else
    __m256d high = extractf64x4_pd_fallback(acc, 1);
    #endif
    __m256d sum = _mm256_add_pd(low, high);
    return reduce_sum(sum);
}
__attribute__((always_inline, hot, flatten)) inline uint64_t reduce_sum(__m512i acc) {
    __m256i low = _mm512_castsi512_si256(acc);
    #if defined(__AVX512DQ__)
    __m256i high = _mm512_extracti64x4_epi64(acc, 1);
    #else
    __m256i high = extracti64x4_epi64_fallback(acc, 1);
    #endif
    __m256i sum = _mm256_add_epi64(low, high);
    return reduce_sum(sum);
}
__attribute__((always_inline, hot, flatten)) inline float reduce_sum(__m128 acc) {
    __m128 sum = _mm_hadd_ps(acc, acc);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
} // namespace detail

#    ifdef __AVX512F__
static inline __m512 andnot_fallback(__m512 a, __m512 b) {
    __m512i a_bits = _mm512_castps_si512(a);
    __m512i not_a_bits = _mm512_xor_si512(a_bits, _mm512_set1_epi32(-1));
    __m512i b_bits = _mm512_castps_si512(b);
    __m512i result_bits = _mm512_and_si512(not_a_bits, b_bits);
    return _mm512_castsi512_ps(result_bits);
}

static inline __m512d andnot_pd_fallback(__m512d a, __m512d b) {
    __m512i a_bits = _mm512_castpd_si512(a);
    __m512i not_a_bits = _mm512_xor_si512(a_bits, _mm512_set1_epi32(-1));
    __m512i b_bits = _mm512_castpd_si512(b);
    __m512i result_bits = _mm512_and_si512(not_a_bits, b_bits);
    return _mm512_castsi512_pd(result_bits);
}
#    endif

namespace simd {

template <typename T, typename ISA = DefaultISA> struct SimdTraits;
template <> struct SimdTraits<float, sse_t> {
    using reg = __m128;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(float x) { return _mm_set1_ps(x); }
    static inline reg set(float a, float b, float c, float d) { return _mm_set_ps(a, b, c, d); }
    template <int i0, int i1, int i2, int i3> static inline reg permute(reg x) {
        return _mm_permute_ps(x, _MM_SHUFFLE(i3, i2, i1, i0));
    }
    static inline reg set4(float a0, float a1, float a2, float a3) {
        return _mm_set_ps(a3, a2, a1, a0);
    }
    static inline void maskstore(float *ptr, reg mask, reg value) {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, value);
        alignas(16) int m[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(m), _mm_castps_si128(mask));
        for (int i = 0; i < 4; ++i)
            if (m[i])
                ptr[i] = tmp[i];
    }

    static inline float extract(reg x, size_t index) {
        alignas(16) float values[4];
        _mm_storeu_ps(values, x);
        return values[index];
    }
    static inline void  stream(float *ptr, reg x) { _mm_stream_ps(ptr, x); }
    static inline reg   setzero() { return _mm_setzero_ps(); }
    static inline reg   fma(reg a, reg b, reg c) {
#    if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#    else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#    endif
    }
    static inline float horizontal_add(reg v) {
        alignas(16) float values[4];
        _mm_store_ps(values, v);
        return values[0] + values[1] + values[2] + values[3];
    }
    static inline reg  load(const float *ptr) { return _mm_load_ps(ptr); }
    static inline reg  loadu(const float *ptr) { return _mm_loadu_ps(ptr); }
    static inline void store(float *ptr, reg x) { _mm_store_ps(ptr, x); }
    static inline void storeu(float *ptr, reg x) { _mm_storeu_ps(ptr, x); }
    static inline reg  zero() { return _mm_setzero_ps(); }
    static inline reg  fmadd(reg a, reg b, reg c) {
#    if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#    else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#    endif
    }
    static inline reg  add(reg a, reg b) { return _mm_add_ps(a, b); }
    static inline reg  mul(reg a, reg b) { return _mm_mul_ps(a, b); }
    static inline reg  sub(reg a, reg b) { return _mm_sub_ps(a, b); }
    static inline reg  andnot(reg a, reg b) { return _mm_andnot_ps(a, b); }
    static inline void store_stream(float *ptr, reg x) { _mm_stream_ps(ptr, x); }
    static inline reg  max(reg a, reg b) { return _mm_max_ps(a, b); }
    static inline reg  min(reg a, reg b) { return _mm_min_ps(a, b); }
};

template <> struct SimdTraits<double, sse_t> {
    using reg = __m128d;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;
    static constexpr size_t width = 2;
    static inline reg       set1(double x) { return _mm_set1_pd(x); }
    static inline reg       set(double a, double b) { return _mm_set_pd(b, a); }
    static inline double    extract(reg x, size_t index) {
        alignas(16) double values[2];
        _mm_storeu_pd(values, x);
        return values[index];
    }
    static inline void   stream(double *ptr, reg x) { _mm_stream_pd(ptr, x); }
    static inline reg    setzero() { return _mm_setzero_pd(); }
    static inline double horizontal_add(reg v) {
        alignas(16) double values[2];
        _mm_store_pd(values, v);
        return values[0] + values[1];
    }
    static inline void maskstore(double *ptr, __m128i mask, __m128d value) {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, value);
        alignas(16) int m[2];
        _mm_store_si128(reinterpret_cast<__m128i *>(m), mask);
        for (int i = 0; i < 2; ++i)
            if (m[i])
                ptr[i] = tmp[i];
    }

    static inline reg  load(const double *ptr) { return _mm_load_pd(ptr); }
    static inline reg  loadu(const double *ptr) { return _mm_loadu_pd(ptr); }
    static inline void store(double *ptr, reg x) { _mm_store_pd(ptr, x); }
    static inline void storeu(double *ptr, reg x) { _mm_storeu_pd(ptr, x); }
    static inline reg  zero() { return _mm_setzero_pd(); }
    static inline reg  fmadd(reg a, reg b, reg c) {
#    if defined(__FMA__)
        return _mm_fmadd_pd(a, b, c);
#    else
        return _mm_add_pd(_mm_mul_pd(a, b), c);
#    endif
    }
    static inline reg  add(reg a, reg b) { return _mm_add_pd(a, b); }
    static inline reg  mul(reg a, reg b) { return _mm_mul_pd(a, b); }
    static inline reg  sub(reg a, reg b) { return _mm_sub_pd(a, b); }
    static inline reg  andnot(reg a, reg b) { return _mm_andnot_pd(a, b); }
    static inline void store_stream(double *ptr, reg x) { _mm_stream_pd(ptr, x); }
    static inline reg  max(reg a, reg b) { return _mm_max_pd(a, b); }
    static inline reg  min(reg a, reg b) { return _mm_min_pd(a, b); }
};

template <> struct SimdTraits<size_t, sse_t> {
    using reg = __m128i;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg      set1(uint64_t x) { return _mm_set1_epi64x(x); }
    static inline reg      set(uint64_t a, uint64_t b) { return _mm_set_epi64x(b, a); }
    static inline uint64_t extract(reg x, size_t index) {
        alignas(16) uint64_t values[2];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(values), x);
        return values[index];
    }
    static inline void stream(uint64_t *ptr, reg x) {
        _mm_stream_si128(reinterpret_cast<__m128i *>(ptr), x);
    }
    static inline reg   setzero() { return _mm_setzero_si128(); }
    static inline float horizontal_add(reg v) {
        alignas(16) uint64_t values[2];
        _mm_store_si128(reinterpret_cast<__m128i *>(values), v);
        return values[0] + values[1];
    }
    static inline reg load(const size_t *ptr) {
        return _mm_load_si128(reinterpret_cast<const __m128i *>(ptr));
    }
    static inline reg loadu(const size_t *ptr) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
    }
    static inline void store(size_t *ptr, reg x) {
        _mm_store_si128(reinterpret_cast<__m128i *>(ptr), x);
    }
    static inline void storeu(size_t *ptr, reg x) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), x);
    }
    static inline reg zero() { return _mm_setzero_si128(); }
    static inline reg fmadd(reg a, reg b, reg c) { return _mm_add_epi64(mul(a, b), c); }
    static inline reg mul(reg a, reg b) {
        alignas(16) size_t lhs[2], rhs[2], out[2];
        _mm_store_si128((__m128i *)lhs, a);
        _mm_store_si128((__m128i *)rhs, b);
        for (size_t i = 0; i < 2; ++i)
            out[i] = lhs[i] * rhs[i];
        return _mm_load_si128((__m128i *)out);
    }
    static inline reg  add(reg a, reg b) { return _mm_add_epi64(a, b); }
    static inline reg  sub(reg a, reg b) { return _mm_sub_epi64(a, b); }
    static inline reg  andnot(reg a, reg b) { return _mm_andnot_si128(a, b); }
    static inline void store_stream(size_t *ptr, reg x) {
        _mm_stream_si128(reinterpret_cast<__m128i *>(ptr), x);
    }
    static inline reg set_epi64(int64_t a, int64_t b) { return _mm_set_epi64x(b, a); }
    static inline reg max(reg a, reg b) { return _mm_max_epi64(a, b); }
    static inline reg min(reg a, reg b) { return _mm_min_epi64(a, b); }
};

#    if defined(__AVX__)
template <> struct SimdTraits<float, avx_t> {
    using reg = __m256;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(float x) { return _mm256_set1_ps(x); }
    static inline reg set(float a, float b, float c, float d) {
        return _mm256_set_ps(a, b, c, d, a, b, c, d);
    }
    static inline void maskstore(float *ptr, __m256i mask, __m256 value) {
        _mm256_maskstore_ps(ptr, mask, value);
    }
    template <int i0, int i1, int i2, int i3> static inline reg permute(reg x) {
        constexpr int imm = _MM_SHUFFLE(i3, i2, i1, i0);
        return _mm256_permute_ps(x, imm);
    }
    static inline reg set8(float a0, float a1, float a2, float a3, float a4, float a5, float a6,
                           float a7) {
        return _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0);
    }
    static inline float extract(reg x, size_t index) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, x);
        return values[index];
    }
    static inline reg   maskload(const float *ptr, __m256i m) { return _mm256_maskload_ps(ptr, m); }
    static inline reg   broadcast(const float *ptr) { return _mm256_broadcast_ss(ptr); }
    static inline void  stream(float *ptr, reg x) { _mm256_stream_ps(ptr, x); }
    static inline reg   setzero() { return _mm256_setzero_ps(); }
    static inline reg   fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const float *ptr) { return _mm256_load_ps(ptr); }
    static inline reg   loadu(const float *ptr) { return _mm256_loadu_ps(ptr); }
    static inline void  store(float *ptr, reg x) { _mm256_store_ps(ptr, x); }
    static inline void  storeu(float *ptr, reg x) { _mm256_storeu_ps(ptr, x); }
    static inline reg   zero() { return _mm256_setzero_ps(); }
    static inline reg   fmadd(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline reg   add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm256_mul_ps(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static inline reg   andnot(reg a, reg b) { return _mm256_andnot_ps(a, b); }
    static inline void  store_stream(float *ptr, reg x) { _mm256_stream_ps(ptr, x); }
    static inline reg   max(reg a, reg b) { return _mm256_max_ps(a, b); }
    static inline reg   min(reg a, reg b) { return _mm256_min_ps(a, b); }
};

template <> struct SimdTraits<double, avx_t> {
    using reg = __m256d;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(double x) { return _mm256_set1_pd(x); }
    static inline reg set(double a, double b, double c, double d) {
        return _mm256_set_pd(a, b, c, d);
    }
    static inline double extract(reg x, size_t index) {
        alignas(32) double values[4];
        _mm256_storeu_pd(values, x);
        return values[index];
    }
    static inline void maskstore(double *ptr, __m256i mask, __m256d value) {
        _mm256_maskstore_pd(ptr, mask, value);
    }
    static inline reg  maskload(const double *ptr, __m256i m) { return _mm256_maskload_pd(ptr, m); }
    static inline reg  broadcast(const double *ptr) { return _mm256_broadcast_sd(ptr); }
    static inline void stream(double *ptr, reg x) { _mm256_stream_pd(ptr, x); }
    static inline reg  setzero() { return _mm256_setzero_pd(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const double *ptr) { return _mm256_load_pd(ptr); }
    static inline reg   loadu(const double *ptr) { return _mm256_loadu_pd(ptr); }
    static inline void  store(double *ptr, reg x) { _mm256_store_pd(ptr, x); }
    static inline void  storeu(double *ptr, reg x) { _mm256_storeu_pd(ptr, x); }
    static inline reg   zero() { return _mm256_setzero_pd(); }
    static inline reg   fmadd(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_pd(a, b, c);
#        else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#        endif
    }
    static inline reg   add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm256_mul_pd(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static inline reg   andnot(reg a, reg b) { return _mm256_andnot_pd(a, b); }
    static inline void  store_stream(double *ptr, reg x) { _mm256_stream_pd(ptr, x); }
    static inline reg   max(reg a, reg b) { return _mm256_max_pd(a, b); }
    static inline reg   min(reg a, reg b) { return _mm256_min_pd(a, b); }
};

template <> struct SimdTraits<size_t, avx_t> {
    using reg = __m256i;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(uint64_t x) { return _mm256_set1_epi64x(x); }
    static inline reg set(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        return _mm256_set_epi64x(a, b, c, d);
    }
    static inline uint64_t extract(reg x, size_t index) {
        alignas(32) uint64_t values[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(values), x);
        return values[index];
    }
    static inline void stream(uint64_t *ptr, reg x) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg   setzero() { return _mm256_setzero_si256(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const size_t *ptr) {
        static_assert(sizeof(size_t) == sizeof(uint64_t),
                      "SIMD::load(size_t*) requires 64-bit size_t");
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static inline reg loadu(const size_t *ptr) {
        static_assert(sizeof(size_t) == sizeof(uint64_t),
                      "SIMD::loadu(size_t*) requires 64-bit size_t");
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static inline void store(size_t *ptr, reg x) {
        _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline void storeu(size_t *ptr, reg x) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg mul(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = lhs[i] * rhs[i];
        return loadu(out);
    }
    static inline reg  zero() { return _mm256_setzero_si256(); }
    static inline reg  fmadd(reg a, reg b, reg c) { return add(mul(a, b), c); }
    static inline reg  add(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = lhs[i] + rhs[i];
        return loadu(out);
    }
    static inline reg  sub(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = lhs[i] - rhs[i];
        return loadu(out);
    }
    static inline reg  andnot(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = (~lhs[i]) & rhs[i];
        return loadu(out);
    }
    static inline void store_stream(size_t *ptr, reg x) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg set_epi64(int64_t a, int64_t b, int64_t c, int64_t d) {
        return _mm256_set_epi64x(a, b, c, d);
    }
    static inline reg max(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = (lhs[i] > rhs[i]) ? lhs[i] : rhs[i];
        return loadu(out);
    }
    static inline reg min(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        storeu(lhs, a);
        storeu(rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = (lhs[i] < rhs[i]) ? lhs[i] : rhs[i];
        return loadu(out);
    }
};
#    endif

#    if defined(__AVX2__)
template <> struct SimdTraits<float, avx2_t> {
    using reg = __m256;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(float x) { return _mm256_set1_ps(x); }
    static inline reg set(float a, float b, float c, float d) {
        return _mm256_set_ps(a, b, c, d, a, b, c, d);
    }
    static inline void maskstore(float *ptr, __m256i mask, __m256 value) {
        _mm256_maskstore_ps(ptr, mask, value);
    }
    template <int i0, int i1, int i2, int i3> static inline reg permute(reg x) {
        constexpr int imm = _MM_SHUFFLE(i3, i2, i1, i0);
        return _mm256_permute_ps(x, imm);
    }
    static inline reg set8(float a0, float a1, float a2, float a3, float a4, float a5, float a6,
                           float a7) {
        return _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0);
    }
    static inline float extract(reg x, size_t index) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, x);
        return values[index];
    }
    static inline reg   maskload(const float *ptr, __m256i m) { return _mm256_maskload_ps(ptr, m); }
    static inline reg   broadcast(const float *ptr) { return _mm256_broadcast_ss(ptr); }
    static inline void  stream(float *ptr, reg x) { _mm256_stream_ps(ptr, x); }
    static inline reg   setzero() { return _mm256_setzero_ps(); }
    static inline reg   fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const float *ptr) { return _mm256_load_ps(ptr); }
    static inline reg   loadu(const float *ptr) { return _mm256_loadu_ps(ptr); }
    static inline void  store(float *ptr, reg x) { _mm256_store_ps(ptr, x); }
    static inline void  storeu(float *ptr, reg x) { _mm256_storeu_ps(ptr, x); }
    static inline reg   zero() { return _mm256_setzero_ps(); }
    static inline reg   fmadd(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline reg   add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm256_mul_ps(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static inline reg   andnot(reg a, reg b) { return _mm256_andnot_ps(a, b); }
    static inline void  store_stream(float *ptr, reg x) { _mm256_stream_ps(ptr, x); }
    static inline reg   max(reg a, reg b) { return _mm256_max_ps(a, b); }
    static inline reg   min(reg a, reg b) { return _mm256_min_ps(a, b); }
};

template <> struct SimdTraits<double, avx2_t> {
    using reg = __m256d;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(double x) { return _mm256_set1_pd(x); }
    static inline reg set(double a, double b, double c, double d) {
        return _mm256_set_pd(a, b, c, d);
    }
    static inline double extract(reg x, size_t index) {
        alignas(32) double values[4];
        _mm256_storeu_pd(values, x);
        return values[index];
    }
    static inline void maskstore(double *ptr, __m256i mask, __m256d value) {
        _mm256_maskstore_pd(ptr, mask, value);
    }
    static inline reg  maskload(const double *ptr, __m256i m) { return _mm256_maskload_pd(ptr, m); }
    static inline reg  broadcast(const double *ptr) { return _mm256_broadcast_sd(ptr); }
    static inline void stream(double *ptr, reg x) { _mm256_stream_pd(ptr, x); }
    static inline reg  setzero() { return _mm256_setzero_pd(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const double *ptr) { return _mm256_load_pd(ptr); }
    static inline reg   loadu(const double *ptr) { return _mm256_loadu_pd(ptr); }
    static inline void  store(double *ptr, reg x) { _mm256_store_pd(ptr, x); }
    static inline void  storeu(double *ptr, reg x) { _mm256_storeu_pd(ptr, x); }
    static inline reg   zero() { return _mm256_setzero_pd(); }
    static inline reg   fmadd(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_pd(a, b, c);
#        else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#        endif
    }
    static inline reg   add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm256_mul_pd(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static inline reg   andnot(reg a, reg b) { return _mm256_andnot_pd(a, b); }
    static inline void  store_stream(double *ptr, reg x) { _mm256_stream_pd(ptr, x); }
    static inline reg   max(reg a, reg b) { return _mm256_max_pd(a, b); }
};

template <> struct SimdTraits<size_t, avx2_t> {
    using reg = __m256i;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(uint64_t x) { return _mm256_set1_epi64x(x); }
    static inline reg set(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        return _mm256_set_epi64x(a, b, c, d);
    }
    static inline uint64_t extract(reg x, size_t index) {
        alignas(32) uint64_t values[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(values), x);
        return values[index];
    }
    static inline void stream(uint64_t *ptr, reg x) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg   setzero() { return _mm256_setzero_si256(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const size_t *ptr) {
        static_assert(sizeof(size_t) == sizeof(uint64_t),
                      "SIMD::load(size_t*) requires 64-bit size_t");
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static inline reg loadu(const uint64_t *ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }
    static inline void store(uint64_t *ptr, reg x) {
        _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline void storeu(uint64_t *ptr, reg x) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg mul(reg a, reg b) {
        alignas(32) size_t lhs[4], rhs[4], out[4];
        _mm256_store_si256((__m256i *)lhs, a);
        _mm256_store_si256((__m256i *)rhs, b);
        for (size_t i = 0; i < 4; ++i)
            out[i] = lhs[i] * rhs[i];
        return _mm256_load_si256((__m256i *)out);
    }
    static inline reg  zero() { return _mm256_setzero_si256(); }
    static inline reg  fmadd(reg a, reg b, reg c) { return _mm256_add_epi64(mul(a, b), c); }
    static inline reg  add(reg a, reg b) { return _mm256_add_epi64(a, b); }
    static inline reg  sub(reg a, reg b) { return _mm256_sub_epi64(a, b); }
    static inline reg  andnot(reg a, reg b) { return _mm256_andnot_si256(a, b); }
    static inline void store_stream(uint64_t *ptr, reg x) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(ptr), x);
    }
    static inline reg set_epi64(int64_t a, int64_t b, int64_t c, int64_t d) {
        return _mm256_set_epi64x(a, b, c, d);
    }
    static inline reg max(reg a, reg b) { return _mm256_max_epi64(a, b); }
};
#    endif
#    ifdef __AVX512F__

template <> struct SimdTraits<float, avx512_t> {
    using reg = __m512;
    static constexpr size_t width = 16;
    static constexpr size_t alignment = 64;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(float x) { return _mm512_set1_ps(x); }
    static inline reg set(float a, float b, float c, float d, float e, float f, float g, float h,
                          float i, float j, float k, float l, float m, float n, float o, float p) {
        return _mm512_set_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    }
    static inline float extract(reg x, size_t index) {
        alignas(64) float values[16];
        _mm512_storeu_ps(values, x);
        return values[index];
    }
    static inline void  stream(float *ptr, reg x) { _mm512_stream_ps(ptr, x); }
    static inline reg   setzero() { return _mm512_setzero_ps(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const float *ptr) { return _mm512_load_ps(ptr); }
    static inline reg   loadu(const float *ptr) { return _mm512_loadu_ps(ptr); }
    static inline void  store(float *ptr, reg x) { _mm512_store_ps(ptr, x); }
    static inline void  storeu(float *ptr, reg x) { _mm512_storeu_ps(ptr, x); }
    static inline reg   loadu_stream(const float *ptr) { return _mm512_loadu_ps(ptr); }
    static inline reg   zero() { return _mm512_setzero_ps(); }
    static inline reg   fmadd(reg a, reg b, reg c) { return _mm512_fmadd_ps(a, b, c); }
    static inline reg   add(reg a, reg b) { return _mm512_add_ps(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm512_mul_ps(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm512_sub_ps(a, b); }
#        if defined(__AVX512DQ__)
    static inline reg andnot(reg a, reg b) { return _mm512_andnot_ps(a, b); }
#        else
    static inline reg andnot(reg a, reg b) { return andnot_fallback(a, b); }
#        endif
    static inline void store_stream(float *ptr, reg x) { _mm512_stream_ps(ptr, x); }
    static inline reg  max(reg a, reg b) { return _mm512_max_ps(a, b); }
};

template <> struct SimdTraits<double, avx512_t> {
    using reg = __m512d;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 64;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(double x) { return _mm512_set1_pd(x); }
    static inline reg set(double a, double b, double c, double d, double e, double f, double g,
                          double h) {
        return _mm512_set_pd(a, b, c, d, e, f, g, h);
    }
    static inline double extract(reg x, size_t index) {
        alignas(64) double values[8];
        _mm512_storeu_pd(values, x);
        return values[index];
    }
    static inline void  stream(double *ptr, reg x) { _mm512_stream_pd(ptr, x); }
    static inline reg   setzero() { return _mm512_setzero_pd(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const double *ptr) { return _mm512_load_pd(ptr); }
    static inline reg   loadu(const double *ptr) { return _mm512_loadu_pd(ptr); }
    static inline void  store(double *ptr, reg x) { _mm512_store_pd(ptr, x); }
    static inline void  storeu(double *ptr, reg x) { _mm512_storeu_pd(ptr, x); }
    static inline reg   loadu_stream(const double *ptr) { return _mm512_loadu_pd(ptr); }
    static inline reg   zero() { return _mm512_setzero_pd(); }
    static inline reg   fmadd(reg a, reg b, reg c) { return _mm512_fmadd_pd(a, b, c); }
    static inline reg   add(reg a, reg b) { return _mm512_add_pd(a, b); }
    static inline reg   mul(reg a, reg b) { return _mm512_mul_pd(a, b); }
    static inline reg   sub(reg a, reg b) { return _mm512_sub_pd(a, b); }
    #        if defined(__AVX512DQ__)
    static inline reg   andnot(reg a, reg b) { return _mm512_andnot_pd(a, b); }
    #        else
    static inline reg   andnot(reg a, reg b) { return andnot_pd_fallback(a, b); }
    #        endif
    static inline void  store_stream(double *ptr, reg x) { _mm512_stream_pd(ptr, x); }
    static inline reg   max(reg a, reg b) { return _mm512_max_pd(a, b); }
};

template <> struct SimdTraits<size_t, avx512_t> {
    using reg = __m512i;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 64;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set1(size_t x) { return _mm512_set1_epi64(x); }
    static inline reg set(size_t a, size_t b, size_t c, size_t d, size_t e, size_t f, size_t g,
                          size_t h) {
        return _mm512_set_epi64(a, b, c, d, e, f, g, h);
    }
    static inline size_t extract(reg x, size_t index) {
        alignas(64) size_t values[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i *>(values), x);
        return values[index];
    }
    static inline reg   setzero() { return _mm512_setzero_si512(); }
    static inline float horizontal_add(reg v) { return detail::reduce_sum(v); }
    static inline reg   load(const size_t *ptr) {
        return _mm512_load_si512(reinterpret_cast<const __m512i *>(ptr));
    }
    static inline reg loadu(const size_t *ptr) {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr));
    }
    static inline void store(size_t *ptr, reg x) {
        _mm512_store_si512(reinterpret_cast<__m512i *>(ptr), x);
    }
    static inline void storeu(size_t *ptr, reg x) {
        _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), x);
    }
    static inline reg zero() { return _mm512_setzero_si512(); }
    static inline reg add(reg a, reg b) { return _mm512_add_epi64(a, b); }
#        if defined(__AVX512DQ__)
    static inline reg mul(reg a, reg b) { return _mm512_mullo_epi64(a, b); }
#        else
    static inline reg mul(reg a, reg b) {
        alignas(64) uint64_t A[8], B[8], R[8];
        _mm512_store_epi64(A, a);
        _mm512_store_epi64(B, b);
        for (int i = 0; i < 8; ++i)
            R[i] = A[i] * B[i];
        return _mm512_load_epi64(R);
    }
#        endif

    static inline reg  fmadd(reg a, reg b, reg c) { return _mm512_add_epi64(mul(a, b), c); }
    static inline reg  sub(reg a, reg b) { return _mm512_sub_epi64(a, b); }
    static inline reg  andnot(reg a, reg b) { return _mm512_andnot_si512(a, b); }
    static inline void store_stream(size_t *ptr, reg x) {
        _mm512_stream_si512(reinterpret_cast<__m512i *>(ptr), x);
    }
    static inline reg set_epi64(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                                int64_t g, int64_t h) {
        return _mm512_set_epi64(a, b, c, d, e, f, g, h);
    }
    static inline reg max(reg a, reg b) { return _mm512_max_epi64(a, b); }
};
#    endif
/*
 * COMPLEX NUMBERS
 */

template <> struct SimdTraits<std::complex<float>, sse_t> {
    using reg = __m128;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<float> a, std::complex<float> b) {
        return _mm_set_ps(b.imag(), a.real(), b.real(), a.imag());
    }
    static inline reg set1(std::complex<float> x) {
        return _mm_set_ps(x.imag(), x.real(), x.imag(), x.real());
    }
    static inline reg load(const std::complex<float> *ptr) {
        return _mm_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline std::complex<float> extract(reg x, size_t index) {
        alignas(16) float values[4];
        _mm_storeu_ps(values, x);
        return std::complex<float>(values[2 * index], values[2 * index + 1]);
    }
    static inline reg loadu(const std::complex<float> *ptr) {
        return _mm_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline void store(std::complex<float> *ptr, reg x) {
        _mm_storeu_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void storeu(std::complex<float> *ptr, reg x) {
        _mm_storeu_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline reg add(reg a, reg b) { return _mm_add_ps(a, b); }
    static inline reg sub(reg a, reg b) { return _mm_sub_ps(a, b); }
    static inline reg mul(reg a, reg b) {
        __m128 a_real = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0));
        __m128 a_imag = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));
        __m128 b_real = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));
        __m128 b_imag = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));

        __m128 real = _mm_sub_ps(_mm_mul_ps(a_real, b_real), _mm_mul_ps(a_imag, b_imag));
        __m128 imag = _mm_add_ps(_mm_mul_ps(a_real, b_imag), _mm_mul_ps(a_imag, b_real));

        return _mm_unpacklo_ps(real, imag);
    }
    static inline reg                 andnot(reg a, reg b) { return _mm_andnot_ps(a, b); }
    static inline reg                 max(reg a, reg b) { return _mm_max_ps(a, b); }
    static inline reg                 min(reg a, reg b) { return _mm_min_ps(a, b); }
    static inline reg                 setzero() { return _mm_setzero_ps(); }
    static inline reg                 fma(reg a, reg b, reg c) {
#    if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#    else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#    endif
    }
    static inline std::complex<float> horizontal_add(reg v) {
        alignas(16) float values[4];
        _mm_storeu_ps(values, v);
        return std::complex<float>(values[0] + values[2], values[1] + values[3]);
    }
    static inline void stream(std::complex<float> *ptr, reg x) {
        _mm_stream_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void store_stream(std::complex<float> *ptr, reg x) {
        _mm_stream_ps(reinterpret_cast<float *>(ptr), x);
    }
};

template <> struct SimdTraits<std::complex<double>, sse_t> {
    using reg = __m128d;
    static constexpr size_t width = 1;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<double> a, std::complex<double> b) {
        return _mm_set_pd(b.imag(), a.real());
    }
    static inline reg set1(std::complex<double> x) { return _mm_set_pd(x.imag(), x.real()); }
    static inline reg load(const std::complex<double> *ptr) {
        return _mm_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline reg loadu(const std::complex<double> *ptr) {
        return _mm_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline std::complex<double> extract(reg x, size_t /*index*/ = 0) {
        alignas(16) double values[2];
        _mm_storeu_pd(values, x);
        return std::complex<double>(values[0], values[1]);
    }
    static inline void store(std::complex<double> *ptr, reg x) {
        _mm_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline void storeu(std::complex<double> *ptr, reg x) {
        _mm_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline reg add(reg a, reg b) { return _mm_add_pd(a, b); }
    static inline reg sub(reg a, reg b) { return _mm_sub_pd(a, b); }
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
    static inline reg andnot(reg a, reg b) { return _mm_andnot_pd(a, b); }
    static inline reg max(reg a, reg b) { return _mm_max_pd(a, b); }
    static inline reg min(reg a, reg b) { return _mm_min_pd(a, b); }
    static inline reg setzero() { return _mm_setzero_pd(); }
};

#    if defined(__AVX__)
template <> struct SimdTraits<std::complex<float>, avx_t> {
    using reg = __m256;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<float> a, std::complex<float> b, std::complex<float> c,
                          std::complex<float> d) {
        return _mm256_set_ps(d.imag(), d.real(), c.imag(), c.real(), b.imag(), b.real(), a.imag(),
                             a.real());
    }
    static inline reg set1(std::complex<float> x) {
        return _mm256_set_ps(x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(),
                             x.real());
    }
    static inline reg load(const std::complex<float> *ptr) {
        return _mm256_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline reg loadu(const std::complex<float> *ptr) {
        return _mm256_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline std::complex<float> extract(reg x, size_t index) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, x);
        return std::complex<float>(values[2 * index], values[2 * index + 1]);
    }
    static inline reg broadcast(const std::complex<float> *ptr) {
        float re = ptr->real();
        float im = ptr->imag();
        return _mm256_set_ps(im, re, im, re, im, re, im, re);
    }

    static inline void store(std::complex<float> *ptr, reg x) {
        _mm256_store_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void storeu(std::complex<float> *ptr, reg x) {
        _mm256_storeu_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void stream(std::complex<float> *ptr, reg x) {
        _mm256_stream_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline reg add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static inline reg sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static inline reg mul(reg a, reg b) {
        __m256 a_real = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 a_imag = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 b_real = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 b_imag = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 3, 1));

        __m256 real = _mm256_sub_ps(_mm256_mul_ps(a_real, b_real), _mm256_mul_ps(a_imag, b_imag));
        __m256 imag = _mm256_add_ps(_mm256_mul_ps(a_real, b_imag), _mm256_mul_ps(a_imag, b_real));

        __m256 result = _mm256_unpacklo_ps(real, imag);
        __m256 result_high = _mm256_unpackhi_ps(real, imag);

        return _mm256_permute2f128_ps(result, result_high, 0x20);
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        reg result = mul(a, b);
        return _mm256_add_ps(result, c);
    }
    static inline reg                 andnot(reg a, reg b) { return _mm256_andnot_ps(a, b); }
    static inline reg                 max(reg a, reg b) { return _mm256_max_ps(a, b); }
    static inline reg                 min(reg a, reg b) { return _mm256_min_ps(a, b); }
    static inline reg                 setzero() { return _mm256_setzero_ps(); }
    static inline reg                 zero() { return _mm256_setzero_ps(); }
    static inline reg                 fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline std::complex<float> horizontal_add(reg v) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, v);
        return std::complex<float>(values[0] + values[2] + values[4] + values[6],
                                   values[1] + values[3] + values[5] + values[7]);
    }
    static inline reg maskload(const std::complex<float> *ptr, __m256i mask) {
        return _mm256_maskload_ps(reinterpret_cast<const float *>(ptr), mask);
    }
    static inline void maskstore(std::complex<float> *ptr, __m256i mask, reg v) {
        _mm256_maskstore_ps(reinterpret_cast<float *>(ptr), mask, v);
    }
};

template <> struct SimdTraits<std::complex<double>, avx_t> {
    using reg = __m256d;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<double> a, std::complex<double> b) {
        return _mm256_set_pd(b.imag(), b.real(), a.imag(), a.real());
    }
    static inline reg set1(std::complex<double> x) {
        return _mm256_set_pd(x.imag(), x.real(), x.imag(), x.real());
    }
    static inline reg load(const std::complex<double> *ptr) {
        return _mm256_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline reg loadu(const std::complex<double> *ptr) {
        return _mm256_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline void store(std::complex<double> *ptr, reg x) {
        _mm256_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline void storeu(std::complex<double> *ptr, reg x) {
        _mm256_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline reg broadcast(const std::complex<double> *ptr) {
        double re = ptr->real();
        double im = ptr->imag();
        return _mm256_set_pd(im, re, im, re);
    }
    static inline reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static inline reg sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static inline reg mul(reg a, reg b) {
        __m128d a_lo = _mm256_castpd256_pd128(a);
        __m128d a_hi = _mm256_extractf128_pd(a, 1);
        __m128d b_lo = _mm256_castpd256_pd128(b);
        __m128d b_hi = _mm256_extractf128_pd(b, 1);

        __m128d a_lo_real = _mm_unpacklo_pd(a_lo, a_lo);
        __m128d a_lo_imag = _mm_unpackhi_pd(a_lo, a_lo);
        __m128d b_lo_real = _mm_unpacklo_pd(b_lo, b_lo);
        __m128d b_lo_imag = _mm_unpackhi_pd(b_lo, b_lo);

        __m128d real_lo =
            _mm_sub_pd(_mm_mul_pd(a_lo_real, b_lo_real), _mm_mul_pd(a_lo_imag, b_lo_imag));
        __m128d imag_lo =
            _mm_add_pd(_mm_mul_pd(a_lo_real, b_lo_imag), _mm_mul_pd(a_lo_imag, b_lo_real));
        __m128d result_lo = _mm_unpacklo_pd(real_lo, imag_lo);

        __m128d a_hi_real = _mm_unpacklo_pd(a_hi, a_hi);
        __m128d a_hi_imag = _mm_unpackhi_pd(a_hi, a_hi);
        __m128d b_hi_real = _mm_unpacklo_pd(b_hi, b_hi);
        __m128d b_hi_imag = _mm_unpackhi_pd(b_hi, b_hi);

        __m128d real_hi =
            _mm_sub_pd(_mm_mul_pd(a_hi_real, b_hi_real), _mm_mul_pd(a_hi_imag, b_hi_imag));
        __m128d imag_hi =
            _mm_add_pd(_mm_mul_pd(a_hi_real, b_hi_imag), _mm_mul_pd(a_hi_imag, b_hi_real));
        __m128d result_hi = _mm_unpacklo_pd(real_hi, imag_hi);

        return _mm256_insertf128_pd(_mm256_castpd128_pd256(result_lo), result_hi, 1);
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        reg result = mul(a, b);
        return _mm256_add_pd(result, c);
    }
    static inline reg                  andnot(reg a, reg b) { return _mm256_andnot_pd(a, b); }
    static inline reg                  max(reg a, reg b) { return _mm256_max_pd(a, b); }
    static inline reg                  min(reg a, reg b) { return _mm256_min_pd(a, b); }
    static inline reg                  setzero() { return _mm256_setzero_pd(); }
    static inline reg                  fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_pd(a, b, c);
#        else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#        endif
    }
    static inline std::complex<double> horizontal_add(reg v) {
        alignas(32) double values[4];
        _mm256_storeu_pd(values, v);
        return std::complex<double>(values[0] + values[2], values[1] + values[3]);
    }
    static inline reg maskload(const std::complex<double> *ptr, __m256i mask) {
        return _mm256_maskload_pd(reinterpret_cast<const double *>(ptr), mask);
    }
    static inline void maskstore(std::complex<double> *ptr, __m256i mask, reg v) {
        _mm256_maskstore_pd(reinterpret_cast<double *>(ptr), mask, v);
    }
};
#    endif

#    if defined(__AVX2__)
template <> struct SimdTraits<std::complex<float>, avx2_t> {
    using reg = __m256;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<float> a, std::complex<float> b, std::complex<float> c,
                          std::complex<float> d) {
        return _mm256_set_ps(d.imag(), d.real(), c.imag(), c.real(), b.imag(), b.real(), a.imag(),
                             a.real());
    }
    static inline reg set1(std::complex<float> x) {
        return _mm256_set_ps(x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(),
                             x.real());
    }
    static inline reg load(const std::complex<float> *ptr) {
        return _mm256_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline reg loadu(const std::complex<float> *ptr) {
        return _mm256_loadu_ps(reinterpret_cast<const float *>(ptr));
    }
    static inline std::complex<float> extract(reg x, size_t index) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, x);
        return std::complex<float>(values[2 * index], values[2 * index + 1]);
    }
    static inline reg broadcast(const std::complex<float> *ptr) {
        float re = ptr->real();
        float im = ptr->imag();
        return _mm256_set_ps(im, re, im, re, im, re, im, re);
    }

    static inline void store(std::complex<float> *ptr, reg x) {
        _mm256_store_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void storeu(std::complex<float> *ptr, reg x) {
        _mm256_storeu_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline void stream(std::complex<float> *ptr, reg x) {
        _mm256_stream_ps(reinterpret_cast<float *>(ptr), x);
    }
    static inline reg add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static inline reg sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static inline reg mul(reg a, reg b) {
        __m256 a_real = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 a_imag = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 b_real = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 b_imag = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 3, 1));

        __m256 real = _mm256_sub_ps(_mm256_mul_ps(a_real, b_real), _mm256_mul_ps(a_imag, b_imag));
        __m256 imag = _mm256_add_ps(_mm256_mul_ps(a_real, b_imag), _mm256_mul_ps(a_imag, b_real));

        __m256 result = _mm256_unpacklo_ps(real, imag);
        __m256 result_high = _mm256_unpackhi_ps(real, imag);

        return _mm256_permute2f128_ps(result, result_high, 0x20);
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        reg result = mul(a, b);
        return _mm256_add_ps(result, c);
    }
    static inline reg                 andnot(reg a, reg b) { return _mm256_andnot_ps(a, b); }
    static inline reg                 max(reg a, reg b) { return _mm256_max_ps(a, b); }
    static inline reg                 min(reg a, reg b) { return _mm256_min_ps(a, b); }
    static inline reg                 setzero() { return _mm256_setzero_ps(); }
    static inline reg                 zero() { return _mm256_setzero_ps(); }
    static inline reg                 fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#        else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#        endif
    }
    static inline std::complex<float> horizontal_add(reg v) {
        alignas(32) float values[8];
        _mm256_storeu_ps(values, v);
        return std::complex<float>(values[0] + values[2] + values[4] + values[6],
                                   values[1] + values[3] + values[5] + values[7]);
    }
    static inline reg maskload(const std::complex<float> *ptr, __m256i mask) {
        return _mm256_maskload_ps(reinterpret_cast<const float *>(ptr), mask);
    }
    static inline void maskstore(std::complex<float> *ptr, __m256i mask, reg v) {
        _mm256_maskstore_ps(reinterpret_cast<float *>(ptr), mask, v);
    }
};

template <> struct SimdTraits<std::complex<double>, avx2_t> {
    using reg = __m256d;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<double> a, std::complex<double> b) {
        return _mm256_set_pd(b.imag(), b.real(), a.imag(), a.real());
    }
    static inline reg set1(std::complex<double> x) {
        return _mm256_set_pd(x.imag(), x.real(), x.imag(), x.real());
    }
    static inline reg load(const std::complex<double> *ptr) {
        return _mm256_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline reg loadu(const std::complex<double> *ptr) {
        return _mm256_loadu_pd(reinterpret_cast<const double *>(ptr));
    }
    static inline void store(std::complex<double> *ptr, reg x) {
        _mm256_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline void storeu(std::complex<double> *ptr, reg x) {
        _mm256_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }
    static inline reg broadcast(const std::complex<double> *ptr) {
        double re = ptr->real();
        double im = ptr->imag();
        return _mm256_set_pd(im, re, im, re);
    }
    static inline reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static inline reg sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static inline reg mul(reg a, reg b) {
        __m128d a_lo = _mm256_castpd256_pd128(a);
        __m128d a_hi = _mm256_extractf128_pd(a, 1);
        __m128d b_lo = _mm256_castpd256_pd128(b);
        __m128d b_hi = _mm256_extractf128_pd(b, 1);

        __m128d a_lo_real = _mm_unpacklo_pd(a_lo, a_lo);
        __m128d a_lo_imag = _mm_unpackhi_pd(a_lo, a_lo);
        __m128d b_lo_real = _mm_unpacklo_pd(b_lo, b_lo);
        __m128d b_lo_imag = _mm_unpackhi_pd(b_lo, b_lo);

        __m128d real_lo =
            _mm_sub_pd(_mm_mul_pd(a_lo_real, b_lo_real), _mm_mul_pd(a_lo_imag, b_lo_imag));
        __m128d imag_lo =
            _mm_add_pd(_mm_mul_pd(a_lo_real, b_lo_imag), _mm_mul_pd(a_lo_imag, b_lo_real));
        __m128d result_lo = _mm_unpacklo_pd(real_lo, imag_lo);

        __m128d a_hi_real = _mm_unpacklo_pd(a_hi, a_hi);
        __m128d a_hi_imag = _mm_unpackhi_pd(a_hi, a_hi);
        __m128d b_hi_real = _mm_unpacklo_pd(b_hi, b_hi);
        __m128d b_hi_imag = _mm_unpackhi_pd(b_hi, b_hi);

        __m128d real_hi =
            _mm_sub_pd(_mm_mul_pd(a_hi_real, b_hi_real), _mm_mul_pd(a_hi_imag, b_hi_imag));
        __m128d imag_hi =
            _mm_add_pd(_mm_mul_pd(a_hi_real, b_hi_imag), _mm_mul_pd(a_hi_imag, b_hi_real));
        __m128d result_hi = _mm_unpacklo_pd(real_hi, imag_hi);

        return _mm256_insertf128_pd(_mm256_castpd128_pd256(result_lo), result_hi, 1);
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        reg result = mul(a, b);
        return _mm256_add_pd(result, c);
    }
    static inline reg                  andnot(reg a, reg b) { return _mm256_andnot_pd(a, b); }
    static inline reg                  max(reg a, reg b) { return _mm256_max_pd(a, b); }
    static inline reg                  min(reg a, reg b) { return _mm256_min_pd(a, b); }
    static inline reg                  setzero() { return _mm256_setzero_pd(); }
    static inline reg                  fma(reg a, reg b, reg c) {
#        if defined(__FMA__)
        return _mm256_fmadd_pd(a, b, c);
#        else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#        endif
    }
    static inline std::complex<double> horizontal_add(reg v) {
        alignas(32) double values[4];
        _mm256_storeu_pd(values, v);
        return std::complex<double>(values[0] + values[2], values[1] + values[3]);
    }
    static inline reg maskload(const std::complex<double> *ptr, __m256i mask) {
        return _mm256_maskload_pd(reinterpret_cast<const double *>(ptr), mask);
    }
    static inline void maskstore(std::complex<double> *ptr, __m256i mask, reg v) {
        _mm256_maskstore_pd(reinterpret_cast<double *>(ptr), mask, v);
    }
};
#    endif

#    ifdef __AVX512F__

template <> struct SimdTraits<std::complex<float>, avx512_t> {
    using reg = __m512;
    static constexpr size_t width = 8;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<float> a, std::complex<float> b, std::complex<float> c,
                          std::complex<float> d, std::complex<float> e, std::complex<float> f,
                          std::complex<float> g, std::complex<float> h) {
        return _mm512_set_ps(h.imag(), h.real(), g.imag(), g.real(), f.imag(), f.real(), e.imag(),
                             e.real(), d.imag(), d.real(), c.imag(), c.real(), b.imag(), b.real(),
                             a.imag(), a.real());
    }

    static inline reg set1(std::complex<float> x) {
        return _mm512_set_ps(x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(),
                             x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(),
                             x.imag(), x.real());
    }

    static inline reg load(const std::complex<float> *ptr) {
        return _mm512_loadu_ps(reinterpret_cast<const float *>(ptr));
    }

    static inline reg loadu(const std::complex<float> *ptr) {
        return _mm512_loadu_ps(reinterpret_cast<const float *>(ptr));
    }

    static inline void store(std::complex<float> *ptr, reg x) {
        _mm512_store_ps(reinterpret_cast<float *>(ptr), x);
    }

    static inline void storeu(std::complex<float> *ptr, reg x) {
        _mm512_storeu_ps(reinterpret_cast<float *>(ptr), x);
    }

    static inline void stream(std::complex<float> *ptr, reg x) {
        _mm512_stream_ps(reinterpret_cast<float *>(ptr), x);
    }

    static inline reg add(reg a, reg b) { return _mm512_add_ps(a, b); }
    static inline reg sub(reg a, reg b) { return _mm512_sub_ps(a, b); }

    static inline reg mul(reg a, reg b) {
        __m512 a_real = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 a_imag = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 3, 1));
        __m512 b_real = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 b_imag = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 3, 1));

        __m512 real = _mm512_sub_ps(_mm512_mul_ps(a_real, b_real), _mm512_mul_ps(a_imag, b_imag));
        __m512 imag = _mm512_add_ps(_mm512_mul_ps(a_real, b_imag), _mm512_mul_ps(a_imag, b_real));

        return _mm512_unpacklo_ps(real, imag);
    }

    static inline reg fma(reg a, reg b, reg c) {
#        if defined(__AVX512F__) && defined(__FMA__)
        return _mm512_fmadd_ps(a, b, c);
#        else
        reg result = mul(a, b);
        return _mm512_add_ps(result, c);
#        endif
    }

    static inline reg setzero() { return _mm512_setzero_ps(); }
    static inline reg zero() { return _mm512_setzero_ps(); }
    #        if defined(__AVX512DQ__)
    static inline reg andnot(reg a, reg b) { return _mm512_andnot_ps(a, b); }
    #        else
    static inline reg andnot(reg a, reg b) { return andnot_fallback(a, b); }
    #        endif
    static inline reg max(reg a, reg b) { return _mm512_max_ps(a, b); }
    static inline reg min(reg a, reg b) { return _mm512_min_ps(a, b); }

    static inline std::complex<float> horizontal_add(reg v) {
        alignas(64) float values[16];
        _mm512_storeu_ps(values, v);
        return std::complex<float>(values[0] + values[2] + values[4] + values[6] + values[8] +
                                       values[10] + values[12] + values[14],
                                   values[1] + values[3] + values[5] + values[7] + values[9] +
                                       values[11] + values[13] + values[15]);
    }
};

template <> struct SimdTraits<std::complex<double>, avx512_t> {
    using reg = __m512d;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 32;
    using reg_aligned = aligned_reg<reg, alignment>;
    static inline reg set(std::complex<double> a, std::complex<double> b, std::complex<double> c,
                          std::complex<double> d) {
        return _mm512_set_pd(d.imag(), d.real(), c.imag(), c.real(), b.imag(), b.real(), a.imag(),
                             a.real());
    }

    static inline reg set1(std::complex<double> x) {
        return _mm512_set_pd(x.imag(), x.real(), x.imag(), x.real(), x.imag(), x.real(), x.imag(),
                             x.real());
    }

    static inline reg load(const std::complex<double> *ptr) {
        return _mm512_loadu_pd(reinterpret_cast<const double *>(ptr));
    }

    static inline reg loadu(const std::complex<double> *ptr) {
        return _mm512_loadu_pd(reinterpret_cast<const double *>(ptr));
    }

    static inline void store(std::complex<double> *ptr, reg x) {
        _mm512_store_pd(reinterpret_cast<double *>(ptr), x);
    }

    static inline void storeu(std::complex<double> *ptr, reg x) {
        _mm512_storeu_pd(reinterpret_cast<double *>(ptr), x);
    }

    static inline void stream(std::complex<double> *ptr, reg x) {
        _mm512_stream_pd(reinterpret_cast<double *>(ptr), x);
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
#        if defined(__AVX512F__) && defined(__FMA__)
        return _mm512_fmadd_pd(a, b, c);
#        else
        reg result = mul(a, b);
        return _mm512_add_pd(result, c);
#        endif
    }

    static inline reg setzero() { return _mm512_setzero_pd(); }
    static inline reg zero() { return _mm512_setzero_pd(); }
    #        if defined(__AVX512DQ__)
    static inline reg andnot(reg a, reg b) { return _mm512_andnot_pd(a, b); }
    #        else
    static inline reg andnot(reg a, reg b) { return andnot_pd_fallback(a, b); }
    #        endif
    static inline reg max(reg a, reg b) { return _mm512_max_pd(a, b); }
    static inline reg min(reg a, reg b) { return _mm512_min_pd(a, b); }

    static inline std::complex<double> horizontal_add(reg v) {
        alignas(64) double values[8];
        _mm512_storeu_pd(values, v);
        return std::complex<double>(values[0] + values[2] + values[4] + values[6],
                                    values[1] + values[3] + values[5] + values[7]);
    }
};
#    endif
} // namespace simd

#endif // TENSORIUM_X86
#ifdef TENSORIUM_ARM
#    include <arm_neon.h>

#    define ALIGN 16
#    define SIMD_WIDTH 4
#    define UNROLL 32

struct complex_reg_f32 {
    float32x4_t v;
};
struct complex_reg_f64 {
    float64x2_t v;
};

struct neon32_t {
    static constexpr size_t width = 4;
    using reg = float32x4_t;
    static constexpr size_t alignment = 16;
};

struct neon64_t {
    static constexpr size_t width = 2;
    using reg = float64x2_t;
    static constexpr size_t alignment = 16;
};

using DefaultISA = neon32_t;

template <typename T, std::size_t Align> struct alignas(Align) aligned_reg {
    T value;
};

template <typename F> inline void dispatch_simd(F &&f) { f(DefaultISA{}); }

namespace simd {

static inline float32x4_t andnot_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t na = veorq_u32(vreinterpretq_u32_f32(a), vdupq_n_u32(~0u));
    return vreinterpretq_f32_u32(vandq_u32(na, vreinterpretq_u32_f32(b)));
}

template <> struct SimdTraits<float, neon32_t> {
    using reg = float32x4_t;
    static constexpr size_t width = 4;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;

    static inline reg  set1(float x) { return vdupq_n_f32(x); }
    static inline reg  load(const float *p) { return vld1q_f32(p); }
    static inline reg  loadu(const float *p) { return vld1q_f32(p); }
    static inline void store(float *p, reg v) { vst1q_f32(p, v); }
    static inline void storeu(float *p, reg v) { vst1q_f32(p, v); }
    static inline void store_stream(float *p, reg v) { vst1q_f32(p, v); }

    static inline reg zero() { return vdupq_n_f32(0.f); }
    static inline reg setzero() { return zero(); }
    static inline reg broadcast(const float *ptr) { return vdupq_n_f32(*ptr); }
    static inline reg add(reg a, reg b) { return vaddq_f32(a, b); }
    static inline reg sub(reg a, reg b) { return vsubq_f32(a, b); }
    static inline reg mul(reg a, reg b) { return vmulq_f32(a, b); }

#    if defined(__aarch64__)
    static inline reg fmadd(reg a, reg b, reg c) { return vfmaq_f32(c, a, b); }
#    else
    static inline reg fmadd(reg a, reg b, reg c) { return vaddq_f32(c, vmulq_f32(a, b)); }
#    endif
    static inline reg fma(reg a, reg b, reg c) { return fmadd(a, b, c); }

    static inline reg max(reg a, reg b) { return vmaxq_f32(a, b); }
    static inline reg min(reg a, reg b) { return vminq_f32(a, b); }
    static inline reg andnot(reg a, reg b) { return andnot_f32(a, b); }

    static inline float extract(reg x, size_t idx) {
        alignas(16) float t[4];
        vst1q_f32(t, x);
        return t[idx & 3];
    }

    static inline float horizontal_add(reg v) { return vaddvq_f32(v); }
};

template <> struct SimdTraits<double, neon32_t> {
    using reg = float64x2_t;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;

    static inline reg  set1(double x) { return vdupq_n_f64(x); }
    static inline reg  load(const double *p) { return vld1q_f64(p); }
    static inline reg  loadu(const double *p) { return vld1q_f64(p); }
    static inline void store(double *p, reg v) { vst1q_f64(p, v); }
    static inline void storeu(double *p, reg v) { vst1q_f64(p, v); }
    static inline void store_stream(double *p, reg v) { vst1q_f64(p, v); }

    static inline reg zero() { return vdupq_n_f64(0.0); }
    static inline reg setzero() { return zero(); }
    static inline reg broadcast(const double *ptr) { return vdupq_n_f64(*ptr); }
    static inline reg add(reg a, reg b) { return vaddq_f64(a, b); }
    static inline reg sub(reg a, reg b) { return vsubq_f64(a, b); }
    static inline reg mul(reg a, reg b) { return vmulq_f64(a, b); }

#    if defined(__aarch64__)
    static inline reg fmadd(reg a, reg b, reg c) { return vfmaq_f64(c, a, b); }
#    else
    static inline reg fmadd(reg a, reg b, reg c) { return vaddq_f64(c, vmulq_f64(a, b)); }
#    endif
    static inline reg fma(reg a, reg b, reg c) { return fmadd(a, b, c); }

    static inline reg max(reg a, reg b) { return vmaxq_f64(a, b); }
    static inline reg min(reg a, reg b) { return vminq_f64(a, b); }

    static inline double extract(reg x, size_t idx) {
        alignas(16) double t[2];
        vst1q_f64(t, x);
        return t[idx & 1];
    }

    static inline double horizontal_add(reg v) { return vaddvq_f64(v); }
};

template <> struct SimdTraits<size_t, neon32_t> {
    using reg = uint64x2_t;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;

    static inline reg  set1(size_t x) { return vdupq_n_u64((uint64_t)x); }
    static inline reg  load(const size_t *p) { return vld1q_u64((const uint64_t *)p); }
    static inline reg  loadu(const size_t *p) { return vld1q_u64((const uint64_t *)p); }
    static inline void store(size_t *p, reg v) { vst1q_u64((uint64_t *)p, v); }
    static inline void storeu(size_t *p, reg v) { vst1q_u64((uint64_t *)p, v); }
    static inline void store_stream(size_t *p, reg v) { vst1q_u64((uint64_t *)p, v); }
    static inline reg  zero() { return vdupq_n_u64(0); }
    static inline reg  setzero() { return zero(); }

    static inline reg add(reg a, reg b) { return vaddq_u64(a, b); }
    static inline reg sub(reg a, reg b) { return vsubq_u64(a, b); }

    static inline reg mul(reg a, reg b) {
        uint64_t A[2], B[2], R[2];
        vst1q_u64(A, a);
        vst1q_u64(B, b);
        R[0] = A[0] * B[0];
        R[1] = A[1] * B[1];
        return vld1q_u64(R);
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        uint64_t A[2], B[2], C[2], R[2];
        vst1q_u64(A, a);
        vst1q_u64(B, b);
        vst1q_u64(C, c);
        R[0] = A[0] * B[0] + C[0];
        R[1] = A[1] * B[1] + C[1];
        return vld1q_u64(R);
    }
    static inline reg fma(reg a, reg b, reg c) { return fmadd(a, b, c); }

    static inline reg andnot(reg a, reg b) {
        uint64x2_t na = veorq_u64(a, vdupq_n_u64(~0ULL));
        return vandq_u64(na, b);
    }

    static inline size_t extract(reg x, size_t idx) {
        alignas(16) uint64_t t[2];
        vst1q_u64(t, x);
        return t[idx & 1];
    }

    static inline uint64_t horizontal_add(reg v) {
        uint64_t low = vgetq_lane_u64(v, 0);
        uint64_t high = vgetq_lane_u64(v, 1);
        return low + high;
    }
};

template <> struct SimdTraits<std::complex<float>, neon32_t> {
    using reg = complex_reg_f32;
    static constexpr size_t width = 2;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;

    static inline reg set1(std::complex<float> x) {
        float t[4] = {x.real(), x.imag(), x.real(), x.imag()};
        return {vld1q_f32(t)};
    }
    static inline reg load(const std::complex<float> *p) {
        return {vld1q_f32(reinterpret_cast<const float *>(p))};
    }
    static inline reg loadu(const std::complex<float> *p) {
        return {vld1q_f32(reinterpret_cast<const float *>(p))};
    }
    static inline void store(std::complex<float> *p, reg r) {
        vst1q_f32(reinterpret_cast<float *>(p), r.v);
    }
    static inline void storeu(std::complex<float> *p, reg r) {
        vst1q_f32(reinterpret_cast<float *>(p), r.v);
    }
    static inline reg broadcast(const std::complex<float> *ptr) {
        float32x2_t c = vld1_f32((const float *)ptr);
        return {vcombine_f32(c, c)};
    }
    static inline reg zero() { return {vdupq_n_f32(0.0f)}; }
    static inline reg setzero() { return zero(); }
    static inline reg add(reg a, reg b) { return {vaddq_f32(a.v, b.v)}; }
    static inline reg sub(reg a, reg b) { return {vsubq_f32(a.v, b.v)}; }
    static inline reg mul(reg a, reg b) {
        float32x4_t av = a.v;
        float32x4_t bv = b.v;
        float32x4_t ar = vuzpq_f32(av, av).val[0];
        float32x4_t ai = vuzpq_f32(av, av).val[1];
        float32x4_t br = vuzpq_f32(bv, bv).val[0];
        float32x4_t bi = vuzpq_f32(bv, bv).val[1];
        float32x4_t real = vsubq_f32(vmulq_f32(ar, br), vmulq_f32(ai, bi));
        float32x4_t imag = vaddq_f32(vmulq_f32(ar, bi), vmulq_f32(ai, br));
        return {vzip1q_f32(real, imag)};
    }
    static inline reg fmadd(reg a, reg b, reg c) {
        reg prod = mul(a, b);
        return add(prod, c);
    }
    static inline reg                 fma(reg a, reg b, reg c) { return fmadd(a, b, c); }
    static inline std::complex<float> extract(reg x, size_t idx) {
        alignas(16) float t[4];
        vst1q_f32(t, x.v);
        return std::complex<float>(t[idx * 2], t[idx * 2 + 1]);
    }
    static inline std::complex<float> horizontal_add(reg x) {
        float t[4];
        vst1q_f32(t, x.v);
        return {t[0] + t[2], t[1] + t[3]};
    }
    static inline void stream(std::complex<float> *ptr, reg x) {
        vst1q_f32(reinterpret_cast<float *>(ptr), x.v);
    }
    static inline void store_stream(std::complex<float> *ptr, reg x) {
        vst1q_f32(reinterpret_cast<float *>(ptr), x.v);
    }
};

template <> struct SimdTraits<std::complex<double>, neon32_t> {
    using reg = complex_reg_f64;
    static constexpr size_t width = 1;
    static constexpr size_t alignment = 16;
    using reg_aligned = aligned_reg<reg, alignment>;

    static inline reg set1(std::complex<double> x) {
        double t[2] = {x.real(), x.imag()};
        return {vld1q_f64(t)};
    }
    static inline reg load(const std::complex<double> *p) { return {vld1q_f64((const double *)p)}; }
    static inline reg loadu(const std::complex<double> *p) {
        return {vld1q_f64((const double *)p)};
    }
    static inline void store(std::complex<double> *p, reg r) { vst1q_f64((double *)p, r.v); }
    static inline void storeu(std::complex<double> *p, reg r) { vst1q_f64((double *)p, r.v); }
    static inline reg  setzero() { return {vdupq_n_f64(0.0)}; }
    static inline reg  add(reg a, reg b) { return {vaddq_f64(a.v, b.v)}; }
    static inline reg  sub(reg a, reg b) { return {vsubq_f64(a.v, b.v)}; }
    static inline reg  mul(reg a, reg b) {
        double ar = vgetq_lane_f64(a.v, 0);
        double ai = vgetq_lane_f64(a.v, 1);
        double br = vgetq_lane_f64(b.v, 0);
        double bi = vgetq_lane_f64(b.v, 1);
        double rr = ar * br - ai * bi;
        double ri = ar * bi + ai * br;
        double res[2] = {rr, ri};
        return {vld1q_f64(res)};
    }
    static inline reg                  fmadd(reg a, reg b, reg c) { return add(mul(a, b), c); }
    static inline reg                  fma(reg a, reg b, reg c) { return fmadd(a, b, c); }
    static inline std::complex<double> extract(reg x, size_t idx = 0) {
        double t[2];
        vst1q_f64(t, x.v);
        return {t[0], t[1]};
    }
    static inline std::complex<double> horizontal_add(reg v) { return extract(v, 0); }

    static inline void stream(std::complex<double> *ptr, reg x) {
        vst1q_f64(reinterpret_cast<double *>(ptr), x.v);
    }
    static inline void store_stream(std::complex<double> *ptr, reg x) {
        vst1q_f64(reinterpret_cast<double *>(ptr), x.v);
    }
};

} // namespace simd

namespace detail {

inline float reduce_sum(float32x4_t v) { return vaddvq_f32(v); }

inline double reduce_sum(float64x2_t v) { return vaddvq_f64(v); }

inline uint64_t reduce_sum(uint64x2_t v) { return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1); }

inline std::complex<float> reduce_sum(complex_reg_f32 w) {
    return simd::SimdTraits<std::complex<float>, neon32_t>::horizontal_add(w);
}

inline std::complex<double> reduce_sum(complex_reg_f64 w) {
    return simd::SimdTraits<std::complex<double>, neon32_t>::horizontal_add(w);
}
} // namespace detail

#endif // TENSORIUM_ARM
