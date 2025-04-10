#pragma once
#include <immintrin.h>
#include <cstddef>
#include <stdexcept>

struct sse_t    { static constexpr size_t width = 4;  using reg = __m128;  static constexpr size_t alignment = 16; };
struct avx2_t   { static constexpr size_t width = 8;  using reg = __m256;  static constexpr size_t alignment = 32; };
struct avx512_t { static constexpr size_t width = 16; using reg = __m512;  static constexpr size_t alignment = 64; };

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

template<typename F>
void dispatch_simd(F&& f) {
    if (supports_avx512()) {
        f(avx512_t{});
    } else if (supports_avx2()) {
        f(avx2_t{});
    } else if (supports_sse()) {
        f(sse_t{});
    } else {
        throw std::runtime_error("No supported SIMD ISA (SSE/AVX2/AVX512).");
    }
}


dispatch_simd([](auto simd) {
    using T = decltype(simd);
    constexpr size_t W = T::width;
    constexpr size_t A = T::alignment;
    typename T::reg v = {}; 

});
