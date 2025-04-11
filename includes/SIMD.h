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
		_mm_storeu_pd(r, sum);
		return r[0] + r[1];
	}

}


template <typename T, std::size_t Alignment>
struct AlignedAllocator {
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	template <typename U>
		struct rebind {
			using other = AlignedAllocator<U, Alignment>;
		};

	AlignedAllocator() noexcept = default;
	template <typename U>
		AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

	[[nodiscard]] T* allocate(std::size_t n) {
		void* ptr = nullptr;
		if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
			throw std::bad_alloc();
		return reinterpret_cast<T*>(ptr);
	}

	void deallocate(T* p, std::size_t) noexcept {
		free(p);
	}
};

template<typename K>
using aligned_vector = std::vector<K, AlignedAllocator<K, 32>>;


namespace simd {

	template<typename T> struct SimdTraits;

	template<>
		struct SimdTraits<float> {
			using reg = __m256;
			static constexpr size_t width = 8;
			static inline reg set1(float x)				{ return _mm256_set1_ps(x); }
			static inline reg load(const float* ptr)	{ return _mm256_load_ps(ptr); }
			static inline void store(float* ptr, reg x)	{ _mm256_store_ps(ptr, x); }
			static inline reg zero()					{ return _mm256_setzero_ps(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm256_fmadd_ps(a, b, c); }
			static inline reg add(reg a, reg b)			{ return _mm256_add_ps(a, b); }
			static inline reg mul(reg a, reg b)			{ return _mm256_mul_ps(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm256_sub_ps(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm256_andnot_ps(a, b); }
			static inline void store_stream(float* ptr, reg x) { _mm256_stream_ps(ptr, x); }
		};

	template<>
		struct SimdTraits<double> {
			using reg = __m256d;
			static constexpr size_t width = 4;
			static inline reg set1(double x)            { return _mm256_set1_pd(x); }
			static inline reg load(const double* ptr)   { return _mm256_load_pd(ptr); }
			static inline void store(double* ptr, reg x){ _mm256_store_pd(ptr, x); }
			static inline reg zero()                    { return _mm256_setzero_pd(); }
			static inline reg fmadd(reg a, reg b, reg c){ return _mm256_fmadd_pd(a, b, c); }
			static inline reg add(reg a, reg b)         { return _mm256_add_pd(a, b); }
			static inline reg mul(reg a, reg b)         { return _mm256_mul_pd(a, b); }
			static inline reg sub(reg a, reg b)			{ return _mm256_sub_pd(a, b); }
			static inline reg andnot(reg a, reg b)		{ return _mm256_andnot_pd(a, b); }
			static inline void store_stream(double* ptr, reg x) { _mm256_stream_pd(ptr, x); }
		};

} 
