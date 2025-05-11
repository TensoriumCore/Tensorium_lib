#pragma once
#include "SIMD.hpp"
#include <iostream>
#include <cmath>
#include <vector>

#if defined(USE_KNL)
#include <hbwmalloc.h>
#endif
/**
 * @brief Aligned memory allocator for high-performance computing.
 * 
 * This allocator provides memory aligned to a specified boundary, ensuring compatibility with SIMD instructions and optimal cache usage.
 * On Intel KNL (Knights Landing) architectures, this allocator automatically uses the high-bandwidth MCDRAM via `hbw_posix_memalign` 
 * if the macro `USE_KNL` is defined. Otherwise, standard `posix_memalign` is used.
 * 
 * This is particularly useful for vectorized numerical libraries, where memory alignment is essential for
 * instruction-level parallelism (e.g., AVX, SSE, AVX-512).
 * 
 * @tparam T         Type of the objects being allocated.
 * @tparam Alignment Memory alignment in bytes. Must be a power of two and compatible with the ISA in use (e.g., 32 for AVX256).
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	/**
	 * @brief Rebinding structure for allocator traits.
	 * 
	 * Allows conversion of an AlignedAllocator<T, Alignment> to AlignedAllocator<U, Alignment>,
	 * which is required by STL containers during type conversions.
	 * 
	 * @tparam U New type for rebind.
	 */
	template <typename U>
	struct rebind {
		using other = AlignedAllocator<U, Alignment>;
	};
	/**
	 * @brief Conversion constructor from another allocator of different type.
	 * 
	 * Required by the STL allocator model. Does nothing as this allocator is stateless.
	 * 
	 * @tparam U Other type.
	 * @param other The other allocator.
	 */
	AlignedAllocator() noexcept = default;
	template <typename U>
	/**
	 * @brief Default constructor.
	 * 
	 * Stateless and noexcept.
	 */
	AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
	/**
	 * @brief Allocates aligned memory for n elements of type T.
	 * 
	 * The alignment is guaranteed to be at least `Alignment` bytes. On KNL platforms,
	 * high-bandwidth memory (HBM) will be used via libhbw.
	 * 
	 * @param n Number of elements to allocate.
	 * @return T* Pointer to aligned memory block.
	 * 
	 * @throws std::bad_alloc If memory allocation fails.
	 */
	[[nodiscard]] T* allocate(std::size_t n) {
		void* ptr = nullptr;
#if defined(USE_KNL)
		if (hbw_posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
			throw std::bad_alloc();
#else
		if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
			throw std::bad_alloc();
#endif
		return reinterpret_cast<T*>(ptr);
	}
	/**
	 * @brief Deallocates memory previously allocated with allocate().
	 * 
	 * On KNL platforms, this calls `hbw_free`. Otherwise, standard `free` is used.
	 * 
	 * @param p Pointer to memory to deallocate.
	 * @param size Number of elements (not used).
	 */
	void deallocate(T* p, std::size_t) noexcept {
#if defined(USE_KNL)
		hbw_free(p);
#else
		free(p);
#endif
	}
};
/**
 * @brief Type alias for a std::vector with aligned memory allocation.
 * 
 * This provides an aligned vector container compatible with SIMD usage.
 * The alignment used is determined by the macro `ALIGN`, typically set
 * based on the SIMD instruction set width (e.g., 16 for SSE, 32 for AVX, 64 for AVX-512).
 * 
 * @tparam K Type of the elements.
 */
template<typename K>
using aligned_vector = std::vector<K, AlignedAllocator<K, ALIGN>>;

/**
 * @brief Equality operator for AlignedAllocator.
 * 
 * Always returns true as the allocator is stateless and does not manage
 * any per-instance resources.
 * 
 * @tparam T Type of allocated elements.
 * @tparam Alignment Alignment in bytes.
 * 
 * @return true
 */
template <typename T, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<T, Alignment>&) noexcept {
    return true;
}
/**
 * @brief Inequality operator for AlignedAllocator.
 * 
 * Always returns false, as there are no distinguishing stateful properties.
 * 
 * @tparam T Type of allocated elements.
 * @tparam Alignment Alignment in bytes.
 * 
 * @return false
 */
template <typename T, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<T, Alignment>&) noexcept {
    return false;
}

