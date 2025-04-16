#pragma once
#include <iostream>
#include <cmath>
#include <vector>

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

