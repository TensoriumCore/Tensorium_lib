// CudaAllocator.hpp
#pragma once
#include <cuda_runtime.h>
#include <new>
#include <cstddef>
#include <iostream>
#include <vector>

template <typename T> struct CudaUnifiedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind {
        using other = CudaUnifiedAllocator<U>;
    };

    CudaUnifiedAllocator() noexcept = default;
    template <typename U>
    CudaUnifiedAllocator(const CudaUnifiedAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;

        T* ptr = nullptr;
        std::size_t bytes = n * sizeof(T);

        cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] cudaMallocManaged failed: "
                      << cudaGetErrorString(err) << "\n";
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (!p) return;
        cudaError_t err = cudaFree(p);
        if (err != cudaSuccess) {
            // en release on peut silencieusement ignorer
            std::cerr << "[CUDA] cudaFree failed: "
                      << cudaGetErrorString(err) << "\n";
        }
    }
};

template <typename T, typename U>
bool operator==(const CudaUnifiedAllocator<T>&, const CudaUnifiedAllocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
bool operator!=(const CudaUnifiedAllocator<T>&, const CudaUnifiedAllocator<U>&) noexcept {
    return false;
}

// alias pratique
template <typename K>
using cuda_aligned_vector = std::vector<K, CudaUnifiedAllocator<K>>;

