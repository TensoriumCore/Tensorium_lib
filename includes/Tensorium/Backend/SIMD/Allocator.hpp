#pragma once
#include "SIMD.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#ifdef USE_NUMA
#    include <numa.h>
#    include <numaif.h>
#    include <sched.h> // for sched_getcpu()
#endif
#if defined(USE_KNL)
#    include <hbwmalloc.h>
#endif
/**
 * @brief Aligned memory allocator for high-performance computing.
 *
 * This allocator provides memory aligned to a specified boundary, ensuring compatibility with SIMD
 * instructions and optimal cache usage. On Intel KNL (Knights Landing) architectures, this
 * allocator automatically uses the high-bandwidth MCDRAM via `hbw_posix_memalign` if the macro
 * `USE_KNL` is defined. Otherwise, standard `posix_memalign` is used.
 *
 * This is particularly useful for vectorized numerical libraries, where memory alignment is
 * essential for instruction-level parallelism (e.g., AVX, SSE, AVX-512).
 *
 * @tparam T         Type of the objects being allocated.
 * @tparam Alignment Memory alignment in bytes. Must be a power of two and compatible with the ISA
 * in use (e.g., 32 for AVX256).
 */
template <typename T, std::size_t Alignment> struct AlignedAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
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
    template <typename U> struct rebind {
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
    AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

    /**
     * @brief Allocates aligned memory for n elements of type T.
     *
     * The alignment is guaranteed to be at least `Alignment` bytes. On KNL platforms,
     * high-bandwidth memory (HBM) will be used via libhbw.
     *
     * On NUMA systems, memory is allocated on the node local to the calling thread.
     *
     * @param n Number of elements to allocate.
     * @param node_id NUMA node to allocate on (-1 = auto/local or no NUMA).
     * @return T* Pointer to aligned memory block.
     *
     * @throws std::bad_alloc If memory allocation fails.
     */
    [[nodiscard]] T *allocate(std::size_t n, int node_id = -1) {
        void             *ptr = nullptr;
        const std::size_t alloc_size = n * sizeof(T) + Alignment;

        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two.");
        static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T).");

#if defined(USE_KNL)
        if (hbw_posix_memalign(&ptr, Alignment, alloc_size) != 0)
            throw std::bad_alloc();

#elif defined(USE_NUMA)
    if (numa_available() >= 0) {
        int target_node = node_id;
        if (target_node < 0) {
            int cpu = sched_getcpu();
            target_node = numa_node_of_cpu(cpu);
        }

        long node_mem = numa_node_size64(target_node, nullptr);
        if (node_mem <= 0) {
            for (int n = 0; n <= numa_max_node(); ++n) {
                if (numa_node_size64(n, nullptr) > 0) {
                    target_node = n;
                    break;
                }
            }
            // std::cerr << "[NUMA WARN] Node " << target_node
            //           << " has no memory, fallback to node " << target_node << "\n";
        }

        ptr = numa_alloc_onnode(alloc_size, target_node);
        if (!ptr) {
            std::cerr << "[NUMA WARN] numa_alloc_onnode failed, fallback to posix_memalign()\n";
            if (posix_memalign(&ptr, Alignment, alloc_size) != 0)
                throw std::bad_alloc();
        }
    } else if (posix_memalign(&ptr, Alignment, alloc_size) != 0) {
        throw std::bad_alloc();
    }

#else
        if (posix_memalign(&ptr, Alignment, alloc_size) != 0)
            throw std::bad_alloc();
#endif
        return reinterpret_cast<T *>(ptr);
    }

    /**
     * @brief Deallocates memory previously allocated with allocate().
     *
     * On KNL platforms, this calls `hbw_free`. On NUMA systems, it uses `numa_free()`. Otherwise,
     * standard `free()` is used.
     *
     * @param p Pointer to memory to deallocate.
     * @param n Number of elements.
     */
    void deallocate(T *p, std::size_t n) noexcept {
#if defined(USE_KNL)
        hbw_free(p);
#elif defined(USE_NUMA)
        if (numa_available() >= 0)
            numa_free(p, n * sizeof(T));
        else
            free(p);
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
template <typename K> using aligned_vector = std::vector<K, AlignedAllocator<K, ALIGN>>;

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
bool operator==(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<T, Alignment> &) noexcept {
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
bool operator!=(const AlignedAllocator<T, Alignment> &,
                const AlignedAllocator<T, Alignment> &) noexcept {
    return false;
}
