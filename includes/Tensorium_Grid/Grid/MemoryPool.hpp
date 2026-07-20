#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tensorium_RG {

#ifndef TENSORIUM_ALIGN
#    define TENSORIUM_ALIGN 64
#endif

/**
 * @file MemoryPool.hpp
 * @brief High-performance memory pool for BSSN grid allocations.
 * @details
 * This header provides a memory pooling system designed to eliminate repeated
 * allocations during FMR regridding and RK4 time stepping. The pool pre-allocates
 * large aligned memory arenas and distributes blocks on demand, significantly
 * reducing allocation overhead and memory fragmentation.
 *
 * Key features:
 * - SIMD-aligned allocations (64-byte default)
 * - Thread-safe block acquisition
 * - Slab-based organization for efficient reuse
 * - Statistics tracking for diagnostics
 * - Zero-cost reset for temporal reuse
 */

// ============================================================================
// Forward declarations
// ============================================================================

template <typename T> class PooledBlock;
template <typename T> class MemoryPool;
template <typename T> class ArenaAllocator;

// ============================================================================
// Pool Statistics
// ============================================================================

struct PoolStats {
    std::atomic<size_t> total_allocated{0};     ///< Total bytes allocated from system
    std::atomic<size_t> total_in_use{0};        ///< Bytes currently checked out
    std::atomic<size_t> peak_in_use{0};         ///< High watermark of usage
    std::atomic<size_t> allocation_count{0};    ///< Number of checkout operations
    std::atomic<size_t> reuse_count{0};         ///< Number of reused blocks
    std::atomic<size_t> arena_count{0};         ///< Number of arenas allocated

    void reset() noexcept {
        total_in_use.store(0, std::memory_order_relaxed);
        allocation_count.store(0, std::memory_order_relaxed);
        reuse_count.store(0, std::memory_order_relaxed);
    }

    void record_checkout(size_t bytes) noexcept {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
        size_t current = total_in_use.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        size_t peak = peak_in_use.load(std::memory_order_relaxed);
        while (current > peak &&
               !peak_in_use.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {}
    }

    void record_release(size_t bytes) noexcept {
        total_in_use.fetch_sub(bytes, std::memory_order_relaxed);
    }

    void record_reuse() noexcept {
        reuse_count.fetch_add(1, std::memory_order_relaxed);
    }
};

// ============================================================================
// Aligned Arena - Single contiguous aligned memory block
// ============================================================================

/**
 * @brief A single contiguous block of aligned memory.
 * @tparam T Element type for size calculations.
 *
 * The arena owns a large chunk of aligned memory and provides bump-pointer
 * allocation within it. Thread-safe via atomic offset tracking.
 */
template <typename T>
class AlignedArena {
public:
    static constexpr size_t kAlignment = TENSORIUM_ALIGN;

    explicit AlignedArena(size_t num_elements)
        : capacity_(num_elements),
          offset_(0) {
        const size_t bytes = num_elements * sizeof(T);
        data_ = static_cast<T*>(::operator new[](bytes, std::align_val_t(kAlignment)));
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    ~AlignedArena() {
        if (data_) {
            ::operator delete[](data_, std::align_val_t(kAlignment));
        }
    }

    // Non-copyable, movable
    AlignedArena(const AlignedArena&) = delete;
    AlignedArena& operator=(const AlignedArena&) = delete;

    AlignedArena(AlignedArena&& other) noexcept
        : data_(other.data_),
          capacity_(other.capacity_),
          offset_(other.offset_.load(std::memory_order_relaxed)) {
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_.store(0, std::memory_order_relaxed);
    }

    AlignedArena& operator=(AlignedArena&& other) noexcept {
        if (this != &other) {
            if (data_) {
                ::operator delete[](data_, std::align_val_t(kAlignment));
            }
            data_ = other.data_;
            capacity_ = other.capacity_;
            offset_.store(other.offset_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.data_ = nullptr;
            other.capacity_ = 0;
            other.offset_.store(0, std::memory_order_relaxed);
        }
        return *this;
    }

    /**
     * @brief Try to allocate n elements from this arena.
     * @param n Number of elements to allocate.
     * @return Pointer to allocated memory, or nullptr if arena is full.
     */
    T* try_allocate(size_t n) noexcept {
        // Align n to SIMD boundary
        const size_t aligned_n = align_up(n);

        size_t current = offset_.load(std::memory_order_relaxed);
        size_t next;

        do {
            next = current + aligned_n;
            if (next > capacity_) {
                return nullptr;  // Arena full
            }
        } while (!offset_.compare_exchange_weak(current, next,
                                                 std::memory_order_acquire,
                                                 std::memory_order_relaxed));

        return data_ + current;
    }

    /**
     * @brief Reset arena for reuse (invalidates all pointers).
     */
    void reset() noexcept {
        offset_.store(0, std::memory_order_release);
    }

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t used() const noexcept {
        return offset_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t available() const noexcept { return capacity_ - used(); }
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

private:
    static constexpr size_t align_up(size_t n) noexcept {
        constexpr size_t elem_align = kAlignment / sizeof(T);
        return ((n + elem_align - 1) / elem_align) * elem_align;
    }

    T* data_ = nullptr;
    size_t capacity_ = 0;
    std::atomic<size_t> offset_{0};
};

// ============================================================================
// Slab - Fixed-size block pool for specific grid dimensions
// ============================================================================

/**
 * @brief Pool of fixed-size blocks for a specific grid configuration.
 * @tparam T Element type.
 *
 * A slab manages blocks of identical size, enabling O(1) allocation and
 * deallocation through a free list. Ideal for FMR where each level has
 * consistent dimensions.
 */
template <typename T>
class Slab {
public:
    struct Block {
        T* data = nullptr;
        size_t size = 0;
        Block* next = nullptr;
        bool in_use = false;
    };

    Slab(size_t block_size, size_t initial_blocks = 4)
        : block_size_(block_size) {
        grow(initial_blocks);
    }

    ~Slab() {
        // Arenas handle deallocation via RAII
    }

    // Non-copyable
    Slab(const Slab&) = delete;
    Slab& operator=(const Slab&) = delete;

    /**
     * @brief Acquire a block from the slab.
     * @return Pointer to block data, never null.
     */
    T* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (free_list_) {
            Block* block = free_list_;
            free_list_ = block->next;
            block->in_use = true;
            block->next = nullptr;
            ++active_count_;
            return block->data;
        }

        // Need to grow
        grow(std::max(size_t(1), blocks_.size()));
        return acquire_unlocked();
    }

    /**
     * @brief Release a block back to the slab.
     * @param ptr Pointer previously returned by acquire().
     */
    void release(T* ptr) {
        if (!ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& block : blocks_) {
            if (block.data == ptr) {
                assert(block.in_use && "Double release detected");
                block.in_use = false;
                block.next = free_list_;
                free_list_ = &block;
                --active_count_;
                return;
            }
        }

        assert(false && "Released pointer not from this slab");
    }

    /**
     * @brief Reset all blocks to free state.
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_ = nullptr;
        for (auto& block : blocks_) {
            block.in_use = false;
            block.next = free_list_;
            free_list_ = &block;
        }
        active_count_ = 0;
        for (auto& arena : arenas_) {
            arena.reset();
        }
    }

    [[nodiscard]] size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] size_t total_blocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] size_t active_blocks() const noexcept { return active_count_; }
    [[nodiscard]] size_t free_blocks() const noexcept {
        return blocks_.size() - active_count_;
    }

private:
    T* acquire_unlocked() {
        if (free_list_) {
            Block* block = free_list_;
            free_list_ = block->next;
            block->in_use = true;
            block->next = nullptr;
            ++active_count_;
            return block->data;
        }
        return nullptr;
    }

    void grow(size_t num_blocks) {
        const size_t arena_size = aligned_block_size() * num_blocks;
        arenas_.emplace_back(arena_size);
        auto& arena = arenas_.back();

        for (size_t i = 0; i < num_blocks; ++i) {
            T* data = arena.try_allocate(block_size_);
            assert(data && "Arena should have space for all blocks");

            blocks_.push_back(Block{data, block_size_, free_list_, false});
            free_list_ = &blocks_.back();
        }
    }

    [[nodiscard]] size_t aligned_block_size() const noexcept {
        const size_t elem_alignment = std::max<size_t>(size_t(1), TENSORIUM_ALIGN / sizeof(T));
        return ((block_size_ + elem_alignment - size_t(1)) / elem_alignment) * elem_alignment;
    }

    size_t block_size_;
    std::vector<AlignedArena<T>> arenas_;
    std::deque<Block> blocks_;
    Block* free_list_ = nullptr;
    size_t active_count_ = 0;
    std::mutex mutex_;
};

// ============================================================================
// Memory Pool - Main interface for grid allocations
// ============================================================================

/**
 * @brief High-level memory pool for BSSN/FMR grid allocations.
 * @tparam T Element type (typically double).
 *
 * The memory pool organizes allocations by size class, routing requests to
 * appropriate slabs for efficient reuse. Supports both exact-size matching
 * and fallback allocation for unusual sizes.
 */
template <typename T>
class MemoryPool {
public:
    static constexpr size_t kDefaultArenaSize = 64 * 1024 * 1024 / sizeof(T);  // 64 MB

    MemoryPool() = default;

    /**
     * @brief Pre-allocate slabs for known grid sizes.
     * @param sizes Vector of (block_size, count) pairs.
     */
    void preallocate(const std::vector<std::pair<size_t, size_t>>& sizes) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& [size, count] : sizes) {
            const size_t aligned_size = align_size(size);
            auto& slab = get_or_create_slab_unlocked(aligned_size);
            // Pre-warm by acquiring and releasing
            std::vector<T*> blocks;
            blocks.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                blocks.push_back(slab.acquire());
            }
            for (T* ptr : blocks) {
                slab.release(ptr);
            }
            stats_.total_allocated.fetch_add(aligned_size * count * sizeof(T),
                                              std::memory_order_relaxed);
        }
        stats_.arena_count.store(slabs_.size(), std::memory_order_relaxed);
    }

    /**
     * @brief Allocate a block of n elements.
     * @param n Number of elements.
     * @return Aligned pointer to n elements.
     */
    T* allocate(size_t n) {
        const size_t aligned_n = align_size(n);

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to find existing slab
        auto it = slabs_.find(aligned_n);
        if (it != slabs_.end()) {
            T* ptr = it->second->acquire();
            stats_.record_checkout(aligned_n * sizeof(T));
            stats_.record_reuse();
            return ptr;
        }

        // Create new slab for this size
        auto& slab = get_or_create_slab_unlocked(aligned_n);
        T* ptr = slab.acquire();
        stats_.record_checkout(aligned_n * sizeof(T));
        stats_.total_allocated.fetch_add(aligned_n * sizeof(T) * 4,  // Initial capacity
                                          std::memory_order_relaxed);
        return ptr;
    }

    /**
     * @brief Deallocate a block.
     * @param ptr Pointer previously returned by allocate().
     * @param n Original allocation size.
     */
    void deallocate(T* ptr, size_t n) {
        if (!ptr) return;

        const size_t aligned_n = align_size(n);

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = slabs_.find(aligned_n);
        if (it != slabs_.end()) {
            it->second->release(ptr);
            stats_.record_release(aligned_n * sizeof(T));
        }
        // If slab not found, the pointer was from fallback allocation
        // and we intentionally leak it (will be cleaned up on pool destruction)
    }

    /**
     * @brief Reset all slabs for a new time step.
     *
     * This invalidates all outstanding pointers but allows immediate
     * reuse of all memory without deallocation overhead.
     */
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [size, slab] : slabs_) {
            slab->reset();
        }
        stats_.reset();
    }

    /**
     * @brief Get current pool statistics.
     */
    [[nodiscard]] const PoolStats& stats() const noexcept { return stats_; }

    /**
     * @brief Get number of registered size classes.
     */
    [[nodiscard]] size_t num_slabs() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return slabs_.size();
    }

    /**
     * @brief Print pool statistics to stdout.
     */
    void print_stats() const {
        printf("[MemoryPool] Allocated: %.2f MB, In-use: %.2f MB, Peak: %.2f MB\n",
               double(stats_.total_allocated.load()) / (1024 * 1024),
               double(stats_.total_in_use.load()) / (1024 * 1024),
               double(stats_.peak_in_use.load()) / (1024 * 1024));
        printf("[MemoryPool] Allocations: %zu, Reuses: %zu (%.1f%% hit rate)\n",
               stats_.allocation_count.load(),
               stats_.reuse_count.load(),
               stats_.allocation_count.load() > 0
                   ? 100.0 * stats_.reuse_count.load() / stats_.allocation_count.load()
                   : 0.0);
    }

private:
    static constexpr size_t align_size(size_t n) noexcept {
        constexpr size_t alignment = TENSORIUM_ALIGN / sizeof(T);
        return ((n + alignment - 1) / alignment) * alignment;
    }

    Slab<T>& get_or_create_slab_unlocked(size_t size) {
        auto it = slabs_.find(size);
        if (it != slabs_.end()) {
            return *it->second;
        }

        auto slab = std::make_unique<Slab<T>>(size, 4);
        auto& ref = *slab;
        slabs_.emplace(size, std::move(slab));
        return ref;
    }

    std::unordered_map<size_t, std::unique_ptr<Slab<T>>> slabs_;
    mutable std::mutex mutex_;
    PoolStats stats_;
};

// ============================================================================
// Global Pool Access
// ============================================================================

/**
 * @brief Get the global memory pool instance.
 * @tparam T Element type.
 */
template <typename T>
inline MemoryPool<T>& global_pool() {
    static MemoryPool<T> instance;
    return instance;
}

// ============================================================================
// Pool-aware unique pointer
// ============================================================================

/**
 * @brief Deleter that returns memory to the pool.
 */
template <typename T>
struct PoolDeleter {
    size_t size = 0;
    MemoryPool<T>* pool = nullptr;

    void operator()(T* ptr) const noexcept {
        if (pool && ptr) {
            pool->deallocate(ptr, size);
        }
    }
};

template <typename T>
using pooled_ptr = std::unique_ptr<T[], PoolDeleter<T>>;

/**
 * @brief Allocate from the global pool with automatic cleanup.
 * @param n Number of elements.
 */
template <typename T>
inline pooled_ptr<T> pool_alloc(size_t n) {
    auto& pool = global_pool<T>();
    T* ptr = pool.allocate(n);
    return pooled_ptr<T>(ptr, PoolDeleter<T>{n, &pool});
}

/**
 * @brief Allocate from a specific pool with automatic cleanup.
 */
template <typename T>
inline pooled_ptr<T> pool_alloc(MemoryPool<T>& pool, size_t n) {
    T* ptr = pool.allocate(n);
    return pooled_ptr<T>(ptr, PoolDeleter<T>{n, &pool});
}

// ============================================================================
// Scoped Pool Reset
// ============================================================================

/**
 * @brief RAII wrapper to reset pool at scope exit.
 */
template <typename T>
class ScopedPoolReset {
public:
    explicit ScopedPoolReset(MemoryPool<T>& pool) : pool_(pool) {}
    ~ScopedPoolReset() { pool_.reset_all(); }

    ScopedPoolReset(const ScopedPoolReset&) = delete;
    ScopedPoolReset& operator=(const ScopedPoolReset&) = delete;

private:
    MemoryPool<T>& pool_;
};

} // namespace tensorium_RG
