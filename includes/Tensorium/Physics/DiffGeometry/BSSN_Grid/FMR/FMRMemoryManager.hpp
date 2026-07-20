#pragma once

#include "../Fields/BSSNPooledGrid.hpp"
#include "BSSNFixedMeshRefinement.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <tuple>
#include <vector>

/**
 * @file FMRMemoryManager.hpp
 * @brief Memory management utilities for FMR hierarchies.
 * @details
 * This header provides memory pooling integration for Fixed Mesh Refinement
 * hierarchies. It manages memory allocation across multiple refinement levels,
 * enabling efficient reuse during regridding and time stepping.
 *
 * Key features:
 * - Pre-allocation based on hierarchy configuration
 * - Epoch-based memory recycling
 * - Statistics tracking for memory usage analysis
 * - Thread-safe allocation for parallel FMR operations
 */

namespace tensorium_RG::bssn::fmr {

// ============================================================================
// FMR Memory Configuration
// ============================================================================

/**
 * @brief Configuration for FMR memory pool allocation.
 */
struct FMRMemoryConfig {
    size_t root_nx = 96;           ///< Root level x dimension
    size_t root_ny = 96;           ///< Root level y dimension
    size_t root_nz = 96;           ///< Root level z dimension
    size_t ghost_cells = 4;        ///< Number of ghost cells
    size_t num_levels = 3;         ///< Number of refinement levels
    size_t refinement_ratio = 2;   ///< Refinement factor per level
    size_t grids_per_level = 4;    ///< Expected grids per level (for RK4 stages)
    bool include_scratch = true;   ///< Allocate scratch buffers
    bool include_rhs = true;       ///< Allocate RHS buffers

    /**
     * @brief Calculate total field count per grid.
     */
    [[nodiscard]] size_t fields_per_grid() const noexcept {
        size_t count = 28;  // Base evolved fields
        count += 6;         // Ricci cache
        if (include_rhs) count += 28;  // RHS buffers
        if (include_scratch) count += 6;  // Scratch for constraints
        return count;
    }

    /**
     * @brief Calculate total memory needed in bytes.
     */
    template <typename T>
    [[nodiscard]] size_t total_memory_bytes() const noexcept {
        size_t total = 0;
        size_t nx = root_nx, ny = root_ny, nz = root_nz;

        for (size_t level = 0; level < num_levels; ++level) {
            const size_t nx_tot = nx + 2 * ghost_cells;
            const size_t ny_tot = ny + 2 * ghost_cells;
            const size_t nz_tot = pad_simd<T>(nz + 2 * ghost_cells);
            const size_t field_size = nx_tot * ny_tot * nz_tot;

            total += field_size * fields_per_grid() * grids_per_level * sizeof(T);

            nx *= refinement_ratio;
            ny *= refinement_ratio;
            nz *= refinement_ratio;
        }

        return total;
    }
};

// ============================================================================
// FMR Memory Manager
// ============================================================================

/**
 * @brief Manages memory allocation for FMR hierarchies.
 * @tparam T Element type (typically double).
 *
 * The memory manager pre-allocates pools based on expected hierarchy
 * configuration and provides efficient allocation/deallocation throughout
 * the simulation lifetime.
 */
template <typename T>
class FMRMemoryManager {
public:
    using Pool = GridMemoryPool<T>;

    FMRMemoryManager() = default;

    /**
     * @brief Initialize memory manager with FMR configuration.
     */
    explicit FMRMemoryManager(const FMRMemoryConfig& config)
        : config_(config) {
        initialize_pool();
    }

    /**
     * @brief Reconfigure and re-initialize the pool.
     */
    void configure(const FMRMemoryConfig& config) {
        config_ = config;
        initialize_pool();
    }

    /**
     * @brief Get the memory pool.
     */
    Pool& pool() { return pool_; }
    const Pool& pool() const { return pool_; }

    /**
     * @brief Allocate a field for a specific level.
     * @param level FMR level (0 = coarsest).
     * @return Pointer to allocated memory.
     */
    T* allocate_level_field(size_t level) {
        const size_t field_size = level_field_size(level);
        return pool_.allocate(field_size);
    }

    /**
     * @brief Deallocate a level field.
     */
    void deallocate_level_field(T* ptr, size_t level) {
        const size_t field_size = level_field_size(level);
        pool_.deallocate(ptr, field_size);
    }

    /**
     * @brief Start a new epoch (time step), enabling memory recycling.
     *
     * Call this at the beginning of each time step to allow the pool
     * to recycle memory from the previous step.
     */
    void begin_epoch() {
        ++current_epoch_;
        epoch_start_time_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief End the current epoch and record statistics.
     */
    void end_epoch() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - epoch_start_time_).count();

        total_epoch_time_us_ += duration;
        ++completed_epochs_;

        // Optionally reset pool for next epoch
        if (reset_per_epoch_) {
            pool_.reset_epoch();
        }
    }

    /**
     * @brief Reset all memory for reuse.
     *
     * This invalidates all outstanding pointers but allows immediate
     * memory reuse. Use with caution.
     */
    void reset() {
        pool_.reset_epoch();
    }

    /**
     * @brief Enable/disable automatic reset at epoch boundaries.
     */
    void set_reset_per_epoch(bool enable) {
        reset_per_epoch_ = enable;
    }

    /**
     * @brief Get current epoch number.
     */
    [[nodiscard]] size_t current_epoch() const noexcept {
        return current_epoch_;
    }

    /**
     * @brief Get average epoch duration in microseconds.
     */
    [[nodiscard]] double average_epoch_time_us() const noexcept {
        if (completed_epochs_ == 0) return 0.0;
        return static_cast<double>(total_epoch_time_us_) / completed_epochs_;
    }

    /**
     * @brief Print memory statistics.
     */
    void print_stats() const {
        printf("[FMRMemoryManager] Configuration:\n");
        printf("  Root grid: %zux%zux%zu, ghost=%zu\n",
               config_.root_nx, config_.root_ny, config_.root_nz, config_.ghost_cells);
        printf("  Levels: %zu, refinement ratio: %zu\n",
               config_.num_levels, config_.refinement_ratio);
        printf("  Expected memory: %.2f MB\n",
               config_.total_memory_bytes<T>() / (1024.0 * 1024.0));
        printf("  Epochs completed: %zu\n", completed_epochs_);
        if (completed_epochs_ > 0) {
            printf("  Avg epoch time: %.2f ms\n", average_epoch_time_us() / 1000.0);
        }
        pool_.print_stats();
    }

    /**
     * @brief Get configuration.
     */
    [[nodiscard]] const FMRMemoryConfig& config() const noexcept {
        return config_;
    }

private:
    void initialize_pool() {
        std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>> grid_configs;

        size_t nx = config_.root_nx;
        size_t ny = config_.root_ny;
        size_t nz = config_.root_nz;

        for (size_t level = 0; level < config_.num_levels; ++level) {
            grid_configs.emplace_back(
                nx, ny, nz, config_.ghost_cells, config_.grids_per_level);

            nx *= config_.refinement_ratio;
            ny *= config_.refinement_ratio;
            nz *= config_.refinement_ratio;
        }

        pool_.preallocate_grids(grid_configs);
        level_sizes_.clear();

        // Cache level sizes
        nx = config_.root_nx;
        ny = config_.root_ny;
        nz = config_.root_nz;
        for (size_t level = 0; level < config_.num_levels; ++level) {
            const size_t nx_tot = nx + 2 * config_.ghost_cells;
            const size_t ny_tot = ny + 2 * config_.ghost_cells;
            const size_t nz_tot = pad_simd<T>(nz + 2 * config_.ghost_cells);
            level_sizes_.push_back(nx_tot * ny_tot * nz_tot);

            nx *= config_.refinement_ratio;
            ny *= config_.refinement_ratio;
            nz *= config_.refinement_ratio;
        }
    }

    [[nodiscard]] size_t level_field_size(size_t level) const {
        if (level < level_sizes_.size()) {
            return level_sizes_[level];
        }
        // Compute on demand for levels beyond initial config
        size_t nx = config_.root_nx;
        size_t ny = config_.root_ny;
        size_t nz = config_.root_nz;
        for (size_t l = 0; l < level; ++l) {
            nx *= config_.refinement_ratio;
            ny *= config_.refinement_ratio;
            nz *= config_.refinement_ratio;
        }
        const size_t nx_tot = nx + 2 * config_.ghost_cells;
        const size_t ny_tot = ny + 2 * config_.ghost_cells;
        const size_t nz_tot = pad_simd<T>(nz + 2 * config_.ghost_cells);
        return nx_tot * ny_tot * nz_tot;
    }

    FMRMemoryConfig config_;
    Pool pool_;
    std::vector<size_t> level_sizes_;

    size_t current_epoch_ = 0;
    size_t completed_epochs_ = 0;
    size_t total_epoch_time_us_ = 0;
    std::chrono::steady_clock::time_point epoch_start_time_;
    bool reset_per_epoch_ = false;
};

// ============================================================================
// RAII Epoch Guard
// ============================================================================

/**
 * @brief RAII guard for epoch lifetime management.
 */
template <typename T>
class EpochGuard {
public:
    explicit EpochGuard(FMRMemoryManager<T>& manager)
        : manager_(manager) {
        manager_.begin_epoch();
    }

    ~EpochGuard() {
        manager_.end_epoch();
    }

    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;

private:
    FMRMemoryManager<T>& manager_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create an FMR memory manager from a MovingPunctureEnvConfig.
 */
template <typename T>
inline FMRMemoryManager<T> make_fmr_memory_manager(
    size_t nx, size_t ny, size_t nz, size_t ng,
    size_t num_levels, size_t refinement_ratio = 2) {

    FMRMemoryConfig config;
    config.root_nx = nx;
    config.root_ny = ny;
    config.root_nz = nz;
    config.ghost_cells = ng;
    config.num_levels = num_levels;
    config.refinement_ratio = refinement_ratio;

    return FMRMemoryManager<T>(config);
}

/**
 * @brief Create an FMR memory manager for a typical moving puncture simulation.
 */
template <typename T>
inline FMRMemoryManager<T> make_moving_puncture_memory_manager(
    size_t grid_n = 96, size_t num_levels = 3) {

    FMRMemoryConfig config;
    config.root_nx = grid_n;
    config.root_ny = grid_n;
    config.root_nz = grid_n;
    config.ghost_cells = 4;
    config.num_levels = num_levels;
    config.refinement_ratio = 2;
    config.grids_per_level = 5;  // Current + 4 RK stages

    return FMRMemoryManager<T>(config);
}

// ============================================================================
// Global FMR Memory Manager
// ============================================================================

/**
 * @brief Get the global FMR memory manager.
 */
template <typename T>
inline FMRMemoryManager<T>& global_fmr_memory_manager() {
    static FMRMemoryManager<T> instance;
    return instance;
}

/**
 * @brief Initialize the global FMR memory manager.
 */
template <typename T>
inline void init_global_fmr_memory(const FMRMemoryConfig& config) {
    global_fmr_memory_manager<T>().configure(config);
}

} // namespace tensorium_RG::bssn::fmr
