/**
 * @file MemoryPoolDemo.cpp
 * @brief Demonstration of memory pool usage for BSSN/FMR simulations.
 *
 * This file shows how to integrate the memory pooling system into a
 * moving puncture simulation for improved performance.
 *
 * Build:
 *   clang++ -std=c++17 -O3 -I../../../includes MemoryPoolDemo.cpp -o memory_pool_demo
 *
 * Run:
 *   ./memory_pool_demo
 */

#include "Tensorium_Grid/Grid/MemoryPool.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNPooledGrid.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/FMR/FMRMemoryManager.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace tensorium_RG;
using namespace tensorium_RG::z4c::fmr;

// ============================================================================
// Benchmark utilities
// ============================================================================

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Demo 1: Basic Memory Pool Usage
// ============================================================================

void demo_basic_pool() {
    printf("\n=== Demo 1: Basic Memory Pool Usage ===\n");

    // Create a pool
    MemoryPool<double> pool;

    // Pre-allocate for known sizes
    const size_t field_size = 100 * 100 * 100;  // 1M elements
    pool.preallocate({{field_size, 10}});       // 10 fields of this size

    printf("Pre-allocated pool for %zu-element fields\n", field_size);
    pool.print_stats();

    // Allocate and deallocate in a loop (simulating RK4 stages)
    Timer timer;
    const int num_iterations = 100;

    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        // Allocate 4 fields (like RK4 stages)
        std::vector<double*> fields;
        for (int j = 0; j < 4; ++j) {
            fields.push_back(pool.allocate(field_size));
        }

        // "Use" the fields
        for (auto* ptr : fields) {
            ptr[0] = 1.0;
            ptr[field_size - 1] = 2.0;
        }

        // Release fields
        for (auto* ptr : fields) {
            pool.deallocate(ptr, field_size);
        }
    }
    double pooled_time = timer.elapsed_ms();

    printf("Pooled allocation: %d iterations in %.2f ms (%.3f ms/iter)\n",
           num_iterations, pooled_time, pooled_time / num_iterations);

    // Compare with direct allocation
    timer.start();
    for (int i = 0; i < num_iterations; ++i) {
        std::vector<double*> fields;
        for (int j = 0; j < 4; ++j) {
            fields.push_back(static_cast<double*>(
                ::operator new[](field_size * sizeof(double), std::align_val_t(64))));
        }

        for (auto* ptr : fields) {
            ptr[0] = 1.0;
            ptr[field_size - 1] = 2.0;
        }

        for (auto* ptr : fields) {
            ::operator delete[](ptr, std::align_val_t(64));
        }
    }
    double direct_time = timer.elapsed_ms();

    printf("Direct allocation: %d iterations in %.2f ms (%.3f ms/iter)\n",
           num_iterations, direct_time, direct_time / num_iterations);

    printf("Speedup: %.2fx\n", direct_time / pooled_time);

    pool.print_stats();
}

// ============================================================================
// Demo 2: Grid Memory Pool for FMR
// ============================================================================

void demo_grid_pool() {
    printf("\n=== Demo 2: Grid Memory Pool for FMR ===\n");

    // Create a grid memory pool
    GridMemoryPool<double> pool;

    // Pre-allocate for FMR hierarchy
    const size_t root_n = 64;
    const size_t ng = 4;
    const size_t num_levels = 3;
    pool.preallocate_fmr_hierarchy(root_n, ng, num_levels);

    printf("Pre-allocated FMR hierarchy: root=%zu, levels=%zu\n", root_n, num_levels);
    pool.print_stats();

    // Simulate creating pooled grids
    Timer timer;
    const int num_regrids = 10;

    timer.start();
    for (int regrid = 0; regrid < num_regrids; ++regrid) {
        // Create a pooled grid for each level
        std::vector<std::unique_ptr<BSSNPooledGridSoA<double>>> grids;

        for (size_t level = 0; level < num_levels; ++level) {
            const size_t n = root_n * (1 << level);  // Double per level
            grids.push_back(std::make_unique<BSSNPooledGridSoA<double>>(
                pool, n, n, n, ng, 1.0 / n, 1.0 / n, 1.0 / n));
        }

        // "Evolve" the grids
        for (auto& grid : grids) {
            const size_t n = grid->total_cells();
            grid->alpha.ptr()[0] = 1.0;
            grid->alpha.ptr()[n - 1] = 1.0;
        }

        // Grids are automatically returned to pool on destruction
    }
    double pooled_time = timer.elapsed_ms();

    printf("Pooled grid creation: %d regrids in %.2f ms (%.3f ms/regrid)\n",
           num_regrids, pooled_time, pooled_time / num_regrids);

    pool.print_stats();
}

// ============================================================================
// Demo 3: FMR Memory Manager with Epochs
// ============================================================================

void demo_fmr_manager() {
    printf("\n=== Demo 3: FMR Memory Manager with Epochs ===\n");

    // Create an FMR memory manager
    auto manager = make_moving_puncture_memory_manager<double>(64, 3);
    manager.print_stats();

    // Simulate time stepping with epochs
    const int num_steps = 50;
    Timer timer;

    timer.start();
    for (int step = 0; step < num_steps; ++step) {
        // Begin epoch (could enable memory recycling)
        EpochGuard<double> guard(manager);

        // Allocate fields for this time step
        std::vector<double*> fields;
        for (size_t level = 0; level < 3; ++level) {
            for (int stage = 0; stage < 4; ++stage) {  // RK4 stages
                fields.push_back(manager.allocate_level_field(level));
            }
        }

        // "Evolve"
        for (auto* ptr : fields) {
            ptr[0] = static_cast<double>(step);
        }

        // Deallocate
        for (size_t level = 0; level < 3; ++level) {
            for (int stage = 0; stage < 4; ++stage) {
                manager.deallocate_level_field(fields[level * 4 + stage], level);
            }
        }
    }
    double total_time = timer.elapsed_ms();

    printf("Simulated %d time steps in %.2f ms (%.3f ms/step)\n",
           num_steps, total_time, total_time / num_steps);

    manager.print_stats();
}

// ============================================================================
// Demo 4: Pool Reset for Temporal Reuse
// ============================================================================

void demo_pool_reset() {
    printf("\n=== Demo 4: Pool Reset for Temporal Reuse ===\n");

    MemoryPool<double> pool;
    const size_t field_size = 50 * 50 * 50;  // 125K elements
    pool.preallocate({{field_size, 20}});

    // Allocate, use, then reset - simulating temporal reuse
    Timer timer;
    const int num_epochs = 100;

    timer.start();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Allocate all fields for this epoch
        std::vector<double*> fields;
        for (int i = 0; i < 20; ++i) {
            fields.push_back(pool.allocate(field_size));
        }

        // Use fields
        for (auto* ptr : fields) {
            ptr[0] = static_cast<double>(epoch);
        }

        // Reset instead of individual deallocation
        pool.reset_all();
    }
    double reset_time = timer.elapsed_ms();

    printf("Reset-based recycling: %d epochs in %.2f ms (%.3f ms/epoch)\n",
           num_epochs, reset_time, reset_time / num_epochs);

    // Compare with individual deallocation
    timer.start();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::vector<double*> fields;
        for (int i = 0; i < 20; ++i) {
            fields.push_back(pool.allocate(field_size));
        }

        for (auto* ptr : fields) {
            ptr[0] = static_cast<double>(epoch);
        }

        // Individual deallocation
        for (auto* ptr : fields) {
            pool.deallocate(ptr, field_size);
        }
    }
    double dealloc_time = timer.elapsed_ms();

    printf("Individual dealloc: %d epochs in %.2f ms (%.3f ms/epoch)\n",
           num_epochs, dealloc_time, dealloc_time / num_epochs);

    printf("Reset speedup: %.2fx\n", dealloc_time / reset_time);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Memory Pool Demonstration for BSSN/FMR\n");
    printf("========================================\n");

    demo_basic_pool();
    demo_grid_pool();
    demo_fmr_manager();
    demo_pool_reset();

    printf("\n========================================\n");
    printf("All demos completed successfully!\n");
    printf("========================================\n");

    return 0;
}
