#pragma once

#if defined(TENSORIUM_USE_MPI) && !defined(TENSORIUM_ENABLE_MPI)
#    define TENSORIUM_ENABLE_MPI
#endif

/**
 * @file MPIContext.hpp
 * @brief RAII wrapper for MPI initialization and communicator management.
 * @details
 * Provides a singleton-style context that initializes MPI with thread support
 * (MPI_THREAD_FUNNELED by default, suitable for OpenMP+MPI hybrid).
 * The context is automatically finalized when the last reference is destroyed.
 *
 * Usage:
 * @code
 *   int main(int argc, char** argv) {
 *       tensorium::mpi::MPIContext ctx(argc, argv);
 *       // ... MPI code ...
 *   } // MPI_Finalize called here
 * @endcode
 */

#ifdef TENSORIUM_ENABLE_MPI
#    include <mpi.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <string>

namespace tensorium::mpi {

#ifdef TENSORIUM_ENABLE_MPI

/**
 * @brief Thread safety level for MPI initialization.
 */
enum class ThreadLevel {
    Single = MPI_THREAD_SINGLE,
    Funneled = MPI_THREAD_FUNNELED,
    Serialized = MPI_THREAD_SERIALIZED,
    Multiple = MPI_THREAD_MULTIPLE
};

/**
 * @brief RAII wrapper for MPI lifecycle management.
 */
class MPIContext {
  public:
    /**
     * @brief Initialize MPI with specified thread support.
     * @param argc Command line argument count.
     * @param argv Command line arguments.
     * @param required Thread level required (default: Funneled for OpenMP+MPI).
     */
    MPIContext(int &argc, char **&argv, ThreadLevel required = ThreadLevel::Funneled) {
        int provided = 0;
        int err = MPI_Init_thread(&argc, &argv, static_cast<int>(required), &provided);
        if (err != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed with error code " +
                                     std::to_string(err));
        }

        provided_level_ = static_cast<ThreadLevel>(provided);
        if (provided < static_cast<int>(required)) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                fprintf(stderr, "[MPIContext] Warning: requested thread level %d, got %d\n",
                        static_cast<int>(required), provided);
            }
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        initialized_ = true;
    }

    ~MPIContext() {
        if (initialized_) {
            MPI_Finalize();
            initialized_ = false;
        }
    }

    MPIContext(const MPIContext &) = delete;
    MPIContext &operator=(const MPIContext &) = delete;
    MPIContext(MPIContext &&) = delete;
    MPIContext &operator=(MPIContext &&) = delete;

    /// @brief Get the rank in MPI_COMM_WORLD.
    int world_rank() const noexcept { return world_rank_; }

    /// @brief Get the size of MPI_COMM_WORLD.
    int world_size() const noexcept { return world_size_; }

    /// @brief Check if this is the root process (rank 0).
    bool is_root() const noexcept { return world_rank_ == 0; }

    /// @brief Get the provided thread level.
    ThreadLevel provided_level() const noexcept { return provided_level_; }

    /// @brief Barrier on MPI_COMM_WORLD.
    void barrier() const { MPI_Barrier(MPI_COMM_WORLD); }

    /// @brief Print message only from root process.
    template <typename... Args> void root_printf(const char *fmt, Args... args) const {
        if (is_root()) {
            if constexpr (sizeof...(args) == 0) {
                fputs(fmt, stdout);
            } else {
                printf(fmt, args...);
            }
            fflush(stdout);
        }
    }

    /// @brief Abort all MPI processes.
    [[noreturn]] void abort(int errorcode = 1) const {
        MPI_Abort(MPI_COMM_WORLD, errorcode);
        std::terminate(); // Should never reach here
    }

  private:
    int         world_rank_ = 0;
    int         world_size_ = 1;
    ThreadLevel provided_level_ = ThreadLevel::Single;
    bool        initialized_ = false;
};

#else // !TENSORIUM_ENABLE_MPI

/**
 * @brief Stub implementation when MPI is disabled.
 */
enum class ThreadLevel { Single, Funneled, Serialized, Multiple };

class MPIContext {
  public:
    MPIContext(int &, char **&, ThreadLevel = ThreadLevel::Funneled) {}
    ~MPIContext() = default;

    MPIContext(const MPIContext &) = delete;
    MPIContext &operator=(const MPIContext &) = delete;

    int         world_rank() const noexcept { return 0; }
    int         world_size() const noexcept { return 1; }
    bool        is_root() const noexcept { return true; }
    ThreadLevel provided_level() const noexcept { return ThreadLevel::Single; }
    void        barrier() const {}

    template <typename... Args> void root_printf(const char *fmt, Args... args) const {
        printf(fmt, args...);
        fflush(stdout);
    }

    [[noreturn]] void abort(int = 1) const { std::terminate(); }
};

#endif // TENSORIUM_ENABLE_MPI

} // namespace tensorium::mpi
