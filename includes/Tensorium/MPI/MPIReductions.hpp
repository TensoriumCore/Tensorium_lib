#pragma once

#if defined(TENSORIUM_USE_MPI) && !defined(TENSORIUM_ENABLE_MPI)
#    define TENSORIUM_ENABLE_MPI
#endif

/**
 * @file MPIReductions.hpp
 * @brief Global reduction operations for distributed BSSN simulations.
 * @details
 * Provides MPI wrappers for computing global CFL numbers, constraint norms,
 * and other diagnostics that require communication across all processes.
 */

#ifdef TENSORIUM_ENABLE_MPI
#    include <mpi.h>
#endif

#include "MPIDomain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tensorium::mpi {

#ifdef TENSORIUM_ENABLE_MPI

/**
 * @brief Utility class for MPI reduction operations.
 */
class Reductions {
  public:
    explicit Reductions(const MPIDomain &domain) : domain_(domain) {}

    /// @brief Compute global maximum of a scalar.
    template <typename T> T allreduce_max(T local_val) const {
        T global_val;
        MPI_Allreduce(&local_val, &global_val, 1, mpi_type<T>(), MPI_MAX, domain_.comm());
        return global_val;
    }

    /// @brief Compute global minimum of a scalar.
    template <typename T> T allreduce_min(T local_val) const {
        T global_val;
        MPI_Allreduce(&local_val, &global_val, 1, mpi_type<T>(), MPI_MIN, domain_.comm());
        return global_val;
    }

    /// @brief Compute global sum of a scalar.
    template <typename T> T allreduce_sum(T local_val) const {
        T global_val;
        MPI_Allreduce(&local_val, &global_val, 1, mpi_type<T>(), MPI_SUM, domain_.comm());
        return global_val;
    }

    /// @brief Compute global maximum of an array.
    template <typename T> void allreduce_max(const T *local, T *global, int count) const {
        MPI_Allreduce(local, global, count, mpi_type<T>(), MPI_MAX, domain_.comm());
    }

    /// @brief Compute global minimum of an array.
    template <typename T> void allreduce_min(const T *local, T *global, int count) const {
        MPI_Allreduce(local, global, count, mpi_type<T>(), MPI_MIN, domain_.comm());
    }

    /// @brief Compute global sum of an array.
    template <typename T> void allreduce_sum(const T *local, T *global, int count) const {
        MPI_Allreduce(local, global, count, mpi_type<T>(), MPI_SUM, domain_.comm());
    }

    /**
     * @brief Compute global L2 norm from local sum of squares.
     * @param local_sum_sq Local contribution: sum of (value^2) over local grid.
     * @param local_count Local number of points summed.
     * @return Global L2 norm: sqrt(sum_all(value^2) / total_count).
     */
    template <typename T> T allreduce_l2_norm(T local_sum_sq, size_t local_count) const {
        std::array<double, 2> local_data = {static_cast<double>(local_sum_sq),
                                            static_cast<double>(local_count)};
        std::array<double, 2> global_data;
        MPI_Allreduce(local_data.data(), global_data.data(), 2, MPI_DOUBLE, MPI_SUM,
                      domain_.comm());

        if (global_data[1] > 0) {
            return static_cast<T>(std::sqrt(global_data[0] / global_data[1]));
        }
        return T(0);
    }

    /**
     * @brief Compute global CFL time step from local maximum wave speed.
     * @param local_max_speed Local maximum characteristic speed.
     * @param local_min_dx Local minimum grid spacing (should be same everywhere for uniform grid).
     * @param cfl CFL factor (e.g., 0.25).
     * @return Global CFL-limited time step.
     */
    template <typename T> T compute_global_dt_cfl(T local_max_speed, T local_min_dx, T cfl) const {
        T global_max_speed = allreduce_max(local_max_speed);

        if (global_max_speed <= T(0)) {
            global_max_speed = T(1);
        }

        return cfl * local_min_dx / global_max_speed;
    }

    /**
     * @brief Broadcast a value from root to all processes.
     */
    template <typename T> void broadcast(T &value, int root = 0) const {
        MPI_Bcast(&value, 1, mpi_type<T>(), root, domain_.comm());
    }

    /**
     * @brief Broadcast an array from root to all processes.
     */
    template <typename T> void broadcast(T *data, int count, int root = 0) const {
        MPI_Bcast(data, count, mpi_type<T>(), root, domain_.comm());
    }

  private:
    const MPIDomain &domain_;

    template <typename T> static MPI_Datatype mpi_type() {
        if constexpr (std::is_same_v<T, double>) {
            return MPI_DOUBLE;
        } else if constexpr (std::is_same_v<T, float>) {
            return MPI_FLOAT;
        } else if constexpr (std::is_same_v<T, int>) {
            return MPI_INT;
        } else if constexpr (std::is_same_v<T, long>) {
            return MPI_LONG;
        } else if constexpr (std::is_same_v<T, size_t>) {
            return MPI_UNSIGNED_LONG;
        } else {
            static_assert(std::is_same_v<T, double>, "Unsupported MPI type");
            return MPI_DATATYPE_NULL;
        }
    }
};

/**
 * @brief Container for constraint monitoring statistics (MPI-aware).
 */
struct GlobalConstraintStats {
    double l2_hamiltonian = 0.0;
    double l2_momentum = 0.0;
    double l2_theta = 0.0;
    double l2_Z = 0.0;

    double max_hamiltonian = 0.0;
    double max_momentum = 0.0;
    double max_det_drift = 0.0;
    double max_trace_A = 0.0;

    size_t total_samples = 0;

    /**
     * @brief Reduce local statistics to global across all MPI processes.
     */
    void allreduce(const MPIDomain &domain) {
        Reductions red(domain);

        // Sum for L2 norms (these should be sum of squares before this call)
        double local_sums[4] = {l2_hamiltonian, l2_momentum, l2_theta, l2_Z};
        double global_sums[4];
        red.allreduce_sum(local_sums, global_sums, 4);

        // Max values
        double local_max[4] = {max_hamiltonian, max_momentum, max_det_drift, max_trace_A};
        double global_max[4];
        red.allreduce_max(local_max, global_max, 4);

        // Total sample count
        size_t global_samples = red.allreduce_sum(total_samples);

        // Finalize L2 norms: sqrt(sum / count)
        if (global_samples > 0) {
            double inv_n = 1.0 / static_cast<double>(global_samples);
            l2_hamiltonian = std::sqrt(global_sums[0] * inv_n);
            l2_momentum = std::sqrt(global_sums[1] * inv_n);
            l2_theta = std::sqrt(global_sums[2] * inv_n);
            l2_Z = std::sqrt(global_sums[3] * inv_n);
        }

        max_hamiltonian = global_max[0];
        max_momentum = global_max[1];
        max_det_drift = global_max[2];
        max_trace_A = global_max[3];
        total_samples = global_samples;
    }
};

#else // !TENSORIUM_ENABLE_MPI

/**
 * @brief Stub implementation when MPI is disabled.
 */
class Reductions {
  public:
    explicit Reductions(const MPIDomain &) {}

    template <typename T> T allreduce_max(T val) const { return val; }
    template <typename T> T allreduce_min(T val) const { return val; }
    template <typename T> T allreduce_sum(T val) const { return val; }

    template <typename T> void allreduce_max(const T *local, T *global, int count) const {
        std::copy(local, local + count, global);
    }

    template <typename T> void allreduce_min(const T *local, T *global, int count) const {
        std::copy(local, local + count, global);
    }

    template <typename T> void allreduce_sum(const T *local, T *global, int count) const {
        std::copy(local, local + count, global);
    }

    template <typename T> T allreduce_l2_norm(T local_sum_sq, size_t local_count) const {
        return local_count > 0 ? std::sqrt(local_sum_sq / static_cast<T>(local_count)) : T(0);
    }

    template <typename T> T compute_global_dt_cfl(T local_max_speed, T local_min_dx, T cfl) const {
        T speed = local_max_speed > T(0) ? local_max_speed : T(1);
        return cfl * local_min_dx / speed;
    }

    template <typename T> void broadcast(T &, int = 0) const {}
    template <typename T> void broadcast(T *, int, int = 0) const {}
};

struct GlobalConstraintStats {
    double l2_hamiltonian = 0.0;
    double l2_momentum = 0.0;
    double l2_theta = 0.0;
    double l2_Z = 0.0;

    double max_hamiltonian = 0.0;
    double max_momentum = 0.0;
    double max_det_drift = 0.0;
    double max_trace_A = 0.0;

    size_t total_samples = 0;

    void allreduce(const MPIDomain &) {
        // Single process: convert sum of squares to L2 norm
        if (total_samples > 0) {
            double inv_n = 1.0 / static_cast<double>(total_samples);
            l2_hamiltonian = std::sqrt(l2_hamiltonian * inv_n);
            l2_momentum = std::sqrt(l2_momentum * inv_n);
            l2_theta = std::sqrt(l2_theta * inv_n);
            l2_Z = std::sqrt(l2_Z * inv_n);
        }
    }
};

#endif // TENSORIUM_ENABLE_MPI

} // namespace tensorium::mpi
