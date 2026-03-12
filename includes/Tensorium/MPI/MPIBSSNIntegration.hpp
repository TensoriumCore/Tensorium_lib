#pragma once

/**
 * @file MPIBSSNIntegration.hpp
 * @brief Integration layer between the public Tensorium MPI API and the BSSN grid.
 * @details
 * This header intentionally uses the public `tensorium::mpi` abstractions
 * (`MPIDomain`, `HaloExchanger`, `Reductions`) so the BSSN layer does not mix
 * two independent MPI APIs with incompatible types.
 */

#include "MPIDomain.hpp"
#include "MPIHaloExchange.hpp"
#include "MPIReductions.hpp"

#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/BSSNGridOperations.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>

namespace tensorium::mpi {

/**
 * @brief Create a local BSSNGridSoA for this MPI process's subdomain.
 * @tparam T Floating-point type.
 * @param domain The Cartesian domain decomposition.
 * @param ng Number of ghost cells.
 * @param dx, dy, dz Grid spacing.
 * @param x0_global, y0_global, z0_global Global domain origin.
 * @return Unique pointer to the local grid.
 */
template <typename T>
std::unique_ptr<tensorium_RG::BSSNGridSoA<T>> create_local_bssn_grid(
    const MPIDomain& domain,
    size_t ng,
    T dx, T dy, T dz,
    T x0_global = T(0), T y0_global = T(0), T z0_global = T(0))
{
    auto grid = std::make_unique<tensorium_RG::BSSNGridSoA<T>>(
        domain.local_nx(), domain.local_ny(), domain.local_nz(), ng, dx, dy, dz);

    // Set local origin (physical coordinates)
    grid->x0 = x0_global + static_cast<T>(domain.global_i0()) * dx;
    grid->y0 = y0_global + static_cast<T>(domain.global_j0()) * dy;
    grid->z0 = z0_global + static_cast<T>(domain.global_k0()) * dz;

    return grid;
}

/**
 * @brief Convenience overload using the spacing and origin stored in `MPIDomain`.
 */
template <typename T>
std::unique_ptr<tensorium_RG::BSSNGridSoA<T>> create_local_bssn_grid(const MPIDomain& domain) {
    return create_local_bssn_grid<T>(domain, domain.ng(), static_cast<T>(domain.dx()),
                                     static_cast<T>(domain.dy()), static_cast<T>(domain.dz()),
                                     static_cast<T>(domain.local_x0()) -
                                         static_cast<T>(domain.global_i0()) * static_cast<T>(domain.dx()),
                                     static_cast<T>(domain.local_y0()) -
                                         static_cast<T>(domain.global_j0()) * static_cast<T>(domain.dy()),
                                     static_cast<T>(domain.local_z0()) -
                                         static_cast<T>(domain.global_k0()) * static_cast<T>(domain.dz()));
}

/**
 * @brief BSSN-specific halo exchanger backed by `tensorium::mpi::HaloExchanger`.
 */
template <typename T>
class BSSNHaloExchanger {
  public:
    BSSNHaloExchanger(const MPIDomain& domain,
                      const tensorium_RG::Strides<T>& st,
                      size_t local_nx,
                      size_t local_ny,
                      size_t local_nz)
        : domain_(domain),
          exchanger_(domain, st, local_nx, local_ny, local_nz) {}

    BSSNHaloExchanger(const MPIDomain& domain,
                      const tensorium_RG::BSSNGridSoA<T>& grid)
        : BSSNHaloExchanger(domain, grid.alpha.st, grid.dims.nx, grid.dims.ny, grid.dims.nz) {}

    /**
     * @brief Exchange halos for a single field.
     */
    template <typename U>
    void exchange(tensorium_RG::Field3D<U>& field) {
        static_assert(std::is_same_v<U, T>, "All BSSN fields must share the same scalar type");
        exchanger_.exchange_blocking(field.ptr());
    }

    /**
     * @brief Exchange halos for all BSSN fields.
     */
    void exchange_all(tensorium_RG::BSSNGridSoA<T>& grid) {
        // Scalars
        exchange(grid.alpha);
        exchange(grid.chi);
        exchange(grid.K);
        exchange(grid.Theta);

        // Vectors
        for (int i = 0; i < 3; ++i) {
            exchange(grid.beta[i]);
            exchange(grid.B[i]);
            exchange(grid.tildeGamma[i]);
            exchange(grid.Z[i]);
        }

        // Symmetric tensors
        for (int s = 0; s < 6; ++s) {
            exchange(grid.gamma_tilde[s]);
            exchange(grid.gamma_tilde_inv[s]);
            exchange(grid.A_tilde[s]);
        }
    }

    const MPIDomain& domain() const { return domain_; }

  private:
    const MPIDomain& domain_;
    HaloExchanger<T> exchanger_;
};

/**
 * @brief MPI-aware boundary condition wrapper.
 * @details
 * Exchanges halos with MPI neighbors, then applies physical boundary
 * conditions only at true domain boundaries.
 */
template <typename T, typename PhysicalBoundary = tensorium_RG::bssn::BoundaryRadiative>
class MPIBoundary {
  public:
    MPIBoundary(const MPIDomain& domain,
                const tensorium_RG::Strides<T>& st,
                size_t local_nx,
                size_t local_ny,
                size_t local_nz)
        : domain_(domain),
          exchanger_(domain, st, local_nx, local_ny, local_nz) {}

    MPIBoundary(const MPIDomain& domain,
                const tensorium_RG::BSSNGridSoA<T>& grid)
        : MPIBoundary(domain, grid.alpha.st, grid.dims.nx, grid.dims.ny, grid.dims.nz) {}

    /**
     * @brief Apply physical boundary conditions (no-op for interior faces).
     */
    template <typename U>
    static void apply_physical(tensorium_RG::Field3D<U>&,
                               const tensorium_RG::BSSNGridSoA<U>&,
                               tensorium_RG::bssn::BoundaryField,
                               int) {
        // Handled by apply_halo
    }

    /**
     * @brief Apply halo boundary conditions (MPI exchange + physical BC).
     */
    template <typename U>
    void apply_halo(tensorium_RG::Field3D<U>& field,
                    const tensorium_RG::BSSNGridSoA<U>& G,
                    tensorium_RG::bssn::BoundaryField which,
                    int component) {
        exchanger_.exchange(field);

        // Apply physical BC at domain edges
        apply_physical_bc_at_edges(field, G, which, component);
    }

    /**
     * @brief Exchange halos for all fields.
     */
    void exchange_all_halos(tensorium_RG::BSSNGridSoA<T>& grid) {
        exchanger_.exchange_all(grid);
    }

    /**
     * @brief Check if at a physical boundary in given direction.
     */
    bool is_boundary_minus(int axis) const {
        switch (axis) {
        case 0:
            return domain_.is_boundary_x_minus();
        case 1:
            return domain_.is_boundary_y_minus();
        default:
            return domain_.is_boundary_z_minus();
        }
    }
    bool is_boundary_plus(int axis) const {
        switch (axis) {
        case 0:
            return domain_.is_boundary_x_plus();
        case 1:
            return domain_.is_boundary_y_plus();
        default:
            return domain_.is_boundary_z_plus();
        }
    }

    const MPIDomain& domain() const { return domain_; }

  private:
    const MPIDomain& domain_;
    BSSNHaloExchanger<T> exchanger_;

    template <typename U>
    void apply_physical_bc_at_edges(tensorium_RG::Field3D<U>& field,
                                    const tensorium_RG::BSSNGridSoA<U>& G,
                                    tensorium_RG::bssn::BoundaryField which,
                                    int component) {
        // Store original settings
        bool orig[6];
        for (int axis = 0; axis < 3; ++axis) {
            orig[axis * 2] = PhysicalBoundary::rhs_sommerfeld_enabled(axis, false);
            orig[axis * 2 + 1] = PhysicalBoundary::rhs_sommerfeld_enabled(axis, true);
        }

        // Only enable at true domain boundaries
        PhysicalBoundary::set_rhs_sommerfeld_faces(
            is_boundary_minus(0) && orig[0],
            is_boundary_plus(0) && orig[1],
            is_boundary_minus(1) && orig[2],
            is_boundary_plus(1) && orig[3],
            is_boundary_minus(2) && orig[4],
            is_boundary_plus(2) && orig[5]
        );

        PhysicalBoundary::apply_halo(field, G, which, component);

        // Restore
        PhysicalBoundary::set_rhs_sommerfeld_faces(
            orig[0], orig[1], orig[2], orig[3], orig[4], orig[5]
        );
    }
};

/**
 * @brief Static boundary adapter for use with BSSNRKStepper template.
 */
template <typename T>
class MPIBoundaryAdapter {
  public:
    inline static MPIBoundary<T>* instance = nullptr;

    template <typename U>
    static void apply_physical(tensorium_RG::Field3D<U>& field,
                               const tensorium_RG::BSSNGridSoA<U>& G,
                               tensorium_RG::bssn::BoundaryField which,
                               int component) {
        if (instance) {
            MPIBoundary<T>::apply_physical(field, G, which, component);
        }
    }

    template <typename U>
    static void apply_halo(tensorium_RG::Field3D<U>& field,
                           const tensorium_RG::BSSNGridSoA<U>& G,
                           tensorium_RG::bssn::BoundaryField which,
                           int component) {
        if (instance) {
            instance->apply_halo(field, G, which, component);
        }
    }

    static void set_characteristic(double speed, double dt) {
        tensorium_RG::bssn::BoundaryRadiative::set_characteristic(speed, dt);
    }
};

/**
 * @brief Compute global CFL time step using existing Comm reductions.
 */
template <typename T>
T compute_global_dt_cfl(const tensorium_RG::BSSNGridSoA<T>& grid,
                        const MPIDomain& domain,
                        T cfl_factor,
                        size_t padding = 4) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i_begin = std::min(I0 + guard, I1);
    const size_t j_begin = std::min(J0 + guard, J1);
    const size_t k_begin = std::min(K0 + guard, K1);
    const size_t i_end = (I1 > guard) ? I1 - guard : I1;
    const size_t j_end = (J1 > guard) ? J1 - guard : J1;
    const size_t k_end = (K1 > guard) ? K1 - guard : K1;

    T local_max_beta = T(0);
#pragma omp parallel for collapse(3) reduction(max : local_max_beta)
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                const T bx = grid.beta[0].ptr()[id];
                const T by = grid.beta[1].ptr()[id];
                const T bz = grid.beta[2].ptr()[id];
                const T beta_mag = std::sqrt(bx * bx + by * by + bz * bz);
                local_max_beta = std::max(local_max_beta, beta_mag);
            }

    Reductions red(domain);
    T local_max_speed = local_max_beta + T(1.0);
    T global_max_speed = red.allreduce_max(local_max_speed);

    if (global_max_speed <= T(0)) {
        global_max_speed = T(1);
    }

    T min_dx = std::min({grid.dx, grid.dy, grid.dz});
    return cfl_factor * min_dx / global_max_speed;
}

} // namespace tensorium::mpi
