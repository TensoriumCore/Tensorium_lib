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
#include <cstdint>
#include <memory>
#include <type_traits>

namespace tensorium::mpi {

enum HaloFieldMask : uint32_t {
    HaloFieldNone = 0,
    HaloFieldAlpha = 1u << 0,
    HaloFieldChi = 1u << 1,
    HaloFieldK = 1u << 2,
    HaloFieldTheta = 1u << 3,
    HaloFieldBeta = 1u << 4,
    HaloFieldB = 1u << 5,
    HaloFieldTildeGamma = 1u << 6,
    HaloFieldZ = 1u << 7,
    HaloFieldGammaTilde = 1u << 8,
    HaloFieldGammaTildeInverse = 1u << 9,
    HaloFieldATilde = 1u << 10,
    HaloFieldAll = (1u << 11) - 1
};

inline constexpr HaloFieldMask operator|(HaloFieldMask lhs, HaloFieldMask rhs) {
    return static_cast<HaloFieldMask>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline constexpr HaloFieldMask operator&(HaloFieldMask lhs, HaloFieldMask rhs) {
    return static_cast<HaloFieldMask>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

inline constexpr HaloFieldMask &operator|=(HaloFieldMask &lhs, HaloFieldMask rhs) {
    lhs = lhs | rhs;
    return lhs;
}

inline constexpr bool halo_mask_has(HaloFieldMask mask, HaloFieldMask field) {
    return static_cast<uint32_t>(mask & field) != 0u;
}

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
std::unique_ptr<tensorium_RG::BSSNGridSoA<T>>
create_local_bssn_grid(const MPIDomain &domain, size_t ng, T dx, T dy, T dz, T x0_global = T(0),
                       T y0_global = T(0), T z0_global = T(0)) {
    auto grid = std::make_unique<tensorium_RG::BSSNGridSoA<T>>(domain.local_nx(), domain.local_ny(),
                                                               domain.local_nz(), ng, dx, dy, dz);

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
std::unique_ptr<tensorium_RG::BSSNGridSoA<T>> create_local_bssn_grid(const MPIDomain &domain) {
    return create_local_bssn_grid<T>(
        domain, domain.ng(), static_cast<T>(domain.dx()), static_cast<T>(domain.dy()),
        static_cast<T>(domain.dz()),
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
template <typename T> class BSSNHaloExchanger {
  public:
    using ExchangeState = typename BatchHaloExchanger<T>::ExchangeState;

    BSSNHaloExchanger(const MPIDomain &domain, const tensorium_RG::Strides<T> &st, size_t local_nx,
                      size_t local_ny, size_t local_nz)
        : domain_(domain),
          exchanger_(domain, st, local_nx, local_ny, local_nz) {}

    BSSNHaloExchanger(const MPIDomain &domain, const tensorium_RG::BSSNGridSoA<T> &grid)
        : BSSNHaloExchanger(domain, grid.alpha.st, grid.dims.nx, grid.dims.ny, grid.dims.nz) {}

    /**
     * @brief Exchange halos for a single field.
     */
    template <typename U> void exchange(tensorium_RG::Field3D<U> &field) {
        static_assert(std::is_same_v<U, T>, "All BSSN fields must share the same scalar type");
        exchanger_.exchanger().exchange_blocking(field.ptr());
    }

    /**
     * @brief Exchange halos for all BSSN fields.
     */
    void exchange_all(tensorium_RG::BSSNGridSoA<T> &grid) {
        auto fields = collect_fields(grid, HaloFieldAll);
        exchanger_.exchange_all(fields);
    }

    void exchange_all_start(tensorium_RG::BSSNGridSoA<T> &grid, ExchangeState &state) {
        auto fields = collect_fields(grid, HaloFieldAll);
        exchanger_.exchange_start(fields, state);
    }

    void exchange_all_wait(ExchangeState &state) { exchanger_.exchange_wait(state); }

    void exchange(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask) {
        auto fields = collect_fields(grid, mask);
        exchanger_.exchange_all(fields);
    }

    void exchange_start(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask,
                        ExchangeState &state) {
        auto fields = collect_fields(grid, mask);
        exchanger_.exchange_start(fields, state);
    }

    const MPIDomain &domain() const { return domain_; }

  private:
    const MPIDomain      &domain_;
    BatchHaloExchanger<T> exchanger_;

    static std::vector<T *> collect_fields(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask) {
        std::vector<T *> fields;
        fields.reserve(34);

        if (halo_mask_has(mask, HaloFieldAlpha))
            fields.push_back(grid.alpha.ptr());
        if (halo_mask_has(mask, HaloFieldChi))
            fields.push_back(grid.chi.ptr());
        if (halo_mask_has(mask, HaloFieldK))
            fields.push_back(grid.K.ptr());
        if (halo_mask_has(mask, HaloFieldTheta))
            fields.push_back(grid.Theta.ptr());

        for (int i = 0; i < 3; ++i) {
            if (halo_mask_has(mask, HaloFieldBeta))
                fields.push_back(grid.beta[i].ptr());
            if (halo_mask_has(mask, HaloFieldB))
                fields.push_back(grid.B[i].ptr());
            if (halo_mask_has(mask, HaloFieldTildeGamma))
                fields.push_back(grid.tildeGamma[i].ptr());
            if (halo_mask_has(mask, HaloFieldZ))
                fields.push_back(grid.Z[i].ptr());
        }

        for (int s = 0; s < 6; ++s) {
            if (halo_mask_has(mask, HaloFieldGammaTilde))
                fields.push_back(grid.gamma_tilde[s].ptr());
            if (halo_mask_has(mask, HaloFieldGammaTildeInverse))
                fields.push_back(grid.gamma_tilde_inv[s].ptr());
            if (halo_mask_has(mask, HaloFieldATilde))
                fields.push_back(grid.A_tilde[s].ptr());
        }

        return fields;
    }
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
    using ExchangeState = typename BSSNHaloExchanger<T>::ExchangeState;

    MPIBoundary(const MPIDomain &domain, const tensorium_RG::Strides<T> &st, size_t local_nx,
                size_t local_ny, size_t local_nz)
        : domain_(domain),
          exchanger_(domain, st, local_nx, local_ny, local_nz) {}

    MPIBoundary(const MPIDomain &domain, const tensorium_RG::BSSNGridSoA<T> &grid)
        : MPIBoundary(domain, grid.alpha.st, grid.dims.nx, grid.dims.ny, grid.dims.nz) {}

    /**
     * @brief Apply physical boundary conditions (no-op for interior faces).
     */
    template <typename U>
    static void apply_physical(tensorium_RG::Field3D<U> &, const tensorium_RG::BSSNGridSoA<U> &,
                               tensorium_RG::bssn::BoundaryField, int) {
        // Handled by apply_halo
    }

    /**
     * @brief Apply halo boundary conditions (MPI exchange + physical BC).
     */
    template <typename U>
    void apply_halo(tensorium_RG::Field3D<U> &field, const tensorium_RG::BSSNGridSoA<U> &G,
                    tensorium_RG::bssn::BoundaryField which, int component) {
        exchanger_.exchange(field);

        // Apply physical BC at domain edges
        apply_physical_bc_at_edges(field, G, which, component);
    }

    /**
     * @brief Exchange halos for all fields.
     */
    void exchange_all_halos(tensorium_RG::BSSNGridSoA<T> &grid) {
        ExchangeState state;
        exchange_all_halos_start(grid, state);
        exchange_all_halos_finish(grid, state);
    }

    void exchange_all_halos_start(tensorium_RG::BSSNGridSoA<T> &grid, ExchangeState &state) {
        exchanger_.exchange_all_start(grid, state);
    }

    void exchange_all_halos_finish(tensorium_RG::BSSNGridSoA<T> &grid, ExchangeState &state) {
        exchanger_.exchange_all_wait(state);
        apply_physical_bc_all_fields(grid, HaloFieldAll);
    }

    void exchange_halos(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask) {
        ExchangeState state;
        exchange_halos_start(grid, mask, state);
        exchange_halos_finish(grid, mask, state);
    }

    void exchange_halos_start(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask,
                              ExchangeState &state) {
        exchanger_.exchange_start(grid, mask, state);
    }

    void exchange_halos_finish(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask,
                               ExchangeState &state) {
        exchanger_.exchange_all_wait(state);
        apply_physical_bc_all_fields(grid, mask);
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

    const MPIDomain &domain() const { return domain_; }

  private:
    const MPIDomain     &domain_;
    BSSNHaloExchanger<T> exchanger_;

    template <typename U>
    void apply_physical_bc_at_edges(tensorium_RG::Field3D<U>           &field,
                                    const tensorium_RG::BSSNGridSoA<U> &G,
                                    tensorium_RG::bssn::BoundaryField which, int component) {
        // Store original settings
        bool orig[6];
        bool active_orig[6];
        for (int axis = 0; axis < 3; ++axis) {
            orig[axis * 2] = PhysicalBoundary::rhs_sommerfeld_enabled(axis, false);
            orig[axis * 2 + 1] = PhysicalBoundary::rhs_sommerfeld_enabled(axis, true);
            active_orig[axis * 2] = PhysicalBoundary::active_enabled(axis, false);
            active_orig[axis * 2 + 1] = PhysicalBoundary::active_enabled(axis, true);
        }

        const bool act_ix1 = is_boundary_minus(0);
        const bool act_ox1 = is_boundary_plus(0);
        const bool act_ix2 = is_boundary_minus(1);
        const bool act_ox2 = is_boundary_plus(1);
        const bool act_ix3 = is_boundary_minus(2);
        const bool act_ox3 = is_boundary_plus(2);

        PhysicalBoundary::set_rhs_sommerfeld_faces(act_ix1 && orig[0], act_ox1 && orig[1],
                                                   act_ix2 && orig[2], act_ox2 && orig[3],
                                                   act_ix3 && orig[4], act_ox3 && orig[5]);
        PhysicalBoundary::set_active_faces(act_ix1, act_ox1, act_ix2, act_ox2, act_ix3, act_ox3);

        PhysicalBoundary::apply_halo(field, G, which, component);

        PhysicalBoundary::set_rhs_sommerfeld_faces(orig[0], orig[1], orig[2], orig[3], orig[4],
                                                   orig[5]);
        PhysicalBoundary::set_active_faces(active_orig[0], active_orig[1], active_orig[2],
                                           active_orig[3], active_orig[4], active_orig[5]);
    }

    void apply_physical_bc_all_fields(tensorium_RG::BSSNGridSoA<T> &grid, HaloFieldMask mask) {
        if (halo_mask_has(mask, HaloFieldAlpha))
            apply_physical_bc_at_edges(grid.alpha, grid, tensorium_RG::bssn::BoundaryField::Alpha,
                                       0);
        if (halo_mask_has(mask, HaloFieldChi))
            apply_physical_bc_at_edges(grid.chi, grid, tensorium_RG::bssn::BoundaryField::Chi, 0);
        if (halo_mask_has(mask, HaloFieldK))
            apply_physical_bc_at_edges(grid.K, grid, tensorium_RG::bssn::BoundaryField::K, 0);
        if (halo_mask_has(mask, HaloFieldTheta))
            apply_physical_bc_at_edges(grid.Theta, grid, tensorium_RG::bssn::BoundaryField::Theta,
                                       0);

        for (int i = 0; i < 3; ++i) {
            if (halo_mask_has(mask, HaloFieldBeta))
                apply_physical_bc_at_edges(grid.beta[i], grid,
                                           tensorium_RG::bssn::BoundaryField::Beta, i);
            if (halo_mask_has(mask, HaloFieldB))
                apply_physical_bc_at_edges(grid.B[i], grid, tensorium_RG::bssn::BoundaryField::B,
                                           i);
            if (halo_mask_has(mask, HaloFieldTildeGamma))
                apply_physical_bc_at_edges(grid.tildeGamma[i], grid,
                                           tensorium_RG::bssn::BoundaryField::TildeGamma, i);
            if (halo_mask_has(mask, HaloFieldZ))
                apply_physical_bc_at_edges(grid.Z[i], grid, tensorium_RG::bssn::BoundaryField::Z,
                                           i);
        }

        for (int s = 0; s < 6; ++s) {
            if (halo_mask_has(mask, HaloFieldGammaTilde))
                apply_physical_bc_at_edges(grid.gamma_tilde[s], grid,
                                           tensorium_RG::bssn::BoundaryField::GammaTilde, s);
            if (halo_mask_has(mask, HaloFieldGammaTildeInverse))
                apply_physical_bc_at_edges(grid.gamma_tilde_inv[s], grid,
                                           tensorium_RG::bssn::BoundaryField::GammaTildeInverse, s);
            if (halo_mask_has(mask, HaloFieldATilde))
                apply_physical_bc_at_edges(grid.A_tilde[s], grid,
                                           tensorium_RG::bssn::BoundaryField::ATilde, s);
        }
    }
};

/**
 * @brief Static boundary adapter for use with BSSNRKStepper template.
 */
template <typename T> class MPIBoundaryAdapter {
  public:
    inline static MPIBoundary<T> *instance = nullptr;

    template <typename U>
    static void apply_physical(tensorium_RG::Field3D<U>           &field,
                               const tensorium_RG::BSSNGridSoA<U> &G,
                               tensorium_RG::bssn::BoundaryField which, int component) {
        if (instance) {
            MPIBoundary<T>::apply_physical(field, G, which, component);
        }
    }

    template <typename U>
    static void apply_halo(tensorium_RG::Field3D<U> &field, const tensorium_RG::BSSNGridSoA<U> &G,
                           tensorium_RG::bssn::BoundaryField which, int component) {
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
T compute_global_dt_cfl(const tensorium_RG::BSSNGridSoA<T> &grid, const MPIDomain &domain,
                        T cfl_factor, size_t padding = 4) {
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
                const T      bx = grid.beta[0].ptr()[id];
                const T      by = grid.beta[1].ptr()[id];
                const T      bz = grid.beta[2].ptr()[id];
                const T      beta_mag = std::sqrt(bx * bx + by * by + bz * bz);
                local_max_beta = std::max(local_max_beta, beta_mag);
            }

    Reductions red(domain);
    T          local_max_speed = local_max_beta + T(1.0);
    T          global_max_speed = red.allreduce_max(local_max_speed);

    if (global_max_speed <= T(0)) {
        global_max_speed = T(1);
    }

    T min_dx = std::min({grid.dx, grid.dy, grid.dz});
    return cfl_factor * min_dx / global_max_speed;
}

} // namespace tensorium::mpi
