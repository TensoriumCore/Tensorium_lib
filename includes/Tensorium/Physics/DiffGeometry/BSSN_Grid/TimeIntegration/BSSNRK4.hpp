#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "../Evolution/BSSNEvolutionATilde.hpp"
#include "../Evolution/BSSNEvolutionChi.hpp"
#include "../Evolution/BSSNEvolutionCommon.hpp"
#include "../Evolution/BSSNEvolutionGamma.hpp"
#include "../Evolution/BSSNEvolutionGammaTilde.hpp"
#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../Evolution/BSSNEvolutionK.hpp"
#include "../Evolution/BSSNEvolutionZ4C.hpp"
#include "../CUDA/BSSNCudaGridOps.hpp"
#include "../Geometry/BSSNAlgebraic.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include "../Geometry/BSSNProjectionMonitor.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Fields/BSSNGridDevice.hpp"
#include "../Fields/BSSNGridViews.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "BSSNPerfTimers.hpp"
#include "BSSNRHSSweep.hpp"
#include <Tensorium/Backend/Common/Backend.hpp>
#include <Tensorium/Backend/CUDA/Core/CudaRuntime.hpp>

/**
 * @file BSSNRK4.hpp
 * @brief Fourth-order Runge–Kutta time integrator that orchestrates halo updates, geometry
 * rebuilds, RHS evaluations, and constraint monitoring.
 */

namespace tensorium_RG::bssn {

namespace detail {

template <typename T> inline void allocate_like(const Field3D<T> &src, Field3D<T> &dst) {
    dst.st = src.st;
    const size_t total = src.st.nx_tot * src.st.ny_tot * src.st.nz_tot;
    dst.data = aligned_alloc_n<T>(total);
}

template <typename T> inline void zero_field(Field3D<T> &f) {
    const size_t total = f.st.nx_tot * f.st.ny_tot * f.st.nz_tot;
    std::fill(f.ptr(), f.ptr() + total, T(0));
}

template <typename T> inline void zero_fields(Field3D<T> *fields, size_t count) {
    for (size_t i = 0; i < count; ++i)
        zero_field(fields[i]);
}

template <typename T> inline T smooth_floor(T val, T floor) {
    const T delta = 1.0e-10;
    return 0.5 * (val + floor + std::sqrt((val - floor) * (val - floor) + delta));
}
} // namespace detail

template <typename Boundary, typename = void> struct BoundaryConfigurator {
    static inline void set_characteristic(double, double) {}
};

template <typename Boundary, typename = void> struct BoundaryFieldSpeedConfigurator {
    static inline void set(double, double, double) {}
};

template <typename Boundary, typename = void> struct BoundaryStageFractionConfigurator {
    static inline void set(double) {}
};

template <typename Boundary>
struct BoundaryConfigurator<Boundary, std::void_t<decltype(Boundary::set_characteristic(
                                     std::declval<double>(), std::declval<double>()))>> {
    static inline void set_characteristic(double speed, double dt) {
        Boundary::set_characteristic(speed, dt);
    }
};

template <typename Boundary>
struct BoundaryFieldSpeedConfigurator<
    Boundary, std::void_t<decltype(Boundary::set_field_characteristic_speeds(
                  std::declval<double>(), std::declval<double>(),
                  std::declval<double>()))>> {
    static inline void set(double gauge_speed, double z4c_speed, double khat_speed) {
        Boundary::set_field_characteristic_speeds(gauge_speed, z4c_speed, khat_speed);
    }
};

template <typename Boundary>
struct BoundaryStageFractionConfigurator<
    Boundary, std::void_t<decltype(Boundary::set_stage_fraction(std::declval<double>()))>> {
    static inline void set(double stage_fraction) { Boundary::set_stage_fraction(stage_fraction); }
};

template <typename Boundary, typename = void> struct BoundaryPhysicalEvolutionTraits {
    static constexpr bool value = std::is_same_v<Boundary, BoundaryRadiative>;
};

template <typename Boundary>
struct BoundaryPhysicalEvolutionTraits<
    Boundary, std::void_t<decltype(Boundary::evolve_physical_cells)>> {
    static constexpr bool value = Boundary::evolve_physical_cells;
};

template <typename Boundary, typename = void> struct BoundaryRadiativeRHSTraits {
    static constexpr bool value = std::is_same_v<Boundary, BoundaryRadiative>;

    static size_t collar() {
        if constexpr (std::is_same_v<Boundary, BoundaryRadiative>)
            return BoundaryRadiative::rhs_collar();
        return size_t(0);
    }
};

template <typename Boundary>
struct BoundaryRadiativeRHSTraits<
    Boundary, std::void_t<decltype(Boundary::radiative_rhs_collar_enabled)>> {
    static constexpr bool value = Boundary::radiative_rhs_collar_enabled;

    static size_t collar() {
        if constexpr (Boundary::radiative_rhs_collar_enabled)
            return Boundary::rhs_collar();
        return size_t(0);
    }
};

template <typename T> class EvolvedStateSoA {
  public:
    GridDims   dims;
    Strides<T> st;

    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;
    Field3D<T> Theta;

    Field3D<T> beta[3];
    Field3D<T> B[3];
    Field3D<T> tildeGamma[3];
    Field3D<T> Z[3];

    Field3D<T> gamma_tilde[6];
    Field3D<T> A_tilde[6];

    T dx, dy, dz;

    EvolvedStateSoA(size_t nx, size_t ny, size_t nz, size_t ng, T dx_, T dy_, T dz_)
        : dims{nx, ny, nz, ng},
          dx(dx_),
          dy(dy_),
          dz(dz_) {
        const size_t nx_tot = nx + 2 * ng;
        const size_t ny_tot = ny + 2 * ng;
        const size_t nz_tot = pad_simd<T>(nz + 2 * ng);

        st.nx_tot = nx_tot;
        st.ny_tot = ny_tot;
        st.nz_tot = nz_tot;

        st.sz = 1;
        st.sy = nz_tot;
        st.sx = ny_tot * nz_tot;

        auto alloc_field = [&](Field3D<T> &f) {
            const size_t N = nx_tot * ny_tot * nz_tot;
            f.data = aligned_alloc_n<T>(N);
            f.st = st;
        };

        alloc_field(alpha);
        alloc_field(chi);
        alloc_field(K);
        alloc_field(Theta);

        for (int i = 0; i < 3; ++i) {
            alloc_field(beta[i]);
            alloc_field(B[i]);
            alloc_field(tildeGamma[i]);
            alloc_field(Z[i]);
        }

        for (int s = 0; s < 6; ++s) {
            alloc_field(gamma_tilde[s]);
            alloc_field(A_tilde[s]);
        }
    }

    T x0 = 0, y0 = 0, z0 = 0;
};

/// @brief Stores RHS buffers for every evolved variable.
template <typename T> struct BSSNRHSWorkspace {
    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;
    Field3D<T> Theta;
    Field3D<T> beta[3];
    Field3D<T> B[3];
    Field3D<T> gamma_tilde[6];
    Field3D<T> A_tilde[6];
    Field3D<T> tildeGamma[3];
    Field3D<T> Z[3];

    void allocate_like(const BSSNGridSoA<T> &grid) {
        detail::allocate_like(grid.alpha, alpha);
        detail::allocate_like(grid.chi, chi);
        detail::allocate_like(grid.K, K);
        detail::allocate_like(grid.Theta, Theta);
        for (int i = 0; i < 3; ++i) {
            detail::allocate_like(grid.beta[i], beta[i]);
            detail::allocate_like(grid.B[i], B[i]);
            detail::allocate_like(grid.tildeGamma[i], tildeGamma[i]);
            detail::allocate_like(grid.Z[i], Z[i]);
        }
        for (int s = 0; s < 6; ++s) {
            detail::allocate_like(grid.gamma_tilde[s], gamma_tilde[s]);
            detail::allocate_like(grid.A_tilde[s], A_tilde[s]);
        }
    }

    void zero() {
        detail::zero_field(alpha);
        detail::zero_field(chi);
        detail::zero_field(K);
        detail::zero_field(Theta);
        detail::zero_fields(beta, 3);
        detail::zero_fields(B, 3);
        detail::zero_fields(gamma_tilde, 6);
        detail::zero_fields(A_tilde, 6);
        detail::zero_fields(tildeGamma, 3);
        detail::zero_fields(Z, 3);
    }
};

struct RHSBoundaryFaceMask {
    bool face[3][2] = {{true, true}, {true, true}, {true, true}};

    [[nodiscard]] bool enabled(int axis, bool outer) const noexcept {
        const int ax = std::clamp(axis, 0, 2);
        return face[ax][outer ? 1 : 0];
    }
};

template <typename GridType, typename T>
inline void apply_z4c_rhs_boundary(const GridType &grid, BSSNRHSWorkspace<T> &rhs,
                                   const RHSBoundaryFaceMask &mask, size_t collar_width = 1) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i0 = I0;
    const size_t j0 = J0;
    const size_t k0 = K0;
    const size_t i1 = I1;
    const size_t j1 = J1;
    const size_t k1 = K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;
    collar_width = std::max<size_t>(size_t(1), collar_width);

    const T inv_2dx = T(0.5) / grid.dx;
    const T inv_2dy = T(0.5) / grid.dy;
    const T inv_2dz = T(0.5) / grid.dz;
    const T inv_dx = T(1) / grid.dx;
    const T inv_dy = T(1) / grid.dy;
    const T inv_dz = T(1) / grid.dz;
    const T inv_r_floor = T(1e-12);
#pragma omp parallel for collapse(3)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const bool in_ix1 = mask.enabled(0, false) && (i - i0 < collar_width);
                const bool in_ox1 = mask.enabled(0, true) && (i1 - 1 - i < collar_width);
                const bool in_ix2 = mask.enabled(1, false) && (j - j0 < collar_width);
                const bool in_ox2 = mask.enabled(1, true) && (j1 - 1 - j < collar_width);
                const bool in_ix3 = mask.enabled(2, false) && (k - k0 < collar_width);
                const bool in_ox3 = mask.enabled(2, true) && (k1 - 1 - k < collar_width);
                const bool in_collar = in_ix1 || in_ox1 || in_ix2 || in_ox2 || in_ix3 || in_ox3;
                if (!in_collar)
                    continue;

                const size_t id = grid.alpha.idx(i, j, k);
                T            x, y, z;
                grid.coords(i, j, k, x, y, z);
                const T r = std::sqrt(x * x + y * y + z * z);
                const T inv_r = T(1) / std::max(r, inv_r_floor);

                auto deriv_axis = [&](const Field3D<T> &f, ptrdiff_t stride, size_t pos, size_t lo,
                                      size_t hi, T inv_h, T inv_2h) -> T {
                    const T *p = f.ptr() + id;
                    if (pos == lo && lo + 2 < hi)
                        return (-T(1.5) * p[0] + T(2) * p[stride] - T(0.5) * p[2 * stride]) *
                               inv_h;
                    if (pos + 1 == hi && lo + 2 < hi)
                        return (T(1.5) * p[0] - T(2) * p[-stride] + T(0.5) * p[-2 * stride]) *
                               inv_h;
                    return (p[stride] - p[-stride]) * inv_2h;
                };

                // Match GRChombo's RHS-only Sommerfeld closure: use the same local Cartesian
                // operator on faces, edges, and corners, rather than a special radial fallback.
                auto sommerfeld_rhs = [&](const Field3D<T> &u, T asymptotic = T(0)) -> T {
                    const T du_dx = deriv_axis(u, u.st.sx, i, i0, i1, inv_dx, inv_2dx);
                    const T du_dy = deriv_axis(u, u.st.sy, j, j0, j1, inv_dy, inv_2dy);
                    const T du_dz = deriv_axis(u, ptrdiff_t(1), k, k0, k1, inv_dz, inv_2dz);
                    const T u0 = u.ptr()[id];
                    const T projected = (du_dx * x + du_dy * y + du_dz * z) * inv_r;
                    return -projected + (asymptotic - u0) * inv_r;
                };

                auto sommerfeld_khat_rhs = [&]() -> T {
                    const T dK_dx =
                        deriv_axis(grid.K, grid.K.st.sx, i, i0, i1, inv_dx, inv_2dx);
                    const T dK_dy =
                        deriv_axis(grid.K, grid.K.st.sy, j, j0, j1, inv_dy, inv_2dy);
                    const T dK_dz =
                        deriv_axis(grid.K, ptrdiff_t(1), k, k0, k1, inv_dz, inv_2dz);
                    const T dTheta_dx = deriv_axis(grid.Theta, grid.Theta.st.sx, i, i0, i1,
                                                   inv_dx, inv_2dx);
                    const T dTheta_dy = deriv_axis(grid.Theta, grid.Theta.st.sy, j, j0, j1,
                                                   inv_dy, inv_2dy);
                    const T dTheta_dz =
                        deriv_axis(grid.Theta, ptrdiff_t(1), k, k0, k1, inv_dz, inv_2dz);
                    const T khat = Khat(grid.K.ptr()[id], grid.Theta.ptr()[id]);
                    const T projected =
                        ((dK_dx - T(2) * dTheta_dx) * x + (dK_dy - T(2) * dTheta_dy) * y +
                         (dK_dz - T(2) * dTheta_dz) * z) *
                        inv_r;
                    return -projected - khat * inv_r;
                };

                const T boundary_alpha = sommerfeld_rhs(
                    grid.alpha,
                    T(tensorium_RG::bssn::detail::minkowski_target(BoundaryField::Alpha, 0)));
                const T boundary_chi = sommerfeld_rhs(
                    grid.chi,
                    T(tensorium_RG::bssn::detail::minkowski_target(BoundaryField::Chi, 0)));
                T boundary_beta[3] = {T(0), T(0), T(0)};
                T boundary_B[3] = {T(0), T(0), T(0)};
                T boundary_tildeGamma[3] = {T(0), T(0), T(0)};
                T boundary_Z[3] = {T(0), T(0), T(0)};
                T boundary_gamma_tilde[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};
                T boundary_A_tilde[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};
                for (int a = 0; a < 3; ++a) {
                    boundary_beta[a] += sommerfeld_rhs(
                        grid.beta[a],
                        T(tensorium_RG::bssn::detail::minkowski_target(BoundaryField::Beta, a)));
                    boundary_B[a] += sommerfeld_rhs(
                        grid.B[a],
                        T(tensorium_RG::bssn::detail::minkowski_target(BoundaryField::B, a)));
                    boundary_tildeGamma[a] += sommerfeld_rhs(grid.tildeGamma[a], T(0));
                    boundary_Z[a] += sommerfeld_rhs(grid.Z[a], T(0));
                }
                for (int s = 0; s < 6; ++s) {
                    boundary_gamma_tilde[s] += sommerfeld_rhs(
                        grid.gamma_tilde[s],
                        T(tensorium_RG::bssn::detail::minkowski_target(BoundaryField::GammaTilde, s)));
                    boundary_A_tilde[s] += sommerfeld_rhs(grid.A_tilde[s], T(0));
                }
                const T boundary_Theta = sommerfeld_rhs(grid.Theta, T(0));
                const T boundary_K = sommerfeld_khat_rhs();

                auto overwrite_rhs = [&](Field3D<T> &target, T boundary_value) {
                    T *slot = target.ptr() + id;
                    *slot = boundary_value;
                };

                overwrite_rhs(rhs.alpha, boundary_alpha);
                overwrite_rhs(rhs.chi, boundary_chi);
                overwrite_rhs(rhs.K, boundary_K);
                overwrite_rhs(rhs.Theta, boundary_Theta);
                for (int a = 0; a < 3; ++a) {
                    overwrite_rhs(rhs.beta[a], boundary_beta[a]);
                    overwrite_rhs(rhs.B[a], boundary_B[a]);
                    overwrite_rhs(rhs.tildeGamma[a], boundary_tildeGamma[a]);
                    overwrite_rhs(rhs.Z[a], boundary_Z[a]);
                }
                for (int s = 0; s < 6; ++s) {
                    overwrite_rhs(rhs.gamma_tilde[s], boundary_gamma_tilde[s]);
                    overwrite_rhs(rhs.A_tilde[s], boundary_A_tilde[s]);
                }
            }
        }
    }
}

/// @brief Control parameters for the CFL-based time-step computation.
template <typename T> struct CFLControl {
    T cfl = T(0.35);      ///< Target CFL number for FD6 + RK4.
    T gauge_speed = T(1); ///< Multiplicative factor applied to the gauge speed (α term).
};

template <typename T>
inline T compute_dt_cfl(const BSSNGridSoA<T> &grid, const CFLControl<T> &control,
                        size_t padding = 4) {
    const T min_dx = std::min({grid.dx, grid.dy, grid.dz});

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i_begin = std::min(I0 + guard, I1);
    const size_t j_begin = std::min(J0 + guard, J1);
    const size_t k_begin = std::min(K0 + guard, K1);
    const size_t i_end = (I1 > guard) ? I1 - guard : I1;
    const size_t j_end = (J1 > guard) ? J1 - guard : J1;
    const size_t k_end = (K1 > guard) ? K1 - guard : K1;

    T max_beta = T(0);
#pragma omp parallel for collapse(3) reduction(max : max_beta)
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                const T      bx = grid.beta[0].ptr()[id];
                const T      by = grid.beta[1].ptr()[id];
                const T      bz = grid.beta[2].ptr()[id];

                const T beta_mag = std::sqrt(bx * bx + by * by + bz * bz);

                max_beta = std::max(max_beta, beta_mag);
            }

    const T gauge_term = control.gauge_speed;

    T denom = max_beta + gauge_term;
    if (denom <= T(0))
        denom = T(1);

    return control.cfl * (min_dx / denom);
}

template <typename T>
inline T compute_dt_cfl(const BSSNGridSoA<T> &grid, T cfl_factor, size_t padding = 4) {
    CFLControl<T> control;
    control.cfl = cfl_factor;
    return compute_dt_cfl(grid, control, padding);
}

template <typename T, typename Boundary> class BSSNRKStepper {
  public:
    using StageState = EvolvedStateSoA<T>;

    explicit BSSNRKStepper(const BSSNGridSoA<T> &prototype, size_t padding = 4)
        : BSSNRKStepper(prototype, tensorium::backend::Options{}, padding) {}

    BSSNRKStepper(const BSSNGridSoA<T> &prototype, const tensorium::backend::Options &backend,
                  size_t padding = 4)
        : padding_(padding),
          backend_options_(backend),
          stage_grid_(prototype.dims.nx, prototype.dims.ny, prototype.dims.nz, prototype.dims.ng,
                      prototype.dx, prototype.dy, prototype.dz) {
        stage_grid_.x0 = prototype.x0;
        stage_grid_.y0 = prototype.y0;
        stage_grid_.z0 = prototype.z0;
        stage_rhs_.allocate_like(prototype);
        detail::allocate_like(prototype.alpha, theta_ricciz4_trace_cache_);
        validate_backend_options();
    }

    void set_gauge_parameters(const GaugeParameters<T> &params) { gauge_params_ = params; }

    void set_backend_options(const tensorium::backend::Options &backend) {
        backend_options_ = backend;
        cuda_stepper_state_.reset();
        validate_backend_options();
    }

    [[nodiscard]] const tensorium::backend::Options &backend_options() const noexcept {
        return backend_options_;
    }

    void set_constraint_callback(
        std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> cb) {
        monitor_callback_ = std::move(cb);
    }

    void set_snapshot_callback(std::function<void(const BSSNGridSoA<T> &, size_t)> cb) {
        snapshot_callback_ = std::move(cb);
    }

    void set_state_log_stride(size_t stride) { state_log_stride_ = std::max<size_t>(size_t(1), stride); }
    [[nodiscard]] size_t state_log_stride() const noexcept { return state_log_stride_; }

    void set_rhs_prep_callback(std::function<void(BSSNRHSWorkspace<T> &)> cb) {
        rhs_prep_callback_ = std::move(cb);
    }

    void set_simulation_time(T time) noexcept { simulation_time_ = time; }
    [[nodiscard]] T simulation_time() const noexcept { return simulation_time_; }

    void set_boundary_dt(T dt) noexcept { boundary_dt_ = dt; }
    [[nodiscard]] T boundary_dt() const noexcept { return boundary_dt_; }

    void copy_runtime_state_from(const BSSNRKStepper &other) {
        gauge_params_ = other.gauge_params_;
        monitor_callback_ = other.monitor_callback_;
        snapshot_callback_ = other.snapshot_callback_;
        rhs_prep_callback_ = other.rhs_prep_callback_;
        state_log_stride_ = other.state_log_stride_;
        boundary_dt_ = other.boundary_dt_;
        simulation_time_ = other.simulation_time_;
    }

    // Diagnostics function (omitted mostly unchanged for brevity in logic check, but included in
    // full paste)
    void log_full_bssn_diagnostics(const BSSNGridSoA<T> &grid, size_t step, double r_min,
                                   double r_max, double chi_cut = 0.0) {
        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        const size_t i0 = I0;
        const size_t j0 = J0;
        const size_t k0 = K0;
        const size_t i1 = I1;
        const size_t j1 = J1;
        const size_t k1 = K1;

        double max_beta = 0.0, max_B = 0.0;
        double min_alpha = 1e100, min_chi = 1e100;
        double max_gamma = 0.0, max_det = 0.0, max_trA = 0.0;
        double min_theta = std::numeric_limits<double>::infinity();
        double max_theta = -std::numeric_limits<double>::infinity();
        double max_K = 0.0, max_A = 0.0;
        double max_Rc = 0.0, max_R3 = 0.0;
        size_t samples = 0;

        auto upd_max_abs = [](double &m, double v) {
            const double a = std::abs(v);
            if (std::isfinite(a))
                m = std::max(m, a);
        };
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                for (size_t k = k0; k < k1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    const double chi = double(grid.chi.ptr()[idx]);
                    if (chi_cut > 0.0 && !(chi > chi_cut))
                        continue;

                    const double x = double(grid.x0) + double(i - grid.dims.ng) * double(grid.dx);
                    const double y = double(grid.y0) + double(j - grid.dims.ng) * double(grid.dy);
                    const double z = double(grid.z0) + double(k - grid.dims.ng) * double(grid.dz);
                    const double r = std::sqrt(x * x + y * y + z * z);
                    if (!(r > r_min && r < r_max))
                        continue;
                    ++samples;

                    const double alpha = double(grid.alpha.ptr()[idx]);
                    if (std::isfinite(alpha))
                        min_alpha = std::min(min_alpha, alpha);
                    if (std::isfinite(chi))
                        min_chi = std::min(min_chi, chi);
                    const double theta = double(grid.Theta.ptr()[idx]);
                    if (std::isfinite(theta)) {
                        min_theta = std::min(min_theta, theta);
                        max_theta = std::max(max_theta, theta);
                    }

                    const double bx = double(grid.beta[0].ptr()[idx]);
                    const double by = double(grid.beta[1].ptr()[idx]);
                    const double bz = double(grid.beta[2].ptr()[idx]);
                    upd_max_abs(max_beta, std::sqrt(bx * bx + by * by + bz * bz));

                    const double Bx = double(grid.B[0].ptr()[idx]);
                    const double By = double(grid.B[1].ptr()[idx]);
                    const double Bz = double(grid.B[2].ptr()[idx]);
                    upd_max_abs(max_B, std::sqrt(Bx * Bx + By * By + Bz * Bz));

                    upd_max_abs(max_gamma, double(grid.tildeGamma[0].ptr()[idx]));
                    upd_max_abs(max_gamma, double(grid.tildeGamma[1].ptr()[idx]));
                    upd_max_abs(max_gamma, double(grid.tildeGamma[2].ptr()[idx]));

                    const double gxx = double(grid.gamma_tilde[XX].ptr()[idx]);
                    const double gxy = double(grid.gamma_tilde[XY].ptr()[idx]);
                    const double gxz = double(grid.gamma_tilde[XZ].ptr()[idx]);
                    const double gyy = double(grid.gamma_tilde[YY].ptr()[idx]);
                    const double gyz = double(grid.gamma_tilde[YZ].ptr()[idx]);
                    const double gzz = double(grid.gamma_tilde[ZZ].ptr()[idx]);

                    const double det = gxx * (gyy * gzz - gyz * gyz) -
                                       gxy * (gxy * gzz - gxz * gyz) +
                                       gxz * (gxy * gyz - gxz * gyy);
                    upd_max_abs(max_det, det - 1.0);

                    const double gixx = double(grid.gamma_tilde_inv[XX].ptr()[idx]);
                    const double gixy = double(grid.gamma_tilde_inv[XY].ptr()[idx]);
                    const double gixz = double(grid.gamma_tilde_inv[XZ].ptr()[idx]);
                    const double giyy = double(grid.gamma_tilde_inv[YY].ptr()[idx]);
                    const double giyz = double(grid.gamma_tilde_inv[YZ].ptr()[idx]);
                    const double gizz = double(grid.gamma_tilde_inv[ZZ].ptr()[idx]);

                    const double Axx = double(grid.A_tilde[XX].ptr()[idx]);
                    const double Axy = double(grid.A_tilde[XY].ptr()[idx]);
                    const double Axz = double(grid.A_tilde[XZ].ptr()[idx]);
                    const double Ayy = double(grid.A_tilde[YY].ptr()[idx]);
                    const double Ayz = double(grid.A_tilde[YZ].ptr()[idx]);
                    const double Azz = double(grid.A_tilde[ZZ].ptr()[idx]);

                    const double trA = gixx * Axx + giyy * Ayy + gizz * Azz +
                                       2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
                    upd_max_abs(max_trA, trA);
                    upd_max_abs(max_K, double(grid.K.ptr()[idx]));

                    upd_max_abs(max_A, Axx);
                    upd_max_abs(max_A, Axy);
                    upd_max_abs(max_A, Axz);
                    upd_max_abs(max_A, Ayy);
                    upd_max_abs(max_A, Ayz);
                    upd_max_abs(max_A, Azz);

                    const double Rxx = double(grid.Ricci[XX].ptr()[idx]);
                    const double Rxy = double(grid.Ricci[XY].ptr()[idx]);
                    const double Rxz = double(grid.Ricci[XZ].ptr()[idx]);
                    const double Ryy = double(grid.Ricci[YY].ptr()[idx]);
                    const double Ryz = double(grid.Ricci[YZ].ptr()[idx]);
                    const double Rzz = double(grid.Ricci[ZZ].ptr()[idx]);
                    upd_max_abs(max_Rc, Rxx);
                    upd_max_abs(max_Rc, Rxy);
                    upd_max_abs(max_Rc, Rxz);
                    upd_max_abs(max_Rc, Ryy);
                    upd_max_abs(max_Rc, Ryz);
                    upd_max_abs(max_Rc, Rzz);

                    const double R3 = gixx * Rxx + giyy * Ryy + gizz * Rzz +
                                      2.0 * (gixy * Rxy + gixz * Rxz + giyz * Ryz);
                    upd_max_abs(max_R3, R3);
                }

        if (samples == 0) {
            printf("[BSSN %04zu] empty-domain r in (%.3e, %.3e) chi_cut=%.3e\n", step, r_min, r_max,
                   chi_cut);
            return;
        }
        if (min_alpha > 1e90)
            min_alpha = 0.0;
        if (min_chi > 1e90)
            min_chi = 0.0;

        const double theta_min_print = std::isfinite(min_theta) ? min_theta : 0.0;
        const double theta_max_print = std::isfinite(max_theta) ? max_theta : 0.0;

        printf("[BSSN %04zu]\n"
               "\talpha_min=% .3e\tchi_min=% .3e\tTheta_min=% .3e\tTheta_max=% .3e\n"
               "\tbeta_max=% .3e\tB_max=% .3e\tGamma_max=% .3e\n"
               "\tdet_err=% .3e\ttrA=% .3e\tK_max=% .3e\tA_max=% .3e\n"
               "\tRicci_cmax=% .3e\tR3_max=% .3e\tsamples=%zu\n",
               step, min_alpha, min_chi, theta_min_print, theta_max_print, max_beta, max_B,
               max_gamma, max_det, max_trA, max_K, max_A, max_Rc, max_R3, samples);
    }

    void step(BSSNGridSoA<T> &grid, T dt, size_t step_index = 0) {
        switch (backend_options_.backend) {
        case tensorium::backend::Kind::CPU:
            step_cpu(grid, dt, step_index);
            return;
        case tensorium::backend::Kind::CUDA:
            step_cuda(grid, dt, step_index);
            return;
        }
    }

  private:
    void step_cpu(BSSNGridSoA<T> &grid, T dt, size_t step_index = 0) {
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
        tensorium_RG::bssn::detail::begin_contracted_warning_step(step_index);
#endif
        static constexpr std::array<double, 4> gam0_ref = {
            0.0, 0.121098479554482, -3.843833699660025, 0.546370891121863};
        static constexpr std::array<double, 4> gam1_ref = {
            1.0, 0.721781678111411, 2.121209265338722, 0.198653035682705};
        static constexpr std::array<double, 4> beta_ref = {
            1.193743905974738, 0.099279895495783, 1.131678018054042, 0.310665766509336};
        static constexpr std::array<double, 4> delta_ref = {
            1.0, 0.217683334308543, 1.065841341361089, 0.0};
        static constexpr std::array<double, 4> stage_time_ref = {0.0, 0.5, 0.5, 1.0};

        boundary_dt_ = dt;
        for (int stage = 0; stage < 4; ++stage) {
            if (stage == 0)
                copy_stage_reference(grid, stage_grid_);
            else
                accumulate_stage_reference(grid, stage_grid_, T(delta_ref[stage]));

            BoundaryStageFractionConfigurator<Boundary>::set(stage_time_ref[stage]);
            prepare_state_for_rhs(grid);
            auto stage_params = gauge_params_;
            stage_params.current_time = simulation_time_ + T(stage_time_ref[stage]) * dt;
            stage_params.frozen_Z_is_synced = !stage_params.evolve_Z;
            evaluate_rhs(grid, stage_rhs_, stage_params);

            apply_explicit_stage_update(grid, stage_grid_, stage_rhs_, T(gam0_ref[stage]),
                                        T(gam1_ref[stage]), T(beta_ref[stage]) * dt);
            if (gauge_params_.alpha_floor > T(0))
                apply_alpha_floor(grid, gauge_params_.alpha_floor);
            if (gauge_params_.chi_floor > T(0))
                apply_chi_floor(grid, gauge_params_.chi_floor);
        }
        simulation_time_ += dt;
        BoundaryStageFractionConfigurator<Boundary>::set(1.0);

        if (gauge_params_.alpha_floor > T(0))
            apply_alpha_floor(grid, gauge_params_.alpha_floor);
        if (gauge_params_.chi_floor > T(0))
            apply_chi_floor(grid, gauge_params_.chi_floor);

        const bool do_state_log = ((step_index + 1) % state_log_stride_ == 0);
        const bool needs_post_step_prepare =
            do_state_log || static_cast<bool>(monitor_callback_) || static_cast<bool>(snapshot_callback_);
        if (needs_post_step_prepare) {
            prepare_state_for_rhs(grid);
        } else {
            // The next stage preparation already rebuilds the algebraic auxiliaries. When no
            // post-step diagnostics need a full rebuild, project only once here instead of after
            // every RK stage update.
            tensorium_RG::bssn::enforce_algebraic_constraints(grid);
            synchronize_z_from_gamma_constraint(grid);
        }

        if (do_state_log) {
            const double min_extent =
                std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                          (grid.dims.nz - 1) * grid.dz});
            const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
            const double r_max = 0.45 * min_extent;
            log_gauge_diagnostics(grid, step_index);
            const double chi_cut = 1e-7;
            log_full_bssn_diagnostics(grid, step_index, r_min, r_max, chi_cut);
        }

        if (monitor_callback_) {
            const double min_extent =
                std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                          (grid.dims.nz - 1) * grid.dz});
            const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
            const double r_max = 0.45 * min_extent;
            auto        H_tmp = tensorium_RG::make_field(grid.alpha.st);
            Field3D<T>  M_tmp[3];
            Field3D<T>  C_tmp[3];
            for (int q = 0; q < 3; ++q) {
                M_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
                C_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
            }
            compute_bssn_constraints(grid, grid.Ricci, H_tmp, M_tmp, C_tmp, r_min, r_max, 0.0, 0.0,
                                     0.0);
            auto stats = compute_constraint_monitor(grid, H_tmp, padding_);
            populate_constraint_norms(grid, M_tmp, stats, padding_);
            if (monitor_callback_)
                monitor_callback_(grid, stats);
        }

        if (snapshot_callback_)
            snapshot_callback_(grid, step_index);

#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
        tensorium_RG::bssn::detail::finalize_contracted_warning_step();
#endif
        report_kernel_timers(step_index);
    }

    void step_cuda(BSSNGridSoA<T> &grid, T dt, size_t step_index) {
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
        tensorium_RG::bssn::detail::begin_contracted_warning_step(step_index);
#endif
        static constexpr std::array<double, 4> gam0_ref = {
            0.0, 0.121098479554482, -3.843833699660025, 0.546370891121863};
        static constexpr std::array<double, 4> gam1_ref = {
            1.0, 0.721781678111411, 2.121209265338722, 0.198653035682705};
        static constexpr std::array<double, 4> beta_ref = {
            1.193743905974738, 0.099279895495783, 1.131678018054042, 0.310665766509336};
        static constexpr std::array<double, 4> delta_ref = {
            1.0, 0.217683334308543, 1.065841341361089, 0.0};
        static constexpr std::array<double, 4> stage_time_ref = {0.0, 0.5, 0.5, 1.0};

        validate_backend_options();
        tensorium::cuda::set_device(backend_options_.device_ordinal);
        ensure_cuda_stepper_state(grid);
        boundary_dt_ = dt;
        sync_host_grid_to_cuda(grid);

        for (int stage = 0; stage < 4; ++stage) {
            const auto device_grid_view =
                static_cast<const BSSNGridDevice<T> &>(cuda_stepper_state_->grid).view();
            auto       device_stage_state_view = cuda_stepper_state_->stage_state.view();
            if (stage == 0) {
                tensorium_RG::bssn::cuda::copy_stage_reference(device_grid_view,
                                                               device_stage_state_view,
                                                               evolution_padding());
            } else {
                tensorium_RG::bssn::cuda::accumulate_stage_reference(
                    device_grid_view, device_stage_state_view, T(delta_ref[stage]),
                    evolution_padding());
            }
            tensorium::cuda::device_synchronize();

            if (stage != 0)
                sync_cuda_grid_to_host(grid);

            BoundaryStageFractionConfigurator<Boundary>::set(stage_time_ref[stage]);
            prepare_state_for_rhs(grid);
            sync_host_grid_to_cuda(grid);
            auto stage_params = gauge_params_;
            stage_params.current_time = simulation_time_ + T(stage_time_ref[stage]) * dt;
            stage_params.frozen_Z_is_synced = !stage_params.evolve_Z;
            evaluate_rhs(grid, stage_rhs_, stage_params);
            const bool rhs_on_device = can_use_cuda_accelerated_rhs(stage_params);

            const auto device_stage_state_const_view =
                static_cast<const BSSNRHSWorkspaceDevice<T> &>(cuda_stepper_state_->stage_state)
                    .view();
            if (!rhs_on_device)
                sync_host_rhs_to_cuda(stage_rhs_);
            tensorium_RG::bssn::cuda::apply_explicit_stage_update(
                cuda_stepper_state_->grid.view(), device_stage_state_const_view,
                static_cast<const BSSNRHSWorkspaceDevice<T> &>(cuda_stepper_state_->rhs_workspace)
                    .view(),
                T(gam0_ref[stage]), T(gam1_ref[stage]), T(beta_ref[stage]) * dt,
                evolution_padding());
            apply_cuda_post_stage_corrections();
        }
        simulation_time_ += dt;
        BoundaryStageFractionConfigurator<Boundary>::set(1.0);
        sync_cuda_grid_to_host(grid);

        const bool do_state_log = ((step_index + 1) % state_log_stride_ == 0);
        const bool needs_post_step_prepare =
            do_state_log || static_cast<bool>(monitor_callback_) || static_cast<bool>(snapshot_callback_);
        if (needs_post_step_prepare) {
            prepare_state_for_rhs(grid);
        } else {
            synchronize_z_from_gamma_constraint(grid);
        }

        if (do_state_log) {
            const double min_extent =
                std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                          (grid.dims.nz - 1) * grid.dz});
            const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
            const double r_max = 0.45 * min_extent;
            log_gauge_diagnostics(grid, step_index);
            const double chi_cut = 1e-7;
            log_full_bssn_diagnostics(grid, step_index, r_min, r_max, chi_cut);
        }

        if (monitor_callback_) {
            const double min_extent =
                std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                          (grid.dims.nz - 1) * grid.dz});
            const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
            const double r_max = 0.45 * min_extent;
            auto        H_tmp = tensorium_RG::make_field(grid.alpha.st);
            Field3D<T>  M_tmp[3];
            Field3D<T>  C_tmp[3];
            for (int q = 0; q < 3; ++q) {
                M_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
                C_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
            }
            compute_bssn_constraints(grid, grid.Ricci, H_tmp, M_tmp, C_tmp, r_min, r_max, 0.0, 0.0,
                                     0.0);
            auto stats = compute_constraint_monitor(grid, H_tmp, padding_);
            populate_constraint_norms(grid, M_tmp, stats, padding_);
            if (monitor_callback_)
                monitor_callback_(grid, stats);
        }

        if (snapshot_callback_)
            snapshot_callback_(grid, step_index);

#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
        tensorium_RG::bssn::detail::finalize_contracted_warning_step();
#endif
        report_kernel_timers(step_index);
    }

    void validate_backend_options() const {
        if (backend_options_.backend != tensorium::backend::Kind::CUDA)
            return;
        if (!tensorium::cuda::is_available()) {
            throw std::runtime_error(
                "BSSNRKStepper requested the CUDA backend, but no CUDA device is available.");
        }
    }

    [[nodiscard]] bool cuda_state_matches_grid(const BSSNGridSoA<T> &grid) const noexcept {
        return cuda_stepper_state_ && dims_match(cuda_stepper_state_->grid.dims, grid.dims) &&
               strides_match(cuda_stepper_state_->grid.st, grid.st);
    }

    void ensure_cuda_stepper_state(const BSSNGridSoA<T> &grid) {
        if (cuda_state_matches_grid(grid))
            return;
        cuda_stepper_state_ = std::make_unique<BSSNCUDAStepperState<T>>();
        cuda_stepper_state_->allocate_like(grid);
    }

    void sync_host_grid_to_cuda(const BSSNGridSoA<T> &grid) {
        cuda_stepper_state_->grid.copy_from_host(grid);
    }

    void sync_cuda_grid_to_host(BSSNGridSoA<T> &grid) const { cuda_stepper_state_->grid.copy_to_host(grid); }

    void sync_host_rhs_to_cuda(const BSSNRHSWorkspace<T> &rhs) {
        cuda_stepper_state_->rhs_workspace.alpha.copy_from_host(rhs.alpha);
        cuda_stepper_state_->rhs_workspace.chi.copy_from_host(rhs.chi);
        cuda_stepper_state_->rhs_workspace.K.copy_from_host(rhs.K);
        cuda_stepper_state_->rhs_workspace.Theta.copy_from_host(rhs.Theta);
        for (int c = 0; c < 3; ++c) {
            cuda_stepper_state_->rhs_workspace.beta[c].copy_from_host(rhs.beta[c]);
            cuda_stepper_state_->rhs_workspace.B[c].copy_from_host(rhs.B[c]);
            cuda_stepper_state_->rhs_workspace.tildeGamma[c].copy_from_host(rhs.tildeGamma[c]);
            cuda_stepper_state_->rhs_workspace.Z[c].copy_from_host(rhs.Z[c]);
        }
        for (int s = 0; s < 6; ++s) {
            cuda_stepper_state_->rhs_workspace.gamma_tilde[s].copy_from_host(rhs.gamma_tilde[s]);
            cuda_stepper_state_->rhs_workspace.A_tilde[s].copy_from_host(rhs.A_tilde[s]);
        }
    }

    void sync_host_cpu_rhs_terms_to_cuda(const BSSNRHSWorkspace<T> &rhs) {
        for (int c = 0; c < 3; ++c)
            cuda_stepper_state_->rhs_workspace.tildeGamma[c].copy_from_host(rhs.tildeGamma[c]);
        for (int s = 0; s < 6; ++s) {
            cuda_stepper_state_->rhs_workspace.gamma_tilde[s].copy_from_host(rhs.gamma_tilde[s]);
            cuda_stepper_state_->rhs_workspace.A_tilde[s].copy_from_host(rhs.A_tilde[s]);
        }
    }

    void sync_host_theta_trace_cache_to_cuda() {
        cuda_stepper_state_->theta_ricciz4_trace_cache.copy_from_host(theta_ricciz4_trace_cache_);
    }

    [[nodiscard]] bool can_use_cuda_accelerated_rhs(const GaugeParameters<T> &params) const noexcept {
        return !params.use_direct_shift_rhs;
    }

    [[nodiscard]] tensorium_RG::bssn::cuda::GaugeRHSCudaConfig<T>
    make_cuda_gauge_rhs_config(const GaugeParameters<T> &params) const {
        tensorium_RG::bssn::cuda::GaugeRHSCudaConfig<T> cfg;
        cfg.ko_sigma = scaled_ko_sigma(params.ko_sigma);
        cfg.ko_boundary_floor = T(current_ko_boundary_floor());
        cfg.beta_B_coeff = params.beta_B_coeff;
        cfg.eta_coeff = params.effective_eta();
        cfg.lapse_harmonicf = params.lapse_harmonicf;
        cfg.lapse_harmonic = params.lapse_harmonic;
        cfg.lapse_oplog = params.lapse_oplog;
        cfg.lapse_advect = params.lapse_advect;
        cfg.shift_advect = params.shift_advect;
        cfg.slow_start_lapse_factor = params.slow_start_lapse_factor();
        cfg.spatial_order = tensorium_RG::fd::max_spatial_derivative_order();
        cfg.ko_boundary_width = current_ko_boundary_width();
        cfg.use_theta_in_lapse = params.use_theta_in_lapse;
        cfg.use_shift_advection = params.use_shift_advection;
        return cfg;
    }

    [[nodiscard]] tensorium_RG::bssn::cuda::ScalarRHSCudaConfig<T>
    make_cuda_scalar_rhs_config(const GaugeParameters<T> &params) const {
        tensorium_RG::bssn::cuda::ScalarRHSCudaConfig<T> cfg;
        cfg.ko_sigma = scaled_ko_sigma(params.ko_sigma);
        cfg.ko_boundary_floor = T(current_ko_boundary_floor());
        cfg.kappa1 = params.kappa1;
        cfg.kappa2 = params.kappa2;
        cfg.chi_div_floor = params.chi_div_floor;
        cfg.spatial_order = tensorium_RG::fd::max_spatial_derivative_order();
        cfg.ko_boundary_width = current_ko_boundary_width();
        cfg.covariant_z4 = params.covariant_z4;
        return cfg;
    }

    [[nodiscard]] tensorium_RG::bssn::cuda::RHSSommerfeldCudaConfig<T>
    make_cuda_rhs_sommerfeld_config() const {
        tensorium_RG::bssn::cuda::RHSSommerfeldCudaConfig<T> cfg;
        cfg.collar_width = BoundaryRadiative::rhs_collar();
        for (int axis = 0; axis < 3; ++axis) {
            cfg.face_enabled[axis][0] = BoundaryRadiative::active_enabled(axis, false) &&
                                        BoundaryRadiative::rhs_sommerfeld_enabled(axis, false) &&
                                        !BoundaryRadiative::reflective_enabled(axis, false);
            cfg.face_enabled[axis][1] = BoundaryRadiative::active_enabled(axis, true) &&
                                        BoundaryRadiative::rhs_sommerfeld_enabled(axis, true) &&
                                        !BoundaryRadiative::reflective_enabled(axis, true);
        }
        cfg.alpha_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::Alpha));
        cfg.chi_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::Chi));
        cfg.K_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::K));
        cfg.Theta_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::Theta));
        cfg.beta_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::Beta));
        cfg.B_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::B));
        cfg.tildeGamma_speed =
            T(BoundaryRadiative::characteristic_speed_for(BoundaryField::TildeGamma));
        cfg.Z_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::Z));
        cfg.gamma_tilde_speed =
            T(BoundaryRadiative::characteristic_speed_for(BoundaryField::GammaTilde));
        cfg.A_tilde_speed = T(BoundaryRadiative::characteristic_speed_for(BoundaryField::ATilde));
        return cfg;
    }

    [[nodiscard]] tensorium_RG::bssn::cuda::RHSSpongeCudaConfig<T> make_cuda_rhs_sponge_config()
        const {
        tensorium_RG::bssn::cuda::RHSSpongeCudaConfig<T> cfg;
        const auto host_cfg = BoundaryRadiative::sponge_config;
        cfg.enabled = host_cfg.enabled;
        cfg.width = host_cfg.width;
        cfg.strength = T(host_cfg.strength);
        cfg.exponent = T(host_cfg.exponent);
        for (int axis = 0; axis < 3; ++axis)
            for (int side = 0; side < 2; ++side) {
                cfg.active_face[axis][side] = BoundaryRadiative::active_face[axis][side];
                cfg.reflective_face[axis][side] = BoundaryRadiative::reflective_face[axis][side];
            }
        return cfg;
    }

    void apply_cuda_post_stage_corrections() {
        auto device_grid_view = cuda_stepper_state_->grid.view();
        tensorium_RG::bssn::cuda::enforce_algebraic_constraints(device_grid_view);
        if (gauge_params_.alpha_floor > T(0))
            tensorium_RG::bssn::cuda::apply_alpha_floor(device_grid_view, gauge_params_.alpha_floor);
        if (gauge_params_.chi_floor > T(0))
            tensorium_RG::bssn::cuda::apply_chi_floor(device_grid_view, gauge_params_.chi_floor);
        tensorium::cuda::device_synchronize();
    }

    size_t rhs_bulk_padding() const {
        if constexpr (BoundaryPhysicalEvolutionTraits<Boundary>::value) {
            // Radiative boundaries use per-face padding via boundary_rhs_padding().
            return 0;
        }
        return padding_;
    }

    static constexpr size_t required_radiative_stencil_collar() noexcept {
        // The widest RHS stencil is the upwind advection stencil, which reaches four cells
        // one-sided. The Tensorium physical Sommerfeld collar must cover that region, otherwise
        // bulk cells consume extrapolated ghosts before the RHS boundary operator replaces them.
        return size_t(4);
    }

    size_t effective_radiative_rhs_collar() const {
        if constexpr (!BoundaryRadiativeRHSTraits<Boundary>::value)
            return size_t(0);
        return std::max(BoundaryRadiativeRHSTraits<Boundary>::collar(),
                        required_radiative_stencil_collar());
    }

    struct BoundaryRhsPadding {
        bool active = false;
        size_t lower[3] = {0, 0, 0};
        size_t upper[3] = {0, 0, 0};
    };

    BoundaryRhsPadding boundary_rhs_padding() const {
        BoundaryRhsPadding out{};
        if constexpr (!BoundaryRadiativeRHSTraits<Boundary>::value)
            return out;
        if (!gauge_params_.apply_rhs_sommerfeld)
            return out;

        const size_t collar = effective_radiative_rhs_collar();
        for (int axis = 0; axis < 3; ++axis) {
            const bool lower_active = BoundaryRadiative::active_enabled(axis, false) &&
                                      BoundaryRadiative::rhs_sommerfeld_enabled(axis, false) &&
                                      !BoundaryRadiative::reflective_enabled(axis, false);
            const bool upper_active = BoundaryRadiative::active_enabled(axis, true) &&
                                      BoundaryRadiative::rhs_sommerfeld_enabled(axis, true) &&
                                      !BoundaryRadiative::reflective_enabled(axis, true);
            out.lower[axis] = lower_active ? collar : size_t(0);
            out.upper[axis] = upper_active ? collar : size_t(0);
            out.active = out.active || lower_active || upper_active;
        }
        return out;
    }

    size_t evolution_padding() const {
        if constexpr (BoundaryPhysicalEvolutionTraits<Boundary>::value)
            return 0;
        return padding_;
    }

    size_t                                                                      padding_ = 4;
    tensorium::backend::Options                                                backend_options_{};
    GaugeParameters<T>                                                          gauge_params_{};
    BSSNRHSWorkspace<T>                                                         stage_rhs_;
    StageState                                                                  stage_grid_;
    Field3D<T>                                                                  theta_ricciz4_trace_cache_;
    std::unique_ptr<BSSNCUDAStepperState<T>>                                    cuda_stepper_state_;
    std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> monitor_callback_;
    std::function<void(const BSSNGridSoA<T> &, size_t)>                         snapshot_callback_;
    std::function<void(BSSNRHSWorkspace<T> &)>                                  rhs_prep_callback_;
    size_t                                                                       state_log_stride_ = 1;
    T                                                                           boundary_dt_ = T(0);
    T                                                                           simulation_time_ = T(0);

    void add_dissipation(const Field3D<T> &u, Field3D<T> &rhs, T sigma) {
        const size_t nx = u.st.nx_tot;
        const size_t ny = u.st.ny_tot;
        const size_t nz = u.st.nz_tot;
        const T     *in = u.ptr();
        T           *out = rhs.ptr();
        const T      coef = sigma / 16.0;

#pragma omp parallel for collapse(3)
        for (size_t i = 2; i < nx - 2; ++i) {
            for (size_t j = 2; j < ny - 2; ++j) {
                for (size_t k = 2; k < nz - 2; ++k) {
                    size_t idx = i * u.st.sx + j * u.st.sy + k * u.st.sz;
                    T diss_x = in[idx - 2 * u.st.sx] - 4.0 * in[idx - u.st.sx] + 6.0 * in[idx] -
                               4.0 * in[idx + u.st.sx] + in[idx + 2 * u.st.sx];
                    T diss_y = in[idx - 2 * u.st.sy] - 4.0 * in[idx - u.st.sy] + 6.0 * in[idx] -
                               4.0 * in[idx + u.st.sy] + in[idx + 2 * u.st.sy];
                    T diss_z = in[idx - 2 * u.st.sz] - 4.0 * in[idx - u.st.sz] + 6.0 * in[idx] -
                               4.0 * in[idx + u.st.sz] + in[idx + 2 * u.st.sz];
                    out[idx] -= coef * (diss_x + diss_y + diss_z);
                }
            }
        }
    }

    /// @brief Smoothly limit alpha using Hyperbola smoothing to preserve C1 continuity
    void apply_alpha_floor(BSSNGridSoA<T> &grid, T floor) {
        const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
        T           *ptr = grid.alpha.ptr();
        // IMPORTANT: Use higher safety floor (e.g. 1e-4) in caller
#pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            ptr[idx] = detail::smooth_floor(ptr[idx], floor);
        }
    }

    /// @brief Enforce hard positivity floors before RHS assembly (GRChombo-style pre-RHS clamp).
    void enforce_positive_chi_alpha_pre_rhs(BSSNGridSoA<T> &grid) {
        const bool do_alpha = gauge_params_.alpha_floor > T(0);
        const bool do_chi = gauge_params_.chi_floor > T(0);
        if (!do_alpha && !do_chi)
            return;

        const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
        T           *alpha = grid.alpha.ptr();
        T           *chi = grid.chi.ptr();
        const T      alpha_floor = gauge_params_.alpha_floor;
        const T      chi_floor = gauge_params_.chi_floor;

#pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            if (do_alpha && alpha[idx] < alpha_floor)
                alpha[idx] = alpha_floor;
            if (do_chi && chi[idx] < chi_floor)
                chi[idx] = chi_floor;
        }
    }

    /// @brief Project det/tr, apply boundaries, and refresh derivatives before RHS evaluation.
    void prepare_state_for_rhs(BSSNGridSoA<T> &grid) {
        enforce_positive_chi_alpha_pre_rhs(grid);
        configure_boundary_characteristics();
        apply_halos_grid<Boundary>(grid);
        enforce_positive_chi_alpha_pre_rhs(grid);

        rebuild_geometry(grid);
        synchronize_z_from_gamma_constraint(grid);
    }

    void rebuild_geometry(BSSNGridSoA<T> &grid) {
        tensorium_RG::bssn::enforce_algebraic_constraints(grid);
        if constexpr (BoundaryRadiativeRHSTraits<Boundary>::value) {
            size_t I0, I1, J0, J1, K0, K1;
            grid.domain_bounds(I0, I1, J0, J1, K0, K1);
            const auto pad = boundary_rhs_padding();
            const size_t i0 = std::min(I0 + pad.lower[0], I1);
            const size_t i1 = (I1 > pad.upper[0]) ? I1 - pad.upper[0] : I1;
            const size_t j0 = std::min(J0 + pad.lower[1], J1);
            const size_t j1 = (J1 > pad.upper[1]) ? J1 - pad.upper[1] : J1;
            const size_t k0 = std::min(K0 + pad.lower[2], K1);
            const size_t k1 = (K1 > pad.upper[2]) ? K1 - pad.upper[2] : K1;
            if (i0 < i1 && j0 < j1 && k0 < k1)
                compute_ricci_bssn_region(grid, grid.Ricci, i0, i1, j0, j1, k0, k1, false);
        } else {
            compute_ricci_bssn(grid, grid.Ricci, false);
        }
    }

    void synchronize_z_from_gamma_constraint(BSSNGridSoA<T> &grid) {
        using namespace tensorium_RG::fd;
        const double inv_12dx = 1.0 / (60.0 * grid.dx);
        const double inv_12dy = 1.0 / (60.0 * grid.dy);
        const double inv_12dz = 1.0 / (60.0 * grid.dz);
        const ptrdiff_t sx = grid.alpha.st.sx;
        const ptrdiff_t sy = grid.alpha.st.sy;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i) {
            for (size_t j = J0; j < J1; ++j) {
                for (size_t k = K0; k < K1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    const T      chi = grid.chi.ptr()[idx];
                    const T      chi_guarded = guard_chi_div(chi, gauge_params_.chi_div_floor);

                    const T *p_xx = grid.gamma_tilde_inv[XX].ptr() + idx;
                    const T *p_xy = grid.gamma_tilde_inv[XY].ptr() + idx;
                    const T *p_xz = grid.gamma_tilde_inv[XZ].ptr() + idx;
                    const T *p_yy = grid.gamma_tilde_inv[YY].ptr() + idx;
                    const T *p_yz = grid.gamma_tilde_inv[YZ].ptr() + idx;
                    const T *p_zz = grid.gamma_tilde_inv[ZZ].ptr() + idx;

                    T z_over_chi[3] = {T(0), T(0), T(0)};
                    recover_z_over_chi_from_gamma_ptr(
                        grid.tildeGamma[0].ptr() + idx, grid.tildeGamma[1].ptr() + idx,
                        grid.tildeGamma[2].ptr() + idx, p_xx, p_xy, p_xz, p_yy, p_yz, p_zz, sx,
                        sy, inv_12dx, inv_12dy, inv_12dz, z_over_chi);

                    grid.Z[0].ptr()[idx] = chi_guarded * z_over_chi[0];
                    grid.Z[1].ptr()[idx] = chi_guarded * z_over_chi[1];
                    grid.Z[2].ptr()[idx] = chi_guarded * z_over_chi[2];
                }
            }
        }
    }

    void configure_boundary_characteristics() {
        constexpr double default_wave_speed = 1.0;
        BoundaryFieldSpeedConfigurator<Boundary>::set(
            double(gauge_params_.boundary_gauge_characteristic_speed),
            double(gauge_params_.boundary_z4c_characteristic_speed),
            double(gauge_params_.boundary_khat_characteristic_speed));
        BoundaryConfigurator<Boundary>::set_characteristic(default_wave_speed,
                                                           double(boundary_dt_));
    }

    /// @brief Smoothly limit chi to keep metric well conditioned
    void apply_chi_floor(BSSNGridSoA<T> &grid, T floor) {
        const size_t total = grid.chi.st.nx_tot * grid.chi.st.ny_tot * grid.chi.st.nz_tot;
        T           *ptr = grid.chi.ptr();
#pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx)
            ptr[idx] = detail::smooth_floor(ptr[idx], floor);
    }

    /// @brief Invoke every RHS kernel using the provided grid snapshot.
    void evaluate_rhs(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs,
                      const GaugeParameters<T> &params) {
        if (backend_options_.backend == tensorium::backend::Kind::CUDA &&
            can_use_cuda_accelerated_rhs(params)) {
            evaluate_rhs_cuda_accelerated(grid, rhs, params);
            return;
        }
        evaluate_rhs_cpu(grid, rhs, params);
    }

    void evaluate_rhs_cpu(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs,
                          const GaugeParameters<T> &params) {
        if (rhs_prep_callback_)
            rhs_prep_callback_(rhs);
        update_ko_scale(grid, boundary_dt_);
        const size_t rhs_padding = rhs_bulk_padding();
        const auto boundary_padding = boundary_rhs_padding();
        if (boundary_padding.active) {
            ScopedInteriorPaddingOverride boundary_scope(
                boundary_padding.lower[0], boundary_padding.upper[0], boundary_padding.lower[1],
                boundary_padding.upper[1], boundary_padding.lower[2], boundary_padding.upper[2]);
            evaluate_rhs_sweep_core(grid, rhs.alpha, rhs.chi, rhs.K, rhs.Theta, rhs.beta, rhs.B,
                                    rhs.gamma_tilde, rhs.A_tilde, rhs.tildeGamma, rhs.Z, params,
                                    rhs_padding, &theta_ricciz4_trace_cache_);
        } else {
            evaluate_rhs_sweep_core(grid, rhs.alpha, rhs.chi, rhs.K, rhs.Theta, rhs.beta, rhs.B,
                                    rhs.gamma_tilde, rhs.A_tilde, rhs.tildeGamma, rhs.Z, params,
                                    rhs_padding, &theta_ricciz4_trace_cache_);
        }
        apply_rhs_sommerfeld(grid, rhs);
        apply_rhs_sponge(grid, rhs);
        if (rhs_prep_callback_)
            stabilize_nonfinite_rhs_collar(grid, rhs, rhs_padding);
        recompose_rhs_K_from_khat_core(grid, rhs.K, rhs.Theta, evolution_padding());
    }

    void evaluate_rhs_cuda_accelerated(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs,
                                       const GaugeParameters<T> &params) {
        if (rhs_prep_callback_)
            rhs_prep_callback_(rhs);
        update_ko_scale(grid, boundary_dt_);
        const size_t rhs_padding = rhs_bulk_padding();
        const auto   device_grid_view =
            static_cast<const BSSNGridDevice<T> &>(cuda_stepper_state_->grid).view();
        auto device_rhs_view = cuda_stepper_state_->rhs_workspace.view();

        compute_rhs_Gamma(grid, rhs.tildeGamma, grid.Z, grid.Theta, params, rhs_padding);
        compute_rhs_gamma_tilde(grid, rhs.gamma_tilde, rhs_padding, params);
        compute_rhs_A_tilde(grid, rhs.A_tilde, rhs_padding, params, &theta_ricciz4_trace_cache_);

        sync_host_cpu_rhs_terms_to_cuda(rhs);
        sync_host_theta_trace_cache_to_cuda();
        tensorium_RG::bssn::cuda::compute_gauge_rhs(
            device_grid_view, device_rhs_view, make_cuda_gauge_rhs_config(params), rhs_padding);
        tensorium_RG::bssn::cuda::compute_scalar_rhs(
            device_grid_view, device_rhs_view,
            static_cast<const DeviceField3D<T> &>(cuda_stepper_state_->theta_ricciz4_trace_cache)
                .view(),
            make_cuda_scalar_rhs_config(params),
            rhs_padding);

        if constexpr (BoundaryRadiativeRHSTraits<Boundary>::value) {
            if (gauge_params_.apply_rhs_sommerfeld)
                tensorium_RG::bssn::cuda::apply_rhs_sommerfeld(
                    device_grid_view, device_rhs_view, make_cuda_rhs_sommerfeld_config());
            tensorium_RG::bssn::cuda::apply_rhs_sponge(device_grid_view, device_rhs_view,
                                                       make_cuda_rhs_sponge_config());
        }
        if (rhs_prep_callback_) {
            tensorium_RG::bssn::cuda::stabilize_nonfinite_rhs_collar(device_grid_view,
                                                                     device_rhs_view, rhs_padding);
        }
        tensorium_RG::bssn::cuda::recompose_rhs_K_from_khat(device_grid_view, device_rhs_view,
                                                            evolution_padding());
        tensorium::cuda::device_synchronize();
    }

    void stabilize_nonfinite_rhs_collar(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs,
                                        size_t collar_width) {
        if (collar_width == 0)
            return;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        if (I0 >= I1 || J0 >= J1 || K0 >= K1)
            return;

        auto reset_nonfinite = [](Field3D<T> &field, size_t idx) {
            T &slot = field.ptr()[idx];
            if (!std::isfinite(static_cast<double>(slot)))
                slot = T(0);
        };

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k) {
                    const bool in_collar = (i - I0 < collar_width) || (I1 - 1 - i < collar_width) ||
                                           (j - J0 < collar_width) || (J1 - 1 - j < collar_width) ||
                                           (k - K0 < collar_width) || (K1 - 1 - k < collar_width);
                    if (!in_collar)
                        continue;

                    const size_t idx = grid.alpha.idx(i, j, k);
                    reset_nonfinite(rhs.alpha, idx);
                    reset_nonfinite(rhs.chi, idx);
                    reset_nonfinite(rhs.K, idx);
                    reset_nonfinite(rhs.Theta, idx);
                    for (int c = 0; c < 3; ++c) {
                        reset_nonfinite(rhs.beta[c], idx);
                        reset_nonfinite(rhs.B[c], idx);
                        reset_nonfinite(rhs.tildeGamma[c], idx);
                        reset_nonfinite(rhs.Z[c], idx);
                    }
                    for (int s = 0; s < 6; ++s) {
                        reset_nonfinite(rhs.gamma_tilde[s], idx);
                        reset_nonfinite(rhs.A_tilde[s], idx);
                    }
                }
    }

    void apply_rhs_sommerfeld(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs) {
        if (!gauge_params_.apply_rhs_sommerfeld)
            return;
        if constexpr (!BoundaryRadiativeRHSTraits<Boundary>::value)
            return;

        RHSBoundaryFaceMask mask{};
        for (int axis = 0; axis < 3; ++axis) {
            mask.face[axis][0] = BoundaryRadiative::active_enabled(axis, false) &&
                                 BoundaryRadiative::rhs_sommerfeld_enabled(axis, false) &&
                                 !BoundaryRadiative::reflective_enabled(axis, false);
            mask.face[axis][1] = BoundaryRadiative::active_enabled(axis, true) &&
                                 BoundaryRadiative::rhs_sommerfeld_enabled(axis, true) &&
                                 !BoundaryRadiative::reflective_enabled(axis, true);
        }
        const size_t physical_surface_width = effective_radiative_rhs_collar();
        tensorium_RG::bssn::apply_z4c_rhs_boundary(
            grid, rhs, mask, physical_surface_width);
    }

    void apply_rhs_sponge(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs) {
        if constexpr (!BoundaryRadiativeRHSTraits<Boundary>::value)
            return;

        const auto cfg = BoundaryRadiative::sponge_config;
        if (!cfg.enabled || cfg.width == 0 || cfg.strength <= 0.0)
            return;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        if (I0 >= I1 || J0 >= J1 || K0 >= K1)
            return;

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k) {
                    size_t layer = 0;
                    if (!tensorium_RG::bssn::detail::nearest_sponge_layer(
                            grid, i, j, k, cfg, BoundaryRadiative::active_face,
                            BoundaryRadiative::reflective_face, layer))
                        continue;

                    const T sigma = T(tensorium_RG::bssn::detail::sponge_profile_value(layer, cfg));
                    if (sigma <= T(0))
                        continue;

                    const size_t id = grid.alpha.idx(i, j, k);
                    auto add_damping = [&](Field3D<T> &target_rhs, const Field3D<T> &field,
                                           BoundaryField which, int component) {
                        const T asymptotic =
                            T(tensorium_RG::bssn::detail::minkowski_target(which, component));
                        target_rhs.ptr()[id] += sigma * (asymptotic - field.ptr()[id]);
                    };

                    add_damping(rhs.alpha, grid.alpha, BoundaryField::Alpha, 0);
                    add_damping(rhs.chi, grid.chi, BoundaryField::Chi, 0);
                    rhs.K.ptr()[id] += sigma * (T(0) - Khat(grid.K.ptr()[id], grid.Theta.ptr()[id]));
                    add_damping(rhs.Theta, grid.Theta, BoundaryField::Theta, 0);
                    for (int a = 0; a < 3; ++a) {
                        add_damping(rhs.beta[a], grid.beta[a], BoundaryField::Beta, a);
                        add_damping(rhs.B[a], grid.B[a], BoundaryField::B, a);
                        add_damping(rhs.tildeGamma[a], grid.tildeGamma[a], BoundaryField::TildeGamma,
                                    a);
                        add_damping(rhs.Z[a], grid.Z[a], BoundaryField::Z, a);
                    }
                    for (int s = 0; s < 6; ++s) {
                        add_damping(rhs.gamma_tilde[s], grid.gamma_tilde[s],
                                    BoundaryField::GammaTilde, s);
                        add_damping(rhs.A_tilde[s], grid.A_tilde[s], BoundaryField::ATilde, s);
                    }
                }
    }

    void log_gauge_diagnostics(const BSSNGridSoA<T> &grid, size_t step_index) {
        if (step_index > 10)
            return;
        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        const size_t guard = std::max(padding_, size_t(2));
        const size_t i0 = std::min(I0 + guard, I1);
        const size_t j0 = std::min(J0 + guard, J1);
        const size_t k0 = std::min(K0 + guard, K1);
        const size_t i1 = (I1 > guard) ? I1 - guard : I1;
        const size_t j1 = (J1 > guard) ? J1 - guard : J1;
        const size_t k1 = (K1 > guard) ? K1 - guard : K1;

        double max_beta = 0.0, max_B = 0.0;
        double min_alpha = 1e100, min_chi = 1e100;
        double max_gamma = 0.0, max_det = 0.0, max_trA = 0.0;
        double min_theta = std::numeric_limits<double>::infinity();
        double max_theta = -std::numeric_limits<double>::infinity();

        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                for (size_t k = k0; k < k1; ++k) {
                    const size_t idx = grid.alpha.idx(i, j, k);
                    const double alpha = grid.alpha.ptr()[idx];
                    const double chi = grid.chi.ptr()[idx];
                    const double bx = grid.beta[0].ptr()[idx];
                    const double by = grid.beta[1].ptr()[idx];
                    const double bz = grid.beta[2].ptr()[idx];
                    const double Bx = grid.B[0].ptr()[idx];
                    const double By = grid.B[1].ptr()[idx];
                    const double Bz = grid.B[2].ptr()[idx];

                    max_beta = std::max(max_beta, std::sqrt(bx * bx + by * by + bz * bz));
                    max_B = std::max(max_B, std::sqrt(Bx * Bx + By * By + Bz * Bz));
                    min_alpha = std::min(min_alpha, alpha);
                    min_chi = std::min(min_chi, chi);
                    const double theta = grid.Theta.ptr()[idx];
                    if (std::isfinite(theta)) {
                        min_theta = std::min(min_theta, theta);
                        max_theta = std::max(max_theta, theta);
                    }
                    max_gamma = std::max({max_gamma, std::abs(grid.tildeGamma[0].ptr()[idx]),
                                          std::abs(grid.tildeGamma[1].ptr()[idx]),
                                          std::abs(grid.tildeGamma[2].ptr()[idx])});

                    const double gxx = grid.gamma_tilde[XX].ptr()[idx];
                    const double gxy = grid.gamma_tilde[XY].ptr()[idx];
                    const double gxz = grid.gamma_tilde[XZ].ptr()[idx];
                    const double gyy = grid.gamma_tilde[YY].ptr()[idx];
                    const double gyz = grid.gamma_tilde[YZ].ptr()[idx];
                    const double gzz = grid.gamma_tilde[ZZ].ptr()[idx];
                    const double det = gxx * (gyy * gzz - gyz * gyz) -
                                       gxy * (gxy * gzz - gxz * gyz) +
                                       gxz * (gxy * gyz - gxz * gyy);
                    max_det = std::max(max_det, std::abs(det - 1.0));

                    const double trA =
                        grid.gamma_tilde_inv[XX].ptr()[idx] * grid.A_tilde[XX].ptr()[idx] +
                        grid.gamma_tilde_inv[YY].ptr()[idx] * grid.A_tilde[YY].ptr()[idx] +
                        grid.gamma_tilde_inv[ZZ].ptr()[idx] * grid.A_tilde[ZZ].ptr()[idx] +
                        2.0 * (grid.gamma_tilde_inv[XY].ptr()[idx] * grid.A_tilde[XY].ptr()[idx] +
                               grid.gamma_tilde_inv[XZ].ptr()[idx] * grid.A_tilde[XZ].ptr()[idx] +
                               grid.gamma_tilde_inv[YZ].ptr()[idx] * grid.A_tilde[YZ].ptr()[idx]);
                    max_trA = std::max(max_trA, std::abs(trA));
                }

        const double theta_min_print = std::isfinite(min_theta) ? min_theta : 0.0;
        const double theta_max_print = std::isfinite(max_theta) ? max_theta : 0.0;
        printf("[Gauge %02zu]\n"
               "\talpha_min=% .3e\tchi_min=% .3e\tTheta_min=% .3e\tTheta_max=% .3e\n"
               "\tbeta_max=% .3e\tB_max=% .3e\tGamma_max=% .3e\n"
               "\tdet_err=% .3e\ttrA=% .3e\n",
               step_index, min_alpha, min_chi, theta_min_print, theta_max_print, max_beta, max_B,
               max_gamma, max_det, max_trA);
    }

    void copy_scalar_interior(const BSSNGridSoA<T> &grid, const Field3D<T> &src, Field3D<T> &dst) {
        const T *in = src.ptr();
        T       *out = dst.ptr();
        for_each_interior_index_parallel(grid, evolution_padding(),
                                         [&](size_t, size_t, size_t, size_t idx) {
                                             out[idx] = in[idx];
                                         });
    }

    void add_scaled_scalar_interior(const BSSNGridSoA<T> &grid, const Field3D<T> &src,
                                    Field3D<T> &dst, T scale) {
        const T *in = src.ptr();
        T       *out = dst.ptr();
        for_each_interior_index_parallel(grid, evolution_padding(),
                                         [&](size_t, size_t, size_t, size_t idx) {
                                             out[idx] += scale * in[idx];
                                         });
    }

    void update_scalar_interior(const BSSNGridSoA<T> &grid, Field3D<T> &u0, const Field3D<T> &u1,
                                const Field3D<T> &rhs, T gam0, T gam1, T beta_dt) {
        T       *out = u0.ptr();
        const T *base = u1.ptr();
        const T *k = rhs.ptr();
        for_each_interior_index_parallel(grid, evolution_padding(),
                                         [&](size_t, size_t, size_t, size_t idx) {
                                             out[idx] =
                                                 gam0 * out[idx] + gam1 * base[idx] + beta_dt * k[idx];
                                         });
    }

    void copy_stage_reference(const BSSNGridSoA<T> &src, StageState &dst) {
        dst.x0 = src.x0;
        dst.y0 = src.y0;
        dst.z0 = src.z0;
        copy_scalar_interior(src, src.alpha, dst.alpha);
        copy_scalar_interior(src, src.chi, dst.chi);
        copy_scalar_interior(src, src.K, dst.K);
        copy_scalar_interior(src, src.Theta, dst.Theta);
        for (int i = 0; i < 3; ++i) {
            copy_scalar_interior(src, src.beta[i], dst.beta[i]);
            copy_scalar_interior(src, src.B[i], dst.B[i]);
            copy_scalar_interior(src, src.tildeGamma[i], dst.tildeGamma[i]);
            copy_scalar_interior(src, src.Z[i], dst.Z[i]);
        }
        for (int s = 0; s < 6; ++s) {
            copy_scalar_interior(src, src.gamma_tilde[s], dst.gamma_tilde[s]);
            copy_scalar_interior(src, src.A_tilde[s], dst.A_tilde[s]);
        }
    }

    void accumulate_stage_reference(const BSSNGridSoA<T> &src, StageState &dst, T delta) {
        add_scaled_scalar_interior(src, src.alpha, dst.alpha, delta);
        add_scaled_scalar_interior(src, src.chi, dst.chi, delta);
        add_scaled_scalar_interior(src, src.K, dst.K, delta);
        add_scaled_scalar_interior(src, src.Theta, dst.Theta, delta);
        for (int i = 0; i < 3; ++i) {
            add_scaled_scalar_interior(src, src.beta[i], dst.beta[i], delta);
            add_scaled_scalar_interior(src, src.B[i], dst.B[i], delta);
            add_scaled_scalar_interior(src, src.tildeGamma[i], dst.tildeGamma[i], delta);
            add_scaled_scalar_interior(src, src.Z[i], dst.Z[i], delta);
        }
        for (int s = 0; s < 6; ++s) {
            add_scaled_scalar_interior(src, src.gamma_tilde[s], dst.gamma_tilde[s], delta);
            add_scaled_scalar_interior(src, src.A_tilde[s], dst.A_tilde[s], delta);
        }
    }

    void apply_explicit_stage_update(BSSNGridSoA<T> &u0, const StageState &u1,
                                     const BSSNRHSWorkspace<T> &rhs, T gam0, T gam1, T beta_dt) {
        update_scalar_interior(u0, u0.alpha, u1.alpha, rhs.alpha, gam0, gam1, beta_dt);
        update_scalar_interior(u0, u0.chi, u1.chi, rhs.chi, gam0, gam1, beta_dt);
        update_scalar_interior(u0, u0.K, u1.K, rhs.K, gam0, gam1, beta_dt);
        update_scalar_interior(u0, u0.Theta, u1.Theta, rhs.Theta, gam0, gam1, beta_dt);
        for (int i = 0; i < 3; ++i) {
            update_scalar_interior(u0, u0.beta[i], u1.beta[i], rhs.beta[i], gam0, gam1, beta_dt);
            update_scalar_interior(u0, u0.B[i], u1.B[i], rhs.B[i], gam0, gam1, beta_dt);
            update_scalar_interior(u0, u0.tildeGamma[i], u1.tildeGamma[i], rhs.tildeGamma[i], gam0,
                                   gam1, beta_dt);
            update_scalar_interior(u0, u0.Z[i], u1.Z[i], rhs.Z[i], gam0, gam1, beta_dt);
        }
        for (int s = 0; s < 6; ++s) {
            update_scalar_interior(u0, u0.gamma_tilde[s], u1.gamma_tilde[s], rhs.gamma_tilde[s],
                                   gam0, gam1, beta_dt);
            update_scalar_interior(u0, u0.A_tilde[s], u1.A_tilde[s], rhs.A_tilde[s], gam0, gam1,
                                   beta_dt);
        }
    }
};

} // namespace tensorium_RG::bssn

namespace tensorium_RG::bssn {

template <typename T, typename Boundary> using Z4cRKStepper = BSSNRKStepper<T, Boundary>;

} // namespace tensorium_RG::bssn
