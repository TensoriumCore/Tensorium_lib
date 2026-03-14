#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
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
#include "../Geometry/BSSNAlgebraic.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include "../Geometry/BSSNProjectionMonitor.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "BSSNPerfTimers.hpp"
#include "BSSNRHSSweep.hpp"

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

template <typename Boundary>
struct BoundaryConfigurator<Boundary, std::void_t<decltype(Boundary::set_characteristic(
                                     std::declval<double>(), std::declval<double>()))>> {
    static inline void set_characteristic(double speed, double dt) {
        Boundary::set_characteristic(speed, dt);
    }
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
    explicit BSSNRKStepper(const BSSNGridSoA<T> &prototype, size_t padding = 4)
        : padding_(padding),
          stage_grid_(prototype.dims.nx, prototype.dims.ny, prototype.dims.nz, prototype.dims.ng,
                      prototype.dx, prototype.dy, prototype.dz) {
        stage_grid_.x0 = prototype.x0;
        stage_grid_.y0 = prototype.y0;
        stage_grid_.z0 = prototype.z0;
        for (auto &stage : stages_)
            stage.allocate_like(prototype);
        detail::allocate_like(prototype.alpha, theta_ricciz4_trace_cache_);
    }

    void set_gauge_parameters(const GaugeParameters<T> &params) { gauge_params_ = params; }

    void set_constraint_callback(
        std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> cb) {
        monitor_callback_ = std::move(cb);
    }

    void set_snapshot_callback(std::function<void(const BSSNGridSoA<T> &, size_t)> cb) {
        snapshot_callback_ = std::move(cb);
    }

    void set_state_log_stride(size_t stride) { state_log_stride_ = std::max<size_t>(size_t(1), stride); }

    void set_rhs_prep_callback(std::function<void(BSSNRHSWorkspace<T> &)> cb) {
        rhs_prep_callback_ = std::move(cb);
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

        boundary_dt_ = dt;
        for (int stage = 0; stage < 4; ++stage) {
            if (stage == 0)
                copy_stage_reference(grid, stage_grid_);
            else
                accumulate_stage_reference(grid, stage_grid_, T(delta_ref[stage]));

            prepare_state_for_rhs(grid);
            auto stage_params = gauge_params_;
            stage_params.current_time = simulation_time_;
            stage_params.frozen_Z_is_synced = !stage_params.evolve_Z;
            evaluate_rhs(grid, stages_[stage], stage_params);

            apply_explicit_stage_update(grid, stage_grid_, stages_[stage], T(gam0_ref[stage]),
                                        T(gam1_ref[stage]), T(beta_ref[stage]) * dt);
            if (gauge_params_.alpha_floor > T(0))
                apply_alpha_floor(grid, gauge_params_.alpha_floor);
            if (gauge_params_.chi_floor > T(0))
                apply_chi_floor(grid, gauge_params_.chi_floor);
        }
        simulation_time_ += dt;

        if (gauge_params_.alpha_floor > T(0))
            apply_alpha_floor(grid, gauge_params_.alpha_floor);
        if (gauge_params_.chi_floor > T(0))
            apply_chi_floor(grid, gauge_params_.chi_floor);

        const bool do_state_log = ((step_index + 1) % state_log_stride_ == 0);
        const bool needs_post_step_prepare =
            do_state_log || static_cast<bool>(monitor_callback_) || static_cast<bool>(snapshot_callback_);
        if (needs_post_step_prepare) {
            prepare_state_for_rhs(grid);
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

  private:
    size_t                                                                      padding_ = 4;
    GaugeParameters<T>                                                          gauge_params_{};
    BSSNRHSWorkspace<T>                                                         stages_[4];
    BSSNGridSoA<T>                                                              stage_grid_;
    Field3D<T>                                                                  theta_ricciz4_trace_cache_;
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
        if (!gauge_params_.evolve_Z) {
            synchronize_z_from_gamma_constraint(grid);
            apply_halos_grid<Boundary>(grid);
        }
    }

    void rebuild_geometry(BSSNGridSoA<T> &grid) {
        tensorium_RG::bssn::enforce_algebraic_constraints(grid);
        compute_ricci_bssn(grid, grid.Ricci, false);
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
        if (rhs_prep_callback_)
            rhs_prep_callback_(rhs);
        update_ko_scale(grid, boundary_dt_);
        evaluate_rhs_sweep_core(grid, rhs.alpha, rhs.chi, rhs.K, rhs.Theta, rhs.beta, rhs.B,
                                rhs.gamma_tilde, rhs.A_tilde, rhs.tildeGamma, rhs.Z, params,
                                padding_, &theta_ricciz4_trace_cache_);
        apply_rhs_sommerfeld(grid, rhs);
        recompose_rhs_K_from_khat_core(grid, rhs.K, rhs.Theta, padding_);
    }

    void apply_rhs_sommerfeld(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs) {
        if (!gauge_params_.apply_rhs_sommerfeld)
            return;
        if constexpr (!std::is_same_v<Boundary, BoundaryRadiative>)
            return;

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

        const T inv_2dx = T(0.5) / grid.dx;
        const T inv_2dy = T(0.5) / grid.dy;
        const T inv_2dz = T(0.5) / grid.dz;
        const T inv_r_floor = T(1e-12);

#pragma omp parallel for collapse(3)
        for (size_t i = i0; i < i1; ++i) {
            for (size_t j = j0; j < j1; ++j) {
                for (size_t k = k0; k < k1; ++k) {
                    const bool on_ix1 = (i == i0) &&
                                        BoundaryRadiative::rhs_sommerfeld_enabled(0, false) &&
                                        !BoundaryRadiative::reflective_enabled(0, false);
                    const bool on_ox1 =
                        (i + 1 == i1) && BoundaryRadiative::rhs_sommerfeld_enabled(0, true) &&
                        !BoundaryRadiative::reflective_enabled(0, true);
                    const bool on_ix2 = (j == j0) &&
                                        BoundaryRadiative::rhs_sommerfeld_enabled(1, false) &&
                                        !BoundaryRadiative::reflective_enabled(1, false);
                    const bool on_ox2 =
                        (j + 1 == j1) && BoundaryRadiative::rhs_sommerfeld_enabled(1, true) &&
                        !BoundaryRadiative::reflective_enabled(1, true);
                    const bool on_ix3 = (k == k0) &&
                                        BoundaryRadiative::rhs_sommerfeld_enabled(2, false) &&
                                        !BoundaryRadiative::reflective_enabled(2, false);
                    const bool on_ox3 =
                        (k + 1 == k1) && BoundaryRadiative::rhs_sommerfeld_enabled(2, true) &&
                        !BoundaryRadiative::reflective_enabled(2, true);
                    const bool on_surface = on_ix1 || on_ox1 || on_ix2 || on_ox2 || on_ix3 || on_ox3;
                    if (!on_surface)
                        continue;

                    const size_t id = grid.alpha.idx(i, j, k);
                    T            x, y, z;
                    grid.coords(i, j, k, x, y, z);
                    const T r = std::sqrt(x * x + y * y + z * z);
                    const T inv_r = T(1) / std::max(r, inv_r_floor);
                    const T sx = x * inv_r;
                    const T sy = y * inv_r;
                    const T sz = z * inv_r;

                    auto radial_derivative = [&](const Field3D<T> &f) -> T {
                        const T *p = f.ptr() + id;
                        const T  dfx = (p[f.st.sx] - p[-f.st.sx]) * inv_2dx;
                        const T  dfy = (p[f.st.sy] - p[-f.st.sy]) * inv_2dy;
                        const T  dfz = (p[1] - p[-1]) * inv_2dz;
                        return sx * dfx + sy * dfy + sz * dfz;
                    };

                    auto sommerfeld_rhs = [&](const Field3D<T> &u, T asymptotic) -> T {
                        const T u0 = u.ptr()[id];
                        return -radial_derivative(u) + (asymptotic - u0) * inv_r;
                    };

                    rhs.alpha.ptr()[id] = sommerfeld_rhs(grid.alpha, T(1));
                    rhs.chi.ptr()[id] = sommerfeld_rhs(grid.chi, T(1));
                    rhs.Theta.ptr()[id] = sommerfeld_rhs(grid.Theta, T(0));
                    // rhs.K currently stores rhs(Khat); enforce Sommerfeld on K then convert.
                    const T rhs_K = sommerfeld_rhs(grid.K, T(0));
                    rhs.K.ptr()[id] = rhs_K - T(2) * rhs.Theta.ptr()[id];

                    for (int a = 0; a < 3; ++a)
                        rhs.beta[a].ptr()[id] = sommerfeld_rhs(grid.beta[a], T(0));
                    for (int a = 0; a < 3; ++a)
                        rhs.B[a].ptr()[id] = sommerfeld_rhs(grid.B[a], T(0));
                    for (int a = 0; a < 3; ++a)
                        rhs.tildeGamma[a].ptr()[id] = sommerfeld_rhs(grid.tildeGamma[a], T(0));
                    for (int a = 0; a < 3; ++a)
                        rhs.Z[a].ptr()[id] = sommerfeld_rhs(grid.Z[a], T(0));

                    rhs.gamma_tilde[XX].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[XX], T(1));
                    rhs.gamma_tilde[XY].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[XY], T(0));
                    rhs.gamma_tilde[XZ].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[XZ], T(0));
                    rhs.gamma_tilde[YY].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[YY], T(1));
                    rhs.gamma_tilde[YZ].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[YZ], T(0));
                    rhs.gamma_tilde[ZZ].ptr()[id] = sommerfeld_rhs(grid.gamma_tilde[ZZ], T(1));

                    for (int s = 0; s < 6; ++s)
                        rhs.A_tilde[s].ptr()[id] = sommerfeld_rhs(grid.A_tilde[s], T(0));
                }
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
        for_each_interior_index_parallel(grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
            out[idx] = in[idx];
        });
    }

    void add_scaled_scalar_interior(const BSSNGridSoA<T> &grid, const Field3D<T> &src,
                                    Field3D<T> &dst, T scale) {
        const T *in = src.ptr();
        T       *out = dst.ptr();
        for_each_interior_index_parallel(grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
            out[idx] += scale * in[idx];
        });
    }

    void update_scalar_interior(const BSSNGridSoA<T> &grid, Field3D<T> &u0, const Field3D<T> &u1,
                                const Field3D<T> &rhs, T gam0, T gam1, T beta_dt) {
        T       *out = u0.ptr();
        const T *base = u1.ptr();
        const T *k = rhs.ptr();
        for_each_interior_index_parallel(grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
            out[idx] = gam0 * out[idx] + gam1 * base[idx] + beta_dt * k[idx];
        });
    }

    void copy_stage_reference(const BSSNGridSoA<T> &src, BSSNGridSoA<T> &dst) {
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

    void accumulate_stage_reference(const BSSNGridSoA<T> &src, BSSNGridSoA<T> &dst, T delta) {
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

    void apply_explicit_stage_update(BSSNGridSoA<T> &u0, const BSSNGridSoA<T> &u1,
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
        tensorium_RG::bssn::enforce_algebraic_constraints(u0);
    }
};

} // namespace tensorium_RG::bssn
