#pragma once
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

    T   max_beta = T(0);
    T   min_alpha = std::numeric_limits<T>::infinity();
    int gradient_flag = 0;
#pragma omp parallel for collapse(3) reduction(max : max_beta) reduction(min : min_alpha)                     \
    reduction(max : gradient_flag)
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                const T      alpha = std::abs(grid.alpha.ptr()[id]);
                const T      bx = grid.beta[0].ptr()[id];
                const T      by = grid.beta[1].ptr()[id];
                const T      bz = grid.beta[2].ptr()[id];
                const T      theta_val = grid.Theta.ptr()[id];

                const T beta_mag = std::sqrt(bx * bx + by * by + bz * bz);

                max_beta = std::max(max_beta, beta_mag);
                min_alpha = std::min(min_alpha, alpha);
                // Gauge-shock sensors: strong shifts, collapsing lapse, or large Theta.
                if (alpha < T(0.1) || beta_mag > T(0.8) || std::abs(theta_val) > T(1.0))
                    gradient_flag = 1;
            }

    const T alpha_floor = T(0.1);
    const T a = std::max(min_alpha, alpha_floor);

    const T gauge_term = control.gauge_speed * std::sqrt(T(1) / a);

    T denom = max_beta + gauge_term;
    if (denom <= T(0))
        denom = T(1);

    T cfl_number = control.cfl;
    if (gradient_flag != 0)
        cfl_number = std::min(cfl_number, T(0.12));

    return cfl_number * (min_dx / denom);
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
    }

    void set_gauge_parameters(const GaugeParameters<T> &params) { gauge_params_ = params; }

    void set_constraint_callback(
        std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> cb) {
        monitor_callback_ = std::move(cb);
    }

    void set_snapshot_callback(std::function<void(const BSSNGridSoA<T> &, size_t)> cb) {
        snapshot_callback_ = std::move(cb);
    }

    void set_rhs_prep_callback(std::function<void(BSSNRHSWorkspace<T> &)> cb) {
        rhs_prep_callback_ = std::move(cb);
    }

    // Diagnostics function (omitted mostly unchanged for brevity in logic check, but included in
    // full paste)
    void log_full_bssn_diagnostics(const BSSNGridSoA<T> &grid, size_t step, double r_min,
                                   double r_max, double chi_cut = 0.0) {
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
        boundary_dt_ = dt;
        prepare_state_for_rhs(grid);
        evaluate_rhs(grid, stages_[0]);

        build_stage_state(grid, stages_[0], dt * T(0.5));
        prepare_state_for_rhs(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[1]);

        build_stage_state(grid, stages_[1], dt * T(0.5));
        prepare_state_for_rhs(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[2]);

        build_stage_state(grid, stages_[2], dt);
        prepare_state_for_rhs(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[3]);

        apply_rk_update(grid, dt);
        prepare_state_for_rhs(grid);

        const double min_extent =
            std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                      (grid.dims.nz - 1) * grid.dz});
        const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
        const double r_max = 0.45 * min_extent;

        log_gauge_diagnostics(grid, step_index);

        const double chi_cut = 1e-7;
        log_full_bssn_diagnostics(grid, step_index, r_min, r_max, chi_cut);

        if (monitor_callback_ || snapshot_callback_) {
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
    std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> monitor_callback_;
    std::function<void(const BSSNGridSoA<T> &, size_t)>                         snapshot_callback_;
    std::function<void(BSSNRHSWorkspace<T> &)>                                  rhs_prep_callback_;
    T                                                                           boundary_dt_ = T(0);

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

    /// @brief Project det/tr, apply boundaries, and refresh derivatives before RHS evaluation.
    void prepare_state_for_rhs(BSSNGridSoA<T> &grid) {
        configure_boundary_characteristics();
        tensorium_RG::bssn::enforce_algebraic_constraints(grid);
        apply_halos_grid<Boundary>(grid);

        apply_chi_floor(grid, T(1e-8));
        apply_alpha_floor(grid, T(1e-8));

        rebuild_geometry(grid);
    }

    void rebuild_geometry(BSSNGridSoA<T> &grid) {
        tensorium_RG::bssn::enforce_algebraic_constraints(grid);
        ProjectionConfig cfg;
        cfg.padding = padding_;
        cfg.renormalize_metric = false;
        cfg.project_A_tilde = false;
        cfg.recompute_inverse = false;
        cfg.resync_contracted_gamma = true;
        project_bssn_state(grid, cfg);
        compute_ricci_bssn(grid, grid.Ricci, false);
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
    void evaluate_rhs(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs) {
        if (rhs_prep_callback_)
            rhs_prep_callback_(rhs);
        update_ko_scale(grid, boundary_dt_);
        compute_rhs_Gamma(grid, rhs.tildeGamma, grid.Z, grid.Theta, gauge_params_, padding_);
        compute_rhs_B(grid, rhs.tildeGamma, rhs.B, gauge_params_, padding_);
        compute_rhs_beta(grid, rhs.beta, gauge_params_, padding_);
        compute_rhs_alpha(grid, rhs.alpha, padding_, gauge_params_);
        compute_rhs_chi(grid, rhs.chi, padding_, gauge_params_);
        compute_rhs_gamma_tilde(grid, rhs.gamma_tilde, padding_);
        compute_rhs_A_tilde(grid, rhs.A_tilde, padding_, gauge_params_);
        compute_rhs_K(grid, rhs.K, padding_, gauge_params_);
        compute_rhs_Theta(grid, rhs.Theta, gauge_params_, padding_);
        compute_rhs_Z(grid, rhs.Z, gauge_params_, padding_);
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

    void blend_field_interior(const BSSNGridSoA<T> &grid, const Field3D<T> &base,
                              const Field3D<T> &delta, Field3D<T> &dst, T scale_dt) {
        const size_t total = base.st.nx_tot * base.st.ny_tot * base.st.nz_tot;
        std::copy(base.ptr(), base.ptr() + total, dst.ptr());
        for_each_interior_index(grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
            dst.ptr()[idx] += scale_dt * delta.ptr()[idx];
        });
    }

    void build_stage_state(const BSSNGridSoA<T> &base, const BSSNRHSWorkspace<T> &delta,
                           T scale_dt) {
        stage_grid_.x0 = base.x0;
        stage_grid_.y0 = base.y0;
        stage_grid_.z0 = base.z0;
        blend_field_interior(base, base.alpha, delta.alpha, stage_grid_.alpha, scale_dt);
        blend_field_interior(base, base.chi, delta.chi, stage_grid_.chi, scale_dt);
        blend_field_interior(base, base.K, delta.K, stage_grid_.K, scale_dt);
        blend_field_interior(base, base.Theta, delta.Theta, stage_grid_.Theta, scale_dt);
        for (int i = 0; i < 3; ++i) {
            blend_field_interior(base, base.beta[i], delta.beta[i], stage_grid_.beta[i], scale_dt);
            blend_field_interior(base, base.B[i], delta.B[i], stage_grid_.B[i], scale_dt);
            blend_field_interior(base, base.tildeGamma[i], delta.tildeGamma[i],
                                 stage_grid_.tildeGamma[i], scale_dt);
            blend_field_interior(base, base.Z[i], delta.Z[i], stage_grid_.Z[i], scale_dt);
        }
        for (int s = 0; s < 6; ++s) {
            blend_field_interior(base, base.gamma_tilde[s], delta.gamma_tilde[s],
                                 stage_grid_.gamma_tilde[s], scale_dt);
            blend_field_interior(base, base.A_tilde[s], delta.A_tilde[s], stage_grid_.A_tilde[s],
                                 scale_dt);
        }
    }

    void accumulate_scalar(const BSSNGridSoA<T> &grid, Field3D<T> &dest, const Field3D<T> &k1,
                           const Field3D<T> &k2, const Field3D<T> &k3, const Field3D<T> &k4, T c1,
                           T c2, T c3, T c4) {
        T       *out = dest.ptr();
        const T *p1 = k1.ptr();
        const T *p2 = k2.ptr();
        const T *p3 = k3.ptr();
        const T *p4 = k4.ptr();
        for_each_interior_index(grid, padding_, [&](size_t, size_t, size_t, size_t idx) {
            out[idx] += c1 * p1[idx] + c2 * p2[idx] + c3 * p3[idx] + c4 * p4[idx];
        });
    }

    void apply_rk_update(BSSNGridSoA<T> &grid, T dt) {
        const T c1 = dt / T(6);
        const T c2 = dt / T(3);
        const T c3 = dt / T(3);
        const T c4 = dt / T(6);
        accumulate_scalar(grid, grid.alpha, stages_[0].alpha, stages_[1].alpha, stages_[2].alpha,
                          stages_[3].alpha, c1, c2, c3, c4);
        accumulate_scalar(grid, grid.chi, stages_[0].chi, stages_[1].chi, stages_[2].chi,
                          stages_[3].chi, c1, c2, c3, c4);
        accumulate_scalar(grid, grid.K, stages_[0].K, stages_[1].K, stages_[2].K, stages_[3].K, c1,
                          c2, c3, c4);
        accumulate_scalar(grid, grid.Theta, stages_[0].Theta, stages_[1].Theta, stages_[2].Theta,
                          stages_[3].Theta, c1, c2, c3, c4);
        for (int i = 0; i < 3; ++i) {
            accumulate_scalar(grid, grid.beta[i], stages_[0].beta[i], stages_[1].beta[i],
                              stages_[2].beta[i], stages_[3].beta[i], c1, c2, c3, c4);
            accumulate_scalar(grid, grid.B[i], stages_[0].B[i], stages_[1].B[i], stages_[2].B[i],
                              stages_[3].B[i], c1, c2, c3, c4);
            accumulate_scalar(grid, grid.tildeGamma[i], stages_[0].tildeGamma[i],
                              stages_[1].tildeGamma[i], stages_[2].tildeGamma[i],
                              stages_[3].tildeGamma[i], c1, c2, c3, c4);
            accumulate_scalar(grid, grid.Z[i], stages_[0].Z[i], stages_[1].Z[i], stages_[2].Z[i],
                              stages_[3].Z[i], c1, c2, c3, c4);
        }
        for (int s = 0; s < 6; ++s) {
            accumulate_scalar(grid, grid.gamma_tilde[s], stages_[0].gamma_tilde[s],
                              stages_[1].gamma_tilde[s], stages_[2].gamma_tilde[s],
                              stages_[3].gamma_tilde[s], c1, c2, c3, c4);
            accumulate_scalar(grid, grid.A_tilde[s], stages_[0].A_tilde[s], stages_[1].A_tilde[s],
                              stages_[2].A_tilde[s], stages_[3].A_tilde[s], c1, c2, c3, c4);
        }

        tensorium_RG::bssn::enforce_algebraic_constraints(grid);
    }
};

} // namespace tensorium_RG::bssn
