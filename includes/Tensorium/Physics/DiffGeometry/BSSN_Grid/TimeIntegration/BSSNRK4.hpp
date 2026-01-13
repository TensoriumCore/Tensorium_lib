#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>

#include "../Evolution/BSSNEvolutionATilde.hpp"
#include "../Evolution/BSSNEvolutionChi.hpp"
#include "../Evolution/BSSNEvolutionGamma.hpp"
#include "../Evolution/BSSNEvolutionGammaTilde.hpp"
#include "../Evolution/BSSNEvolutionGauge.hpp"
#include "../Evolution/BSSNEvolutionK.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include "../Geometry/BSSNProjectionMonitor.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Grid/BSSNGridOperations.hpp"

/**
 * @file BSSNRK4.hpp
 * @brief Fourth-order Runge–Kutta time integrator that orchestrates halo updates, geometry rebuilds,
 *        RHS evaluations, and constraint monitoring.
 * @details The step sequence follows: halo application → geometry projection/inversion → RHS per
 * stage → stage blending → final projection + diagnostics.  The driver templated on a `Boundary`
 * functor so users can attach periodic, radiative, or clamp boundary conditions without modifying
 * the physics kernels.
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

template <typename T> inline void copy_blend(const Field3D<T> &base, const Field3D<T> &delta,
                                              Field3D<T> &dst, T scale_dt) {
    const size_t total = base.st.nx_tot * base.st.ny_tot * base.st.nz_tot;
    const T    *base_ptr = base.ptr();
    const T    *delta_ptr = delta.ptr();
    T          *dst_ptr = dst.ptr();
    for (size_t idx = 0; idx < total; ++idx)
        dst_ptr[idx] = base_ptr[idx] + scale_dt * delta_ptr[idx];
}

} // namespace detail

/// @brief Stores RHS buffers for every evolved variable.
template <typename T> struct BSSNRHSWorkspace {
    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;
    Field3D<T> beta[3];
    Field3D<T> B[3];
    Field3D<T> gamma_tilde[6];
    Field3D<T> A_tilde[6];
    Field3D<T> tildeGamma[3];

    /// @brief Allocate RHS buffers with the same strides as the prototype grid.
    void allocate_like(const BSSNGridSoA<T> &grid) {
        detail::allocate_like(grid.alpha, alpha);
        detail::allocate_like(grid.chi, chi);
        detail::allocate_like(grid.K, K);
        for (int i = 0; i < 3; ++i) {
            detail::allocate_like(grid.beta[i], beta[i]);
            detail::allocate_like(grid.B[i], B[i]);
            detail::allocate_like(grid.tildeGamma[i], tildeGamma[i]);
        }
        for (int s = 0; s < 6; ++s) {
            detail::allocate_like(grid.gamma_tilde[s], gamma_tilde[s]);
            detail::allocate_like(grid.A_tilde[s], A_tilde[s]);
        }
    }

    /// @brief Reset all RHS accumulators to zero before reuse.
    void zero() {
        detail::zero_field(alpha);
        detail::zero_field(chi);
        detail::zero_field(K);
        detail::zero_fields(beta, 3);
        detail::zero_fields(B, 3);
        detail::zero_fields(gamma_tilde, 6);
        detail::zero_fields(A_tilde, 6);
        detail::zero_fields(tildeGamma, 3);
    }
};

/// @brief Compute a timestep satisfying \f$dt = \,\text{CFL}\times \min(\Delta x)/(|\vec{\beta}|+\alpha_{max})\f$.
template <typename T>
inline T compute_dt_cfl(const BSSNGridSoA<T> &grid, T cfl_factor, size_t padding = 4) {
    const T min_dx = std::min({grid.dx, grid.dy, grid.dz});
    size_t  I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i_begin = std::min(I0 + guard, I1);
    const size_t j_begin = std::min(J0 + guard, J1);
    const size_t k_begin = std::min(K0 + guard, K1);
    const size_t i_end = (I1 > guard) ? I1 - guard : I1;
    const size_t j_end = (J1 > guard) ? J1 - guard : J1;
    const size_t k_end = (K1 > guard) ? K1 - guard : K1;

    T max_speed = T(0);
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                const T alpha = grid.alpha.ptr()[id];
                const T beta_x = grid.beta[0].ptr()[id];
                const T beta_y = grid.beta[1].ptr()[id];
                const T beta_z = grid.beta[2].ptr()[id];
                const T beta_mag = std::sqrt(beta_x * beta_x + beta_y * beta_y + beta_z * beta_z);
                max_speed = std::max(max_speed, beta_mag + alpha);
            }
    if (max_speed <= T(0))
        max_speed = T(1);
    return cfl_factor * (min_dx / max_speed);
}

/**
 * @brief Classical RK4 stepper that manages halo exchanges, geometry refresh, and diagnostics.
 * @tparam Boundary Boundary functor satisfying the interface expected by `apply_halos_grid`.
 */
template <typename T, typename Boundary> class BSSNRKStepper {
  public:
    /// @brief Build a stepper that clones the SoA layout for stage grids.
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

    /// @brief Override the default \f(\beta^i,B^i\f) driver coefficients.
    void set_gauge_parameters(const GaugeParameters<T> &params) { gauge_params_ = params; }

    /// @brief Register a callback invoked after each step with constraint statistics.
    void set_constraint_callback(
        std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> cb) {
        monitor_callback_ = std::move(cb);
    }

    /// @brief Register a callback invoked after each step (or stage) for I/O or visualization.
    void set_snapshot_callback(std::function<void(const BSSNGridSoA<T> &, size_t)> cb) {
        snapshot_callback_ = std::move(cb);
    }

    /**
     * @brief Advance the grid by `dt` using four RK stages.
     * @details Each stage performs halo application → geometry refresh → RHS evaluation.  After the
     * final combination the state is projected onto the algebraic manifold and optional monitors are
     * invoked.
     */
    void step(BSSNGridSoA<T> &grid, T dt, size_t step_index = 0) {
        // Stage 1
        apply_halos_grid<Boundary>(grid);
        ensure_geometry(grid);
        apply_halos_grid<Boundary>(grid);
        evaluate_rhs(grid, stages_[0]);

        // Stage 2
        build_stage_state(grid, stages_[0], dt * T(0.5));
        apply_halos_grid<Boundary>(stage_grid_);
        ensure_geometry(stage_grid_);
        apply_halos_grid<Boundary>(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[1]);

        // Stage 3
        build_stage_state(grid, stages_[1], dt * T(0.5));
        apply_halos_grid<Boundary>(stage_grid_);
        ensure_geometry(stage_grid_);
        apply_halos_grid<Boundary>(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[2]);

        // Stage 4
        build_stage_state(grid, stages_[2], dt);
        apply_halos_grid<Boundary>(stage_grid_);
        ensure_geometry(stage_grid_);
        apply_halos_grid<Boundary>(stage_grid_);
        evaluate_rhs(stage_grid_, stages_[3]);

        // Combine updates.
        apply_rk_update(grid, dt);
        ensure_geometry(grid);
        apply_halos_grid<Boundary>(grid);
        log_gauge_diagnostics(grid, step_index);

        if (monitor_callback_ || snapshot_callback_) {
            const double min_extent =
                std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                          (grid.dims.nz - 1) * grid.dz});
            const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
            const double r_max = 0.45 * min_extent;
            compute_bssn_constraints(grid, grid.Ricci, grid.Hc, grid.Mc, grid.Cc, r_min, r_max, 0.0,
                                     0.0, 0.0);
            const auto stats = compute_constraint_monitor(grid, grid.Hc, padding_);
            if (monitor_callback_)
                monitor_callback_(grid, stats);
        }
        if (snapshot_callback_)
            snapshot_callback_(grid, step_index);
    }

  private:
    size_t padding_ = 4;
    GaugeParameters<T> gauge_params_{};
    BSSNRHSWorkspace<T> stages_[4];
    BSSNGridSoA<T> stage_grid_;
    std::function<void(const BSSNGridSoA<T> &, const ConstraintMonitorStats &)> monitor_callback_;
    std::function<void(const BSSNGridSoA<T> &, size_t)> snapshot_callback_;

    /// @brief Reproject the grid, floor \f$\chi\f$, and rebuild Christoffels and Ricci.
    void ensure_geometry(BSSNGridSoA<T> &grid) {
        project_bssn_after_update(grid, padding_);
        apply_chi_floor(grid, T(1e-6));
        compute_tildeGamma_full(grid, grid.Gamma_tilde);
        compute_ricci_bssn(grid, grid.Ricci, false);
    }

    /// @brief Prevent \f$\chi\f$ from reaching zero to keep \f$\gamma_{ij}\f$ well conditioned.
    void apply_chi_floor(BSSNGridSoA<T> &grid, T floor) {
        const size_t total = grid.chi.st.nx_tot * grid.chi.st.ny_tot * grid.chi.st.nz_tot;
        T           *ptr = grid.chi.ptr();
        for (size_t idx = 0; idx < total; ++idx)
            ptr[idx] = std::max(ptr[idx], floor);
    }

    /// @brief Invoke every RHS kernel using the provided grid snapshot.
    void evaluate_rhs(const BSSNGridSoA<T> &grid, BSSNRHSWorkspace<T> &rhs) {
        rhs.zero();
        compute_rhs_Gamma(grid, rhs.tildeGamma, padding_);
        compute_rhs_B(grid, rhs.tildeGamma, rhs.B, gauge_params_, padding_);
        compute_rhs_beta(grid, rhs.beta, gauge_params_, padding_);
        compute_rhs_alpha(grid, rhs.alpha, padding_);
        compute_rhs_chi(grid, rhs.chi, padding_);
        compute_rhs_gamma_tilde(grid, rhs.gamma_tilde, padding_);
        compute_rhs_A_tilde(grid, rhs.A_tilde, padding_);
        compute_rhs_K(grid, rhs.K, padding_);
    }

    /// @brief Emit one-line diagnostics on the gauge sector for early steps.
    void log_gauge_diagnostics(const BSSNGridSoA<T> &grid, size_t step_index) {
        if (step_index > 10)
            return;

        const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
        double max_beta = 0.0;
        double max_B = 0.0;
        double min_alpha = std::numeric_limits<double>::infinity();
        double min_chi = std::numeric_limits<double>::infinity();
        double max_gamma = 0.0;
        double max_det_drift = 0.0;
        double max_trace_A = 0.0;

        using tensorium_RG::XX;
        using tensorium_RG::XY;
        using tensorium_RG::XZ;
        using tensorium_RG::YY;
        using tensorium_RG::YZ;
        using tensorium_RG::ZZ;

        for (size_t idx = 0; idx < total; ++idx) {
            const double alpha = double(grid.alpha.ptr()[idx]);
            const double chi = double(grid.chi.ptr()[idx]);
            const double bx = double(grid.beta[0].ptr()[idx]);
            const double by = double(grid.beta[1].ptr()[idx]);
            const double bz = double(grid.beta[2].ptr()[idx]);
            const double Bx = double(grid.B[0].ptr()[idx]);
            const double By = double(grid.B[1].ptr()[idx]);
            const double Bz = double(grid.B[2].ptr()[idx]);
            const double gamma0 = double(grid.tildeGamma[0].ptr()[idx]);
            const double gamma1 = double(grid.tildeGamma[1].ptr()[idx]);
            const double gamma2 = double(grid.tildeGamma[2].ptr()[idx]);

            max_beta = std::max(max_beta, std::sqrt(bx * bx + by * by + bz * bz));
            max_B = std::max(max_B, std::sqrt(Bx * Bx + By * By + Bz * Bz));
            max_gamma = std::max({max_gamma, std::abs(gamma0), std::abs(gamma1), std::abs(gamma2)});
            min_alpha = std::min(min_alpha, alpha);
            min_chi = std::min(min_chi, chi);

            double gxx = double(grid.gamma_tilde[XX].ptr()[idx]);
            double gxy = double(grid.gamma_tilde[XY].ptr()[idx]);
            double gxz = double(grid.gamma_tilde[XZ].ptr()[idx]);
            double gyy = double(grid.gamma_tilde[YY].ptr()[idx]);
            double gyz = double(grid.gamma_tilde[YZ].ptr()[idx]);
            double gzz = double(grid.gamma_tilde[ZZ].ptr()[idx]);

            const double det = gxx * (gyy * gzz - gyz * gyz) -
                               gxy * (gxy * gzz - gxz * gyz) +
                               gxz * (gxy * gyz - gxz * gyy);
            max_det_drift = std::max(max_det_drift, std::abs(det - 1.0));

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

            const double traceA = gixx * Axx + giyy * Ayy + gizz * Azz +
                                  2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
            max_trace_A = std::max(max_trace_A, std::abs(traceA));
        }

        printf("[GaugeDiag step=%zu] max|beta|=%.3e max|B|=%.3e min(alpha)=%.3e min(chi)=%.3e "
               "max|Gamma|=%.3e max_det_drift=%.3e max_traceA=%.3e\n",
               step_index, max_beta, max_B, min_alpha, min_chi, max_gamma, max_det_drift,
               max_trace_A);
    }

    /// @brief Form \f$q^{(n)} + c \Delta t\,k^{(n)}\f$ for the intermediate RK stages.
    void build_stage_state(const BSSNGridSoA<T> &base, const BSSNRHSWorkspace<T> &delta,
                           T scale_dt) {
        stage_grid_.x0 = base.x0;
        stage_grid_.y0 = base.y0;
        stage_grid_.z0 = base.z0;
        detail::copy_blend(base.alpha, delta.alpha, stage_grid_.alpha, scale_dt);
        detail::copy_blend(base.chi, delta.chi, stage_grid_.chi, scale_dt);
        detail::copy_blend(base.K, delta.K, stage_grid_.K, scale_dt);
        for (int i = 0; i < 3; ++i) {
            detail::copy_blend(base.beta[i], delta.beta[i], stage_grid_.beta[i], scale_dt);
            detail::copy_blend(base.B[i], delta.B[i], stage_grid_.B[i], scale_dt);
            detail::copy_blend(base.tildeGamma[i], delta.tildeGamma[i], stage_grid_.tildeGamma[i],
                               scale_dt);
        }
        for (int s = 0; s < 6; ++s) {
            detail::copy_blend(base.gamma_tilde[s], delta.gamma_tilde[s], stage_grid_.gamma_tilde[s],
                               scale_dt);
            detail::copy_blend(base.A_tilde[s], delta.A_tilde[s], stage_grid_.A_tilde[s], scale_dt);
        }
    }

    void accumulate_scalar(Field3D<T> &dest, const Field3D<T> &k1, const Field3D<T> &k2,
                           const Field3D<T> &k3, const Field3D<T> &k4, T c1, T c2, T c3, T c4) {
        const size_t total = dest.st.nx_tot * dest.st.ny_tot * dest.st.nz_tot;
        T           *out = dest.ptr();
        const T     *p1 = k1.ptr();
        const T     *p2 = k2.ptr();
        const T     *p3 = k3.ptr();
        const T     *p4 = k4.ptr();
        for (size_t idx = 0; idx < total; ++idx)
            out[idx] += c1 * p1[idx] + c2 * p2[idx] + c3 * p3[idx] + c4 * p4[idx];
    }

    /// @brief Combine the four RK stages with weights (1,2,2,1)/6.
    void apply_rk_update(BSSNGridSoA<T> &grid, T dt) {
        const T c1 = dt / T(6);
        const T c2 = dt / T(3);
        const T c3 = dt / T(3);
        const T c4 = dt / T(6);
        accumulate_scalar(grid.alpha, stages_[0].alpha, stages_[1].alpha, stages_[2].alpha,
                          stages_[3].alpha, c1, c2, c3, c4);
        accumulate_scalar(grid.chi, stages_[0].chi, stages_[1].chi, stages_[2].chi, stages_[3].chi, c1,
                          c2, c3, c4);
        accumulate_scalar(grid.K, stages_[0].K, stages_[1].K, stages_[2].K, stages_[3].K, c1, c2, c3,
                          c4);
        for (int i = 0; i < 3; ++i) {
            accumulate_scalar(grid.beta[i], stages_[0].beta[i], stages_[1].beta[i], stages_[2].beta[i],
                              stages_[3].beta[i], c1, c2, c3, c4);
            accumulate_scalar(grid.B[i], stages_[0].B[i], stages_[1].B[i], stages_[2].B[i],
                              stages_[3].B[i], c1, c2, c3, c4);
            accumulate_scalar(grid.tildeGamma[i], stages_[0].tildeGamma[i], stages_[1].tildeGamma[i],
                              stages_[2].tildeGamma[i], stages_[3].tildeGamma[i], c1, c2, c3, c4);
        }
        for (int s = 0; s < 6; ++s) {
            accumulate_scalar(grid.gamma_tilde[s], stages_[0].gamma_tilde[s], stages_[1].gamma_tilde[s],
                              stages_[2].gamma_tilde[s], stages_[3].gamma_tilde[s], c1, c2, c3, c4);
            accumulate_scalar(grid.A_tilde[s], stages_[0].A_tilde[s], stages_[1].A_tilde[s],
                              stages_[2].A_tilde[s], stages_[3].A_tilde[s], c1, c2, c3, c4);
        }
    }
};

} // namespace tensorium_RG::bssn
