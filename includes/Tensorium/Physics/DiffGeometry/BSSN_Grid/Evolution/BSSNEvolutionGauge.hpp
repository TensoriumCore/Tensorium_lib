#pragma once

#include <algorithm>
#include <limits>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../TimeIntegration/BSSNPerfTimers.hpp"
#include "BSSNEvolutionCommon.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionGauge.hpp
 * @brief Gauge drivers for \f$\alpha\f$, \f$\beta^i\f$, and \f$B^i\f$ (1+log + Gamma-driver).
 * @details Documents Eq. (5.43) of Baumgarte–Shapiro:
 * \f$(\partial_t-\mathcal{L}_\beta)\alpha=-2\alpha K\f$, and the hyperbolic shift driver
 * \f$(\partial_t-\mathcal{L}_\beta)\beta^i = (3/4)B^i\f$,
 * \f$(\partial_t-\mathcal{L}_\beta)B^i =
 * (\partial_t\tilde{\Gamma}^i-\mathcal{L}_\beta\tilde{\Gamma}^i) - \eta B^i\f$.
 */

namespace tensorium_RG::bssn {

/// @brief Tunable coefficients for the Gamma-driver system.
template <typename T> struct GaugeParameters {
    T beta_B_coeff = T(0.75);
    T eta = T(2);
    T mass_scale = T(1);
    T kappa1 = T(0.0);
    T kappa2 = T(0.0);
    T kappa3 = T(1.0);
    T kappa_z = T(1.0);
    bool covariant_z4 = true;         ///< If true use kappa1 (not lapse*kappa1) in Z4 damping.
    T chi_div_floor = T(-1000.0);     ///< Guard for divisions by chi: use max(chi, chi_div_floor).
    T ko_sigma = T(0.0);              ///< KO6 filter strength.
    T min_lapse_for_K = T(5e-3);      ///< Floor applied when sourcing the K quadratic term.
    T max_K_squared = T(1e4);         ///< Safety cap for K^2 when alpha collapses.
    T alpha_floor = T(0.0);           ///< Optional post-RK floor for alpha (0 disables).
    T chi_floor = T(0.0);             ///< Optional post-RK floor for chi (0 disables).
    bool use_theta_in_lapse = true;       ///< Z4c-compatible lapse source with Khat = K - 2 Theta.
    bool use_shift_advection = true;      ///< If false, drop β·∂ terms in the Gamma-driver.
    T lapse_harmonicf = T(1.0);           ///< Harmonic lapse prefactor.
    T lapse_harmonic = T(0.0);            ///< Harmonic lapse coefficient.
    T lapse_oplog = T(2.0);               ///< 1+log factor (when set to 2).
    T lapse_advect = T(1.0);              ///< Lapse advection factor.
    T shift_Gamma = T(1.0);               ///< Gamma-coupling in shift RHS.
    T shift_advect = T(1.0);              ///< Shift advection factor.
    T shift_alpha2Gamma = T(0.0);         ///< alpha^2 * Gamma coupling.
    T shift_H = T(0.0);                   ///< Harmonic shift coupling.
    T shift_eta = T(2.0);                 ///< Shift damping.
    bool slow_start_lapse = false;        ///< Enable early-time damping of lapse source.
    T ssl_damping_amp = T(0.6);           ///< Amplitude of slow-start lapse damping.
    T ssl_damping_time = T(20.0);         ///< Characteristic damping time.
    int ssl_damping_index = 1;            ///< Exponent index in the damping profile.
    T current_time = T(0.0);              ///< Runtime time provided by the RK driver.
    bool use_direct_shift_rhs = false;    ///< Prefer the reference Gamma-driver over the direct beta RHS.
    bool evolve_Z = true;                 ///< Compatibility knob: Z_i is reconstructed/diagnostic, not a driving state.
    bool frozen_Z_is_synced = false;      ///< Compatibility flag for callers that explicitly synchronize Z_i from Gamma.
    bool gamma_damping_uses_metric = false; ///< Dampen using (Gamma - Gamma(metric)).
    bool apply_rhs_sommerfeld = false;    ///< Apply Sommerfeld-like RHS corrections near boundaries.
    T boundary_gauge_characteristic_speed = T(1); ///< Gauge/metric radiative BC speed used in halo and collar updates.
    T boundary_z4c_characteristic_speed = T(1);   ///< Theta/Gamma/A/Z radiative BC speed.
    T boundary_khat_characteristic_speed = T(1.4142135623730951); ///< Khat radiative BC speed.

    inline T effective_eta() const noexcept {
        const T scale = std::max(mass_scale, std::numeric_limits<T>::epsilon());
        return eta / scale;
    }

    inline T effective_shift_eta() const noexcept {
        const T scale = std::max(mass_scale, std::numeric_limits<T>::epsilon());
        return shift_eta / scale;
    }

    inline T kappa1_times_lapse(T lapse) const noexcept {
        return covariant_z4 ? kappa1 : (kappa1 * lapse);
    }

    inline T slow_start_lapse_factor() const noexcept {
        if (!slow_start_lapse)
            return T(1);
        const T tau = std::max(ssl_damping_time, std::numeric_limits<T>::epsilon());
        const int p = std::max(ssl_damping_index, 1);
        const T x = std::max(current_time, T(0)) / tau;
        T x_pow = T(1);
        for (int n = 0; n < p; ++n)
            x_pow *= x;
        const T factor = T(1) - ssl_damping_amp * std::exp(-x_pow);
        return std::clamp(factor, T(0), T(1));
    }
};

/**
 * @brief Build \f$\partial_t\alpha\f$ including advection.
 */

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#    define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}
#endif

/**
 * @brief Build \f$\partial_t\alpha\f$ including advection and KO6 dissipation.
 */
template <typename T>
inline void compute_rhs_alpha(const BSSNGridSoA<T> &G, Field3D<T> &rhs_alpha, size_t padding = 4,
                              const GaugeParameters<T> &params = {}) {
    BSSN_PROFILE_KERNEL(Alpha);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma); // Adaptive KO6 strength.
    const bool use_theta = params.use_theta_in_lapse;
    const T ssl_factor = params.slow_start_lapse_factor();

    const auto region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
        return;

    // Constants
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

    auto loop_ij = [&](size_t i, size_t j) {

        size_t idx_start = G.alpha.idx(i, j, k0);

        const T *p_alpha = G.alpha.ptr() + idx_start;
        const T *p_K = G.K.ptr() + idx_start;
        const T *p_theta = use_theta ? (G.Theta.ptr() + idx_start) : nullptr;
        const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                              G.beta[2].ptr() + idx_start};
        T       *p_rhs = rhs_alpha.ptr() + idx_start;

        for (size_t k = k0; k < k1; ++k) {
            const T alpha = *p_alpha;
            const T K = *p_K;
            const T theta = use_theta ? *p_theta : T(0);
            const T bx = *p_beta[0];
            const T by = *p_beta[1];
            const T bz = *p_beta[2];

            const T advection = bx * Dx_upwind_ptr(p_alpha, sx, inv_2dx, bx) +
                                by * Dy_upwind_ptr(p_alpha, sy, inv_2dy, by) +
                                bz * Dz_upwind_ptr(p_alpha, inv_2dz, bz);

            const T diss = KO6_axis_ptr(p_alpha, sx) + KO6_axis_ptr(p_alpha, sy) +
                           KO6_axis_ptr(p_alpha, 1);
            const T diss_scaled = local_ko_scale(G, ko_sigma, i, j, k) * diss;

            const T lapse_K = use_theta ? Khat(K, theta) : K;
            const T f = params.lapse_oplog * params.lapse_harmonicf + params.lapse_harmonic * alpha;
            *p_rhs = params.lapse_advect * advection - ssl_factor * f * alpha * lapse_K + diss_scaled;

            ++p_alpha;
            ++p_K;
            ++p_rhs;
            if (use_theta)
                ++p_theta;
            ++p_beta[0];
            ++p_beta[1];
            ++p_beta[2];
        }
    };

    if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    } else {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    }
}

/**
 * @brief Assemble \f$\partial_t\beta^i\f$.
 * @details
 * - Gamma-driver mode (`use_direct_shift_rhs=false`): \f$\partial_t\beta^i = \beta^k\partial_k\beta^i +
 *   c_B B^i\f$.
 * - Direct-shift mode (`use_direct_shift_rhs=true`): optional direct shift RHS
 *   \f$\partial_t\beta^i = c_\Gamma\tilde{\Gamma}^i + c_{\text{adv}}\mathcal{L}_\beta\beta^i -
 *   \eta\beta^i + c_{\alpha^2\Gamma}\alpha^2\tilde{\Gamma}^i + c_H \alpha\chi \tilde{\gamma}^{ij}
 *   (0.5\alpha\partial_j\chi-\partial_j\alpha)\f$.
 */
template <typename T>
inline void compute_rhs_beta(const BSSNGridSoA<T> &G, Field3D<T> rhs[3],
                             const GaugeParameters<T> &params = {}, size_t padding = 4) {
    BSSN_PROFILE_KERNEL(Beta);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma); // Shared adaptive filter strength.

    const auto region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
        return;

    const ptrdiff_t sx = G.beta[0].st.sx;
    const ptrdiff_t sy = G.beta[0].st.sy;

    if (params.use_direct_shift_rhs) {
        const double inv_12dx = 1.0 / (60.0 * G.dx);
        const double inv_12dy = 1.0 / (60.0 * G.dy);
        const double inv_12dz = 1.0 / (60.0 * G.dz);
        const double inv_2dx = 1.0 / (2.0 * G.dx);
        const double inv_2dy = 1.0 / (2.0 * G.dy);
        const double inv_2dz = 1.0 / (2.0 * G.dz);
        const T      shift_eta = params.effective_shift_eta();

        auto loop_ij = [&](size_t i, size_t j) {
            size_t idx_start = G.beta[0].idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_tildeGamma[3] = {G.tildeGamma[0].ptr() + idx_start,
                                        G.tildeGamma[1].ptr() + idx_start,
                                        G.tildeGamma[2].ptr() + idx_start};
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_ginv[6];
            for (int s = 0; s < 6; ++s)
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
            T *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                           rhs[2].ptr() + idx_start};

            const T *p_beta0 = p_beta[0];
            const T *p_beta1 = p_beta[1];
            const T *p_beta2 = p_beta[2];
            const T *p_tg0 = p_tildeGamma[0];
            const T *p_tg1 = p_tildeGamma[1];
            const T *p_tg2 = p_tildeGamma[2];
            const T *p_g0 = p_ginv[0];
            const T *p_g1 = p_ginv[1];
            const T *p_g2 = p_ginv[2];
            const T *p_g3 = p_ginv[3];
            const T *p_g4 = p_ginv[4];
            const T *p_g5 = p_ginv[5];
            T       *p_rhs0 = p_rhs[0];
            T       *p_rhs1 = p_rhs[1];
            T       *p_rhs2 = p_rhs[2];

            #pragma omp simd
            for (size_t k = k0; k < k1; ++k) {
                const T bx = *p_beta0;
                const T by = *p_beta1;
                const T bz = *p_beta2;
                const T alpha = *p_alpha;
                const T chi = *p_chi;
                const T g_xx = *p_g0;
                const T g_xy = *p_g1;
                const T g_xz = *p_g2;
                const T g_yy = *p_g3;
                const T g_yz = *p_g4;
                const T g_zz = *p_g5;

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};
                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};
                const T gauge_combo_cov[3] = {T(0.5) * alpha * d_chi[0] - d_alpha[0],
                                              T(0.5) * alpha * d_chi[1] - d_alpha[1],
                                              T(0.5) * alpha * d_chi[2] - d_alpha[2]};

                // comp = 0
                const T *p_f0 = p_beta0;
                const T adv0 = bx * Dx_upwind_ptr(p_f0, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_f0, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_f0, inv_2dz, bz);
                const T diss0 = KO6_axis_ptr(p_f0, sx) + KO6_axis_ptr(p_f0, sy) + KO6_axis_ptr(p_f0, 1);
                const T gamma0 = *p_tg0;
                T gauge0 = g_xx * gauge_combo_cov[0] + g_xy * gauge_combo_cov[1] + g_xz * gauge_combo_cov[2];
                T rhs0 = params.shift_Gamma * gamma0 + params.shift_advect * adv0 - shift_eta * (*p_beta0);
                rhs0 += params.shift_alpha2Gamma * alpha * alpha * gamma0;
                rhs0 += params.shift_H * alpha * chi * gauge0;
                *p_rhs0 = rhs0 + local_ko_scale(G, ko_sigma, i, j, k) * diss0;

                // comp = 1
                const T *p_f1 = p_beta1;
                const T adv1 = bx * Dx_upwind_ptr(p_f1, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_f1, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_f1, inv_2dz, bz);
                const T diss1 = KO6_axis_ptr(p_f1, sx) + KO6_axis_ptr(p_f1, sy) + KO6_axis_ptr(p_f1, 1);
                const T gamma1 = *p_tg1;
                T gauge1 = g_xy * gauge_combo_cov[0] + g_yy * gauge_combo_cov[1] + g_yz * gauge_combo_cov[2];
                T rhs1 = params.shift_Gamma * gamma1 + params.shift_advect * adv1 - shift_eta * (*p_beta1);
                rhs1 += params.shift_alpha2Gamma * alpha * alpha * gamma1;
                rhs1 += params.shift_H * alpha * chi * gauge1;
                *p_rhs1 = rhs1 + local_ko_scale(G, ko_sigma, i, j, k) * diss1;

                // comp = 2
                const T *p_f2 = p_beta2;
                const T adv2 = bx * Dx_upwind_ptr(p_f2, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_f2, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_f2, inv_2dz, bz);
                const T diss2 = KO6_axis_ptr(p_f2, sx) + KO6_axis_ptr(p_f2, sy) + KO6_axis_ptr(p_f2, 1);
                const T gamma2 = *p_tg2;
                T gauge2 = g_xz * gauge_combo_cov[0] + g_yz * gauge_combo_cov[1] + g_zz * gauge_combo_cov[2];
                T rhs2 = params.shift_Gamma * gamma2 + params.shift_advect * adv2 - shift_eta * (*p_beta2);
                rhs2 += params.shift_alpha2Gamma * alpha * alpha * gamma2;
                rhs2 += params.shift_H * alpha * chi * gauge2;
                *p_rhs2 = rhs2 + local_ko_scale(G, ko_sigma, i, j, k) * diss2;

                ++p_beta0;
                ++p_beta1;
                ++p_beta2;
                ++p_tg0;
                ++p_tg1;
                ++p_tg2;
                ++p_rhs0;
                ++p_rhs1;
                ++p_rhs2;
                ++p_alpha;
                ++p_chi;
                ++p_g0;
                ++p_g1;
                ++p_g2;
                ++p_g3;
                ++p_g4;
                ++p_g5;
            }
        };

        if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_ij(i, j);
        } else {
#pragma omp parallel for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_ij(i, j);
        }
        return;
    }

    const bool   use_advect = (params.shift_advect != T(0));
    const double inv_2dx = use_advect ? (1.0 / (2.0 * G.dx)) : 0.0;
    const double inv_2dy = use_advect ? (1.0 / (2.0 * G.dy)) : 0.0;
    const double inv_2dz = use_advect ? (1.0 / (2.0 * G.dz)) : 0.0;

    auto loop_ij = [&](size_t i, size_t j) {
        size_t idx_start = G.beta[0].idx(i, j, k0);

        const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                              G.beta[2].ptr() + idx_start};
        const T *p_B[3] = {G.B[0].ptr() + idx_start, G.B[1].ptr() + idx_start,
                           G.B[2].ptr() + idx_start};
        T *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                       rhs[2].ptr() + idx_start};

        T *p_rhs0 = p_rhs[0];
        T *p_rhs1 = p_rhs[1];
        T *p_rhs2 = p_rhs[2];
        const T *p_beta0 = p_beta[0];
        const T *p_beta1 = p_beta[1];
        const T *p_beta2 = p_beta[2];
        const T *p_B0 = p_B[0];
        const T *p_B1 = p_B[1];
        const T *p_B2 = p_B[2];

        #pragma omp simd
        for (size_t k = k0; k < k1; ++k) {
            if (use_advect) {
                const T bx = *p_beta0;
                const T by = *p_beta1;
                const T bz = *p_beta2;

                const T adv0 = bx * Dx_upwind_ptr(p_beta0, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_beta0, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_beta0, inv_2dz, bz);
                const T adv1 = bx * Dx_upwind_ptr(p_beta1, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_beta1, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_beta1, inv_2dz, bz);
                const T adv2 = bx * Dx_upwind_ptr(p_beta2, sx, inv_2dx, bx) +
                               by * Dy_upwind_ptr(p_beta2, sy, inv_2dy, by) +
                               bz * Dz_upwind_ptr(p_beta2, inv_2dz, bz);

                const T diss0 =
                    KO6_axis_ptr(p_beta0, sx) + KO6_axis_ptr(p_beta0, sy) + KO6_axis_ptr(p_beta0, 1);
                const T diss1 =
                    KO6_axis_ptr(p_beta1, sx) + KO6_axis_ptr(p_beta1, sy) + KO6_axis_ptr(p_beta1, 1);
                const T diss2 =
                    KO6_axis_ptr(p_beta2, sx) + KO6_axis_ptr(p_beta2, sy) + KO6_axis_ptr(p_beta2, 1);

                const T ko_scale = local_ko_scale(G, ko_sigma, i, j, k);
                const T diss0_scaled = ko_scale * diss0;
                const T diss1_scaled = ko_scale * diss1;
                const T diss2_scaled = ko_scale * diss2;

                *p_rhs0 = params.beta_B_coeff * (*p_B0) + params.shift_advect * adv0 + diss0_scaled;
                *p_rhs1 = params.beta_B_coeff * (*p_B1) + params.shift_advect * adv1 + diss1_scaled;
                *p_rhs2 = params.beta_B_coeff * (*p_B2) + params.shift_advect * adv2 + diss2_scaled;
            } else {
                const T diss0 =
                    KO6_axis_ptr(p_beta0, sx) + KO6_axis_ptr(p_beta0, sy) + KO6_axis_ptr(p_beta0, 1);
                const T diss1 =
                    KO6_axis_ptr(p_beta1, sx) + KO6_axis_ptr(p_beta1, sy) + KO6_axis_ptr(p_beta1, 1);
                const T diss2 =
                    KO6_axis_ptr(p_beta2, sx) + KO6_axis_ptr(p_beta2, sy) + KO6_axis_ptr(p_beta2, 1);
                const T ko_scale = local_ko_scale(G, ko_sigma, i, j, k);
                const T diss0_scaled = ko_scale * diss0;
                const T diss1_scaled = ko_scale * diss1;
                const T diss2_scaled = ko_scale * diss2;

                *p_rhs0 = params.beta_B_coeff * (*p_B0) + diss0_scaled;
                *p_rhs1 = params.beta_B_coeff * (*p_B1) + diss1_scaled;
                *p_rhs2 = params.beta_B_coeff * (*p_B2) + diss2_scaled;
            }

            ++p_beta0;
            ++p_beta1;
            ++p_beta2;
            ++p_B0;
            ++p_B1;
            ++p_B2;
            ++p_rhs0;
            ++p_rhs1;
            ++p_rhs2;
        }
    };

    if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    } else {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    }
}

/**
 * @brief Assemble \f$\partial_t B^i\f$ given the previously computed
 * \f$\partial_t\tilde{\Gamma}^i\f$.
 * @details Two Gamma-driver modes are supported:
 * - If `use_direct_shift_rhs=true`, \f$B^i\f$ is disabled and this routine writes zero RHS.
 * - `use_shift_advection = true` (GRChombo Eq. 21–22): rhs_Gamma is \f$\partial_t\tilde{\Gamma}^i\f\f$, so
 *   this kernel forms \f$(\partial_t-\beta^k\partial_k)\tilde{\Gamma}^i\f\f$ explicitly and includes
 *   \f$\beta^k\partial_k B^i\f\f$.
 * - `use_shift_advection = false`: rhs_Gamma is already \f$(\partial_t-\beta^k\partial_k)\tilde{\Gamma}^i\f\f$,
 *   so the advection terms are omitted.
 */
template <typename T>
inline void compute_rhs_B(const BSSNGridSoA<T> &G, const Field3D<T> rhs_Gamma[3],
                          Field3D<T> rhs_B[3], const GaugeParameters<T> &params = {},
                          size_t padding = 4) {
    BSSN_PROFILE_KERNEL(B);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma); // Adaptive KO6 dial for B^i.
    const auto region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
        return;

    if (params.use_direct_shift_rhs) {
        auto loop_direct_ij = [&](size_t i, size_t j) {
                size_t idx_start = rhs_B[0].idx(i, j, k0);
                T     *p_rhs_B[3] = {rhs_B[0].ptr() + idx_start, rhs_B[1].ptr() + idx_start,
                                     rhs_B[2].ptr() + idx_start};
                for (size_t k = k0; k < k1; ++k) {
                    for (int comp = 0; comp < 3; ++comp)
                        *p_rhs_B[comp] = T(0);
                    for (int comp = 0; comp < 3; ++comp)
                        ++p_rhs_B[comp];
                }
        };
        if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_direct_ij(i, j);
        } else {
#pragma omp parallel for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_direct_ij(i, j);
        }
        return;
    }

    const ptrdiff_t sx = G.B[0].st.sx;
    const ptrdiff_t sy = G.B[0].st.sy;
    const T         eta_coeff = params.effective_eta();
    const bool      use_shift_advection =
        params.use_shift_advection && (params.shift_advect != T(0));

    if (use_shift_advection) {
        const double inv_2dx = 1.0 / (2.0 * G.dx);
        const double inv_2dy = 1.0 / (2.0 * G.dy);
        const double inv_2dz = 1.0 / (2.0 * G.dz);
        const T      advect_coeff = params.shift_advect;

        auto loop_ij = [&](size_t i, size_t j) {
            size_t idx_start = G.B[0].idx(i, j, k0);

            const T *p_beta0 = G.beta[0].ptr() + idx_start;
            const T *p_beta1 = G.beta[1].ptr() + idx_start;
            const T *p_beta2 = G.beta[2].ptr() + idx_start;
            const T *p_B0 = G.B[0].ptr() + idx_start;
            const T *p_B1 = G.B[1].ptr() + idx_start;
            const T *p_B2 = G.B[2].ptr() + idx_start;
            const T *p_tg0 = G.tildeGamma[0].ptr() + idx_start;
            const T *p_tg1 = G.tildeGamma[1].ptr() + idx_start;
            const T *p_tg2 = G.tildeGamma[2].ptr() + idx_start;
            const T *p_rhs_G0 = rhs_Gamma[0].ptr() + idx_start;
            const T *p_rhs_G1 = rhs_Gamma[1].ptr() + idx_start;
            const T *p_rhs_G2 = rhs_Gamma[2].ptr() + idx_start;
            T       *p_rhs_B0 = rhs_B[0].ptr() + idx_start;
            T       *p_rhs_B1 = rhs_B[1].ptr() + idx_start;
            T       *p_rhs_B2 = rhs_B[2].ptr() + idx_start;

#pragma omp simd
            for (size_t k = k0; k < k1; ++k) {
                const T bx = *p_beta0;
                const T by = *p_beta1;
                const T bz = *p_beta2;

                const T adv_B0 = advect_coeff *
                                 (bx * Dx_upwind_ptr(p_B0, sx, inv_2dx, bx) +
                                  by * Dy_upwind_ptr(p_B0, sy, inv_2dy, by) +
                                  bz * Dz_upwind_ptr(p_B0, inv_2dz, bz));
                const T adv_B1 = advect_coeff *
                                 (bx * Dx_upwind_ptr(p_B1, sx, inv_2dx, bx) +
                                  by * Dy_upwind_ptr(p_B1, sy, inv_2dy, by) +
                                  bz * Dz_upwind_ptr(p_B1, inv_2dz, bz));
                const T adv_B2 = advect_coeff *
                                 (bx * Dx_upwind_ptr(p_B2, sx, inv_2dx, bx) +
                                  by * Dy_upwind_ptr(p_B2, sy, inv_2dy, by) +
                                  bz * Dz_upwind_ptr(p_B2, inv_2dz, bz));

                const T adv_Gamma0 = advect_coeff *
                                     (bx * Dx_upwind_ptr(p_tg0, sx, inv_2dx, bx) +
                                      by * Dy_upwind_ptr(p_tg0, sy, inv_2dy, by) +
                                      bz * Dz_upwind_ptr(p_tg0, inv_2dz, bz));
                const T adv_Gamma1 = advect_coeff *
                                     (bx * Dx_upwind_ptr(p_tg1, sx, inv_2dx, bx) +
                                      by * Dy_upwind_ptr(p_tg1, sy, inv_2dy, by) +
                                      bz * Dz_upwind_ptr(p_tg1, inv_2dz, bz));
                const T adv_Gamma2 = advect_coeff *
                                     (bx * Dx_upwind_ptr(p_tg2, sx, inv_2dx, bx) +
                                      by * Dy_upwind_ptr(p_tg2, sy, inv_2dy, by) +
                                      bz * Dz_upwind_ptr(p_tg2, inv_2dz, bz));

                const T diss0 = KO6_axis_ptr(p_B0, sx) + KO6_axis_ptr(p_B0, sy) + KO6_axis_ptr(p_B0, 1);
                const T diss1 = KO6_axis_ptr(p_B1, sx) + KO6_axis_ptr(p_B1, sy) + KO6_axis_ptr(p_B1, 1);
                const T diss2 = KO6_axis_ptr(p_B2, sx) + KO6_axis_ptr(p_B2, sy) + KO6_axis_ptr(p_B2, 1);
                const T ko_scale = local_ko_scale(G, ko_sigma, i, j, k);

                *p_rhs_B0 = (*p_rhs_G0 - adv_Gamma0) + adv_B0 - eta_coeff * (*p_B0) + ko_scale * diss0;
                *p_rhs_B1 = (*p_rhs_G1 - adv_Gamma1) + adv_B1 - eta_coeff * (*p_B1) + ko_scale * diss1;
                *p_rhs_B2 = (*p_rhs_G2 - adv_Gamma2) + adv_B2 - eta_coeff * (*p_B2) + ko_scale * diss2;

                ++p_beta0;
                ++p_beta1;
                ++p_beta2;
                ++p_B0;
                ++p_B1;
                ++p_B2;
                ++p_tg0;
                ++p_tg1;
                ++p_tg2;
                ++p_rhs_G0;
                ++p_rhs_G1;
                ++p_rhs_G2;
                ++p_rhs_B0;
                ++p_rhs_B1;
                ++p_rhs_B2;
            }
        };

        if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_ij(i, j);
        } else {
#pragma omp parallel for collapse(2)
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    loop_ij(i, j);
        }
        return;
    }

    auto loop_ij = [&](size_t i, size_t j) {
        size_t idx_start = G.B[0].idx(i, j, k0);

        const T *p_B0 = G.B[0].ptr() + idx_start;
        const T *p_B1 = G.B[1].ptr() + idx_start;
        const T *p_B2 = G.B[2].ptr() + idx_start;
        const T *p_rhs_G0 = rhs_Gamma[0].ptr() + idx_start;
        const T *p_rhs_G1 = rhs_Gamma[1].ptr() + idx_start;
        const T *p_rhs_G2 = rhs_Gamma[2].ptr() + idx_start;
        T       *p_rhs_B0 = rhs_B[0].ptr() + idx_start;
        T       *p_rhs_B1 = rhs_B[1].ptr() + idx_start;
        T       *p_rhs_B2 = rhs_B[2].ptr() + idx_start;

#pragma omp simd
        for (size_t k = k0; k < k1; ++k) {
            const T diss0 = KO6_axis_ptr(p_B0, sx) + KO6_axis_ptr(p_B0, sy) + KO6_axis_ptr(p_B0, 1);
            const T diss1 = KO6_axis_ptr(p_B1, sx) + KO6_axis_ptr(p_B1, sy) + KO6_axis_ptr(p_B1, 1);
            const T diss2 = KO6_axis_ptr(p_B2, sx) + KO6_axis_ptr(p_B2, sy) + KO6_axis_ptr(p_B2, 1);
            const T ko_scale = local_ko_scale(G, ko_sigma, i, j, k);

            *p_rhs_B0 = *p_rhs_G0 - eta_coeff * (*p_B0) + ko_scale * diss0;
            *p_rhs_B1 = *p_rhs_G1 - eta_coeff * (*p_B1) + ko_scale * diss1;
            *p_rhs_B2 = *p_rhs_G2 - eta_coeff * (*p_B2) + ko_scale * diss2;

            ++p_B0;
            ++p_B1;
            ++p_B2;
            ++p_rhs_G0;
            ++p_rhs_G1;
            ++p_rhs_G2;
            ++p_rhs_B0;
            ++p_rhs_B1;
            ++p_rhs_B2;
        }
    };

    if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    } else {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_ij(i, j);
    }
}
} // namespace tensorium_RG::bssn
