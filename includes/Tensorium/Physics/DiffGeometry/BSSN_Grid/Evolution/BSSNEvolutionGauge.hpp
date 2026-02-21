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
    bool use_direct_shift_rhs = true;     ///< Use direct beta RHS instead of B-driver.
    bool evolve_Z = true;                 ///< Evolve Z_i as an independent field.
    bool gamma_damping_uses_metric = false; ///< Dampen using (Gamma - Gamma(metric)).
    bool apply_rhs_sommerfeld = false;    ///< Apply Sommerfeld-like RHS corrections near boundaries.

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

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    // Constants
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

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
                const T diss_scaled = (ko_sigma / G.dx) * diss;

                const T lapse_K = use_theta ? Khat(K, theta) : K;
                const T f = params.lapse_oplog * params.lapse_harmonicf +
                            params.lapse_harmonic * alpha;
                *p_rhs =
                    params.lapse_advect * advection - ssl_factor * f * alpha * lapse_K + diss_scaled;

                // Increments
                ++p_alpha;
                ++p_K;
                ++p_rhs;
                if (use_theta)
                    ++p_theta;
                ++p_beta[0];
                ++p_beta[1];
                ++p_beta[2];
            }
        }
    }
}

/**
 * @brief Assemble \f$\partial_t\beta^i\f$.
 * @details
 * - Legacy mode (`use_direct_shift_rhs=false`): \f$\partial_t\beta^i = \beta^k\partial_k\beta^i +
 *   c_B B^i\f$.
 * - Direct-shift mode (`use_direct_shift_rhs=true`): direct shift RHS
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

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    // Constants
    const double    inv_12dx = 1.0 / (60.0 * G.dx);
    const double    inv_12dy = 1.0 / (60.0 * G.dy);
    const double    inv_12dz = 1.0 / (60.0 * G.dz);
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.beta[0].st.sx;
    const ptrdiff_t sy = G.beta[0].st.sy;
    const T         shift_eta = params.effective_shift_eta();

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.beta[0].idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_B[3] = {G.B[0].ptr() + idx_start, G.B[1].ptr() + idx_start,
                               G.B[2].ptr() + idx_start};
            const T *p_tildeGamma[3] = {G.tildeGamma[0].ptr() + idx_start,
                                        G.tildeGamma[1].ptr() + idx_start,
                                        G.tildeGamma[2].ptr() + idx_start};
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_ginv[6];
            for (int s = 0; s < 6; ++s)
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
            T       *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                                 rhs[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];
                const T alpha = *p_alpha;
                const T chi = *p_chi;
                const T g_xx = *p_ginv[0];
                const T g_xy = *p_ginv[1];
                const T g_xz = *p_ginv[2];
                const T g_yy = *p_ginv[3];
                const T g_yz = *p_ginv[4];
                const T g_zz = *p_ginv[5];

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};
                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};
                const T gauge_combo_cov[3] = {T(0.5) * alpha * d_chi[0] - d_alpha[0],
                                              T(0.5) * alpha * d_chi[1] - d_alpha[1],
                                              T(0.5) * alpha * d_chi[2] - d_alpha[2]};

                for (int comp = 0; comp < 3; ++comp) {
                    const T *p_f = p_beta[comp];
                    const T  adv = bx * Dx_upwind_ptr(p_f, sx, inv_2dx, bx) +
                                   by * Dy_upwind_ptr(p_f, sy, inv_2dy, by) +
                                   bz * Dz_upwind_ptr(p_f, inv_2dz, bz);

                    const T diss =
                        KO6_axis_ptr(p_f, sx) + KO6_axis_ptr(p_f, sy) + KO6_axis_ptr(p_f, 1);
                    const T diss_scaled = (ko_sigma / G.dx) * diss;

                    if (params.use_direct_shift_rhs) {
                        const T gamma_comp = *p_tildeGamma[comp];
                        T       gauge_combo_up = T(0);
                        if (comp == 0) {
                            gauge_combo_up = g_xx * gauge_combo_cov[0] + g_xy * gauge_combo_cov[1] +
                                             g_xz * gauge_combo_cov[2];
                        } else if (comp == 1) {
                            gauge_combo_up = g_xy * gauge_combo_cov[0] + g_yy * gauge_combo_cov[1] +
                                             g_yz * gauge_combo_cov[2];
                        } else {
                            gauge_combo_up = g_xz * gauge_combo_cov[0] + g_yz * gauge_combo_cov[1] +
                                             g_zz * gauge_combo_cov[2];
                        }

                        T rhs_beta = params.shift_Gamma * gamma_comp + params.shift_advect * adv -
                                     shift_eta * (*p_beta[comp]);
                        rhs_beta += params.shift_alpha2Gamma * alpha * alpha * gamma_comp;
                        rhs_beta += params.shift_H * alpha * chi * gauge_combo_up;
                        *p_rhs[comp] = rhs_beta + diss_scaled;
                    } else {
                        // Legacy Gamma-driver mode: allow disabling advection to match
                        // the standard moving-puncture choice.
                        *p_rhs[comp] =
                            params.shift_advect * adv + params.beta_B_coeff * (*p_B[comp]) +
                            diss_scaled;
                    }
                }

                // Increments
                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_B[c];
                    ++p_tildeGamma[c];
                    ++p_rhs[c];
                }
                ++p_alpha;
                ++p_chi;
                for (int s = 0; s < 6; ++s)
                    ++p_ginv[s];
            }
        }
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
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    if (params.use_direct_shift_rhs) {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i) {
            for (size_t j = j0; j < j1; ++j) {
                size_t idx_start = rhs_B[0].idx(i, j, k0);
                T     *p_rhs_B[3] = {rhs_B[0].ptr() + idx_start, rhs_B[1].ptr() + idx_start,
                                     rhs_B[2].ptr() + idx_start};
                for (size_t k = k0; k < k1; ++k) {
                    for (int comp = 0; comp < 3; ++comp)
                        *p_rhs_B[comp] = T(0);
                    for (int comp = 0; comp < 3; ++comp)
                        ++p_rhs_B[comp];
                }
            }
        }
        return;
    }

    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.B[0].st.sx;
    const ptrdiff_t sy = G.B[0].st.sy;
    const T         eta_coeff = params.effective_eta();
    const bool      use_shift_advection = params.use_shift_advection;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.B[0].idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_B[3] = {G.B[0].ptr() + idx_start, G.B[1].ptr() + idx_start,
                               G.B[2].ptr() + idx_start};
            const T *p_tildeGamma[3] = {G.tildeGamma[0].ptr() + idx_start,
                                        G.tildeGamma[1].ptr() + idx_start,
                                        G.tildeGamma[2].ptr() + idx_start};
            const T *p_Z[3] = {G.Z[0].ptr() + idx_start, G.Z[1].ptr() + idx_start,
                               G.Z[2].ptr() + idx_start};
            const T *p_rhs_G[3] = {rhs_Gamma[0].ptr() + idx_start, rhs_Gamma[1].ptr() + idx_start,
                                   rhs_Gamma[2].ptr() + idx_start};
            T       *p_rhs_B[3] = {rhs_B[0].ptr() + idx_start, rhs_B[1].ptr() + idx_start,
                                   rhs_B[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];

                for (int comp = 0; comp < 3; ++comp) {
                    const T adv_B = use_shift_advection
                                        ? (params.shift_advect *
                                           (bx * Dx_upwind_ptr(p_B[comp], sx, inv_2dx, bx) +
                                            by * Dy_upwind_ptr(p_B[comp], sy, inv_2dy, by) +
                                            bz * Dz_upwind_ptr(p_B[comp], inv_2dz, bz)))
                                        : T(0);
                    const T adv_Gamma = use_shift_advection
                                            ? (params.shift_advect *
                                               (bx * Dx_upwind_ptr(p_tildeGamma[comp], sx, inv_2dx,
                                                                   bx) +
                                                by * Dy_upwind_ptr(p_tildeGamma[comp], sy, inv_2dy,
                                                                   by) +
                                                bz * Dz_upwind_ptr(p_tildeGamma[comp], inv_2dz,
                                                                   bz)))
                                            : T(0);

                    const T diss = KO6_axis_ptr(p_B[comp], sx) + KO6_axis_ptr(p_B[comp], sy) +
                                   KO6_axis_ptr(p_B[comp], 1);
                    const T diss_scaled = (ko_sigma / G.dx) * diss;
                    const T rhs_gamma_eff = *p_rhs_G[comp] - adv_Gamma;
                    const T z_feedback =
                        params.evolve_Z ? (-params.kappa_z * (*p_Z[comp])) : T(0);

                    *p_rhs_B[comp] = rhs_gamma_eff + adv_B - eta_coeff * (*p_B[comp]) +
                                     z_feedback + diss_scaled;
                }

                // Increments
                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_B[c];
                    ++p_tildeGamma[c];
                    ++p_Z[c];
                    ++p_rhs_G[c];
                    ++p_rhs_B[c];
                }
            }
        }
    }
}
} // namespace tensorium_RG::bssn
