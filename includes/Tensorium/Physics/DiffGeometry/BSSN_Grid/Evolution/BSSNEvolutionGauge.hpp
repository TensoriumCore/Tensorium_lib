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
    T eta = T(6);
    T mass_scale = T(1);
    T kappa1 = T(0.1);
    T kappa2 = T(0.0);
    T kappa_z = T(1.0);
    T ko_sigma = T(0.35);             ///< KO6 filter strength (scaled by dx).
    T min_lapse_for_K = T(5e-3);      ///< Floor applied when sourcing the K quadratic term.
    T max_K_squared = T(1e4);         ///< Safety cap for K^2 when alpha collapses.
    bool use_theta_in_lapse = false;      ///< Optional Z4c term: -2 alpha (K - 2 Theta).
    bool use_shift_advection = true;      ///< If false, drop β·∂ terms in the Gamma-driver.

    inline T effective_eta() const noexcept {
        const T scale = std::max(mass_scale, std::numeric_limits<T>::epsilon());
        return eta / scale;
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

                const T khat_val = use_theta ? Khat(K, theta) : K;
                *p_rhs = advection - T(2) * alpha * khat_val + diss_scaled;

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
 * @brief Assemble \f$\partial_t\beta^i = \beta^k\partial_k\beta^i + (3/4)B^i\f$.
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
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.beta[0].st.sx;
    const ptrdiff_t sy = G.beta[0].st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.beta[0].idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_B[3] = {G.B[0].ptr() + idx_start, G.B[1].ptr() + idx_start,
                               G.B[2].ptr() + idx_start};
            T       *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                                 rhs[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];

                for (int comp = 0; comp < 3; ++comp) {
                    const T *p_f = p_beta[comp];
                    const T  adv = bx * Dx_upwind_ptr(p_f, sx, inv_2dx, bx) +
                                   by * Dy_upwind_ptr(p_f, sy, inv_2dy, by) +
                                   bz * Dz_upwind_ptr(p_f, inv_2dz, bz);

                    const T diss =
                        KO6_axis_ptr(p_f, sx) + KO6_axis_ptr(p_f, sy) + KO6_axis_ptr(p_f, 1);
                    const T diss_scaled = (ko_sigma / G.dx) * diss;

                    *p_rhs[comp] = adv + params.beta_B_coeff * (*p_B[comp]) + diss_scaled;
                }

                // Increments
                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_B[c];
                    ++p_rhs[c];
                }
            }
        }
    }
}

/**
 * @brief Assemble \f$\partial_t B^i\f$ given the previously computed
 * \f$\partial_t\tilde{\Gamma}^i\f$.
 * @details Two Gamma-driver modes are supported:
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
                                        ? (bx * Dx_upwind_ptr(p_B[comp], sx, inv_2dx, bx) +
                                           by * Dy_upwind_ptr(p_B[comp], sy, inv_2dy, by) +
                                           bz * Dz_upwind_ptr(p_B[comp], inv_2dz, bz))
                                        : T(0);
                    const T adv_Gamma = use_shift_advection
                                            ? (bx * Dx_upwind_ptr(p_tildeGamma[comp], sx, inv_2dx, bx) +
                                               by * Dy_upwind_ptr(p_tildeGamma[comp], sy, inv_2dy, by) +
                                               bz * Dz_upwind_ptr(p_tildeGamma[comp], inv_2dz, bz))
                                            : T(0);

                    const T diss = KO6_axis_ptr(p_B[comp], sx) + KO6_axis_ptr(p_B[comp], sy) +
                                   KO6_axis_ptr(p_B[comp], 1);
                    const T diss_scaled = (ko_sigma / G.dx) * diss;
                    const T rhs_gamma_eff = *p_rhs_G[comp] - adv_Gamma;
                    const T z_feedback = -params.kappa_z * (*p_Z[comp]);

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
