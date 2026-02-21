#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../TimeIntegration/BSSNPerfTimers.hpp"
#include "BSSNEvolutionCommon.hpp"
#include "BSSNEvolutionGauge.hpp"
/**
 * @file BSSNEvolutionChi.hpp
 * @brief RHS for the conformal factor equation \f$(\partial_t-\mathcal{L}_\beta)\chi =
 * \frac{2}{3}\chi(\alpha K-\partial_i\beta^i)\f$.
 * @details The implementation evaluates
 * \f[
 * \partial_t\chi = \beta^i\partial_i\chi + \tfrac{2}{3}\chi(\alpha K-\partial_i\beta^i) +
 * \mathcal{D}_6[\chi]
 * \f]
 * where advection uses third-order upwind stencils tied to the sign of each \f$\beta^i\f$ component
 * and
 * \f$\mathcal{D}_6\f$ is the KO6 dissipation.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Fill the RHS buffer for \f$\chi\f$ inside the padded interior region.
 * @param padding Number of guard cells skipped on each side before looping.
 * @param params Gauge knobs (KO6 strength, optional Θ-in-lapse) shared with the other RHS kernels.
 *
 * **Mapping to code.**
 * - `d_chi` corresponds to the advective term \f$\beta^i\partial_i\chi\f$.
 * - `div_beta` uses centered derivatives to compute \f$\partial_i\beta^i\f$.
 * - The scalar multiplier `T(2/3)*chi*(alpha*K - div_beta)` encodes the source term.
 * - KO6 call adds dissipation using `params.ko_sigma` so all fields share the same filter.
 */
template <typename T>
inline void compute_rhs_chi(const BSSNGridSoA<T> &G, Field3D<T> &rhs_chi, size_t padding = 4,
                            const GaugeParameters<T> &params = {}) {
    BSSN_PROFILE_KERNEL(Chi);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma);

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

    const double    inv_12dx = 1.0 / (60.0 * G.dx);
    const double    inv_12dy = 1.0 / (60.0 * G.dy);
    const double    inv_12dz = 1.0 / (60.0 * G.dz);
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const ptrdiff_t sx = G.chi.st.sx;
    const ptrdiff_t sy = G.chi.st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.chi.idx(i, j, k0);

            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;
            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            T       *p_rhs = rhs_chi.ptr() + idx_start;

            for (size_t k = k0; k < k1; ++k) {
                const T chi = *p_chi;
                const T alpha = *p_alpha;
                const T K = *p_K;
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];

                const T d_chi = bx * Dx_upwind_ptr(p_chi, sx, inv_2dx, bx) +
                                by * Dy_upwind_ptr(p_chi, sy, inv_2dy, by) +
                                bz * Dz_upwind_ptr(p_chi, inv_2dz, bz);

                const T div_beta = Dx_ptr(p_beta[0], sx, inv_12dx) +
                                   Dy_ptr(p_beta[1], sy, inv_12dy) + Dz_ptr(p_beta[2], inv_12dz);

                const T diss =
                    KO6_axis_ptr(p_chi, sx) + KO6_axis_ptr(p_chi, sy) + KO6_axis_ptr(p_chi, 1);

                *p_rhs = d_chi + (T(2.0 / 3.0)) * chi * (alpha * K - div_beta) +
                         (ko_sigma / G.dx) * diss;

                // Increment pointers
                ++p_chi;
                ++p_alpha;
                ++p_K;
                ++p_rhs;
                ++p_beta[0];
                ++p_beta[1];
                ++p_beta[2];
            }
        }
    }
}

} // namespace tensorium_RG::bssn
