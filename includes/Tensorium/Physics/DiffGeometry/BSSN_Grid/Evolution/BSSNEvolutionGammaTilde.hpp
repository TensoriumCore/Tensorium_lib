#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../TimeIntegration/BSSNPerfTimers.hpp"
#include "BSSNEvolutionCommon.hpp"
#include "BSSNEvolutionGauge.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionGammaTilde.hpp
 * @brief RHS for the conformal metric \f$\tilde{\gamma}_{ij}\f$.
 * @details Evaluates
 * \f[
 * \partial_t\tilde{\gamma}_{ij} = -2\alpha\tilde{A}_{ij} + \mathcal{L}_\beta \tilde{\gamma}_{ij} -
 * \tfrac{2}{3}\tilde{\gamma}_{ij}\partial_k\beta^k.
 * \f]
 * Lie derivatives expand into \f$\tilde{\gamma}_{ik}\partial_j\beta^k +
 * \tilde{\gamma}_{jk}\partial_i\beta^k\f$ plus the advection term
 * \f$\beta^k\partial_k\tilde{\gamma}_{ij}\f$.
 */

namespace tensorium_RG::bssn {

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#    define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}
#endif

template <typename T>
inline void compute_rhs_gamma_tilde(const BSSNGridSoA<T> &G, Field3D<T> rhs[6], size_t padding = 4,
                                    const GaugeParameters<T> &params = {}) {
    BSSN_PROFILE_KERNEL(GammaTilde);
    using namespace tensorium_RG::fd;
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

    const T         ko_sigma = scaled_ko_sigma(params.ko_sigma);
    const double    inv_12dx = 1.0 / (60.0 * G.dx);
    const double    inv_12dy = 1.0 / (60.0 * G.dy);
    const double    inv_12dz = 1.0 / (60.0 * G.dz);
    const double    inv_2dx = 1.0 / (2.0 * G.dx);
    const double    inv_2dy = 1.0 / (2.0 * G.dy);
    const double    inv_2dz = 1.0 / (2.0 * G.dz);
    const T         two_thirds = T(2) / T(3);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.gamma_tilde[0].idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_gam[6];
            const T *p_A[6];
            T       *p_rhs[6];

            for (int s = 0; s < 6; ++s) {
                p_gam[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
                p_rhs[s] = rhs[s].ptr() + idx_start;
            }
            const T *p_alpha = G.alpha.ptr() + idx_start;

            for (size_t k = k0; k < k1; ++k) {

                const T alpha = *p_alpha;

                T d_beta[3][3];
                for (int c = 0; c < 3; ++c) {
                    d_beta[c][0] = Dx_ptr(p_beta[c], sx, inv_12dx);
                    d_beta[c][1] = Dy_ptr(p_beta[c], sy, inv_12dy);
                    d_beta[c][2] = Dz_ptr(p_beta[c], inv_12dz);
                }
                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                const T beta_vec[3] = {*p_beta[0], *p_beta[1], *p_beta[2]};

                const T g_xx = *p_gam[tensorium_RG::XX];
                const T g_xy = *p_gam[tensorium_RG::XY];
                const T g_xz = *p_gam[tensorium_RG::XZ];
                const T g_yy = *p_gam[tensorium_RG::YY];
                const T g_yz = *p_gam[tensorium_RG::YZ];
                const T g_zz = *p_gam[tensorium_RG::ZZ];

                const T lie_vals[6] = {
                    T(2) * (g_xx * d_beta[0][0] + g_xy * d_beta[1][0] + g_xz * d_beta[2][0]),
                    (g_xx * d_beta[0][1] + g_xy * d_beta[1][1] + g_xz * d_beta[2][1]) +
                        (g_xy * d_beta[0][0] + g_yy * d_beta[1][0] + g_yz * d_beta[2][0]),
                    (g_xx * d_beta[0][2] + g_xy * d_beta[1][2] + g_xz * d_beta[2][2]) +
                        (g_xz * d_beta[0][0] + g_yz * d_beta[1][0] + g_zz * d_beta[2][0]),
                    T(2) * (g_xy * d_beta[0][1] + g_yy * d_beta[1][1] + g_yz * d_beta[2][1]),
                    (g_xy * d_beta[0][2] + g_yy * d_beta[1][2] + g_yz * d_beta[2][2]) +
                        (g_xz * d_beta[0][1] + g_yz * d_beta[1][1] + g_zz * d_beta[2][1]),
                    T(2) * (g_xz * d_beta[0][2] + g_yz * d_beta[1][2] + g_zz * d_beta[2][2])};

                const T trace_vals[6] = {
                    -two_thirds * g_xx * div_beta, -two_thirds * g_xy * div_beta,
                    -two_thirds * g_xz * div_beta, -two_thirds * g_yy * div_beta,
                    -two_thirds * g_yz * div_beta, -two_thirds * g_zz * div_beta};

                for (int s = 0; s < 6; ++s) {
                    const T *p_g = p_gam[s];
                    const T  adv = beta_vec[0] * Dx_upwind_ptr(p_g, sx, inv_2dx, beta_vec[0]) +
                                  beta_vec[1] * Dy_upwind_ptr(p_g, sy, inv_2dy, beta_vec[1]) +
                                  beta_vec[2] * Dz_upwind_ptr(p_g, inv_2dz, beta_vec[2]);

                    const T source = -T(2) * alpha * (*p_A[s]);
                    const T diss =
                        KO6_axis_ptr(p_g, sx) + KO6_axis_ptr(p_g, sy) + KO6_axis_ptr(p_g, 1);
                    const T diss_scaled = (ko_sigma / G.dx) * diss;

                    *p_rhs[s] = adv + lie_vals[s] + source + trace_vals[s] + diss_scaled;
                }

                for (int c = 0; c < 3; ++c)
                    ++p_beta[c];
                for (int s = 0; s < 6; ++s) {
                    ++p_gam[s];
                    ++p_A[s];
                    ++p_rhs[s];
                }
                ++p_alpha;
            }
        }
    }
}
} // namespace tensorium_RG::bssn
