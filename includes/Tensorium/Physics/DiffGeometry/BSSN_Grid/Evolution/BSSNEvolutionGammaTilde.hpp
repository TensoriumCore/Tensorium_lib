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

namespace detail {

template <int Order, typename T>
inline void compute_rhs_gamma_tilde_impl(const BSSNGridSoA<T> &G, Field3D<T> rhs[6],
                                         size_t padding, const GaugeParameters<T> &params) {
    BSSN_PROFILE_KERNEL(GammaTilde);
    const auto region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
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

    auto loop_ij = [&](size_t i, size_t j) {

        size_t idx_start = G.gamma_tilde[0].idx(i, j, k0);

        const T *p_beta0 = G.beta[0].ptr() + idx_start;
        const T *p_beta1 = G.beta[1].ptr() + idx_start;
        const T *p_beta2 = G.beta[2].ptr() + idx_start;
        const T *p_gam0 = G.gamma_tilde[XX].ptr() + idx_start;
        const T *p_gam1 = G.gamma_tilde[XY].ptr() + idx_start;
        const T *p_gam2 = G.gamma_tilde[XZ].ptr() + idx_start;
        const T *p_gam3 = G.gamma_tilde[YY].ptr() + idx_start;
        const T *p_gam4 = G.gamma_tilde[YZ].ptr() + idx_start;
        const T *p_gam5 = G.gamma_tilde[ZZ].ptr() + idx_start;
        const T *p_A0 = G.A_tilde[XX].ptr() + idx_start;
        const T *p_A1 = G.A_tilde[XY].ptr() + idx_start;
        const T *p_A2 = G.A_tilde[XZ].ptr() + idx_start;
        const T *p_A3 = G.A_tilde[YY].ptr() + idx_start;
        const T *p_A4 = G.A_tilde[YZ].ptr() + idx_start;
        const T *p_A5 = G.A_tilde[ZZ].ptr() + idx_start;
        T       *p_rhs0 = rhs[XX].ptr() + idx_start;
        T       *p_rhs1 = rhs[XY].ptr() + idx_start;
        T       *p_rhs2 = rhs[XZ].ptr() + idx_start;
        T       *p_rhs3 = rhs[YY].ptr() + idx_start;
        T       *p_rhs4 = rhs[YZ].ptr() + idx_start;
        T       *p_rhs5 = rhs[ZZ].ptr() + idx_start;
        const T *p_alpha = G.alpha.ptr() + idx_start;

#pragma omp simd
        for (size_t k = k0; k < k1; ++k) {

                const T alpha = *p_alpha;

                T d_beta[3][3];
                d_beta[0][0] = tensorium_RG::fd::Dx_ptr_order<Order>(p_beta0, sx, inv_12dx);
                d_beta[0][1] = tensorium_RG::fd::Dy_ptr_order<Order>(p_beta0, sy, inv_12dy);
                d_beta[0][2] = tensorium_RG::fd::Dz_ptr_order<Order>(p_beta0, inv_12dz);
                d_beta[1][0] = tensorium_RG::fd::Dx_ptr_order<Order>(p_beta1, sx, inv_12dx);
                d_beta[1][1] = tensorium_RG::fd::Dy_ptr_order<Order>(p_beta1, sy, inv_12dy);
                d_beta[1][2] = tensorium_RG::fd::Dz_ptr_order<Order>(p_beta1, inv_12dz);
                d_beta[2][0] = tensorium_RG::fd::Dx_ptr_order<Order>(p_beta2, sx, inv_12dx);
                d_beta[2][1] = tensorium_RG::fd::Dy_ptr_order<Order>(p_beta2, sy, inv_12dy);
                d_beta[2][2] = tensorium_RG::fd::Dz_ptr_order<Order>(p_beta2, inv_12dz);
                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                const T beta0 = *p_beta0;
                const T beta1 = *p_beta1;
                const T beta2 = *p_beta2;

                const T g_xx = *p_gam0;
                const T g_xy = *p_gam1;
                const T g_xz = *p_gam2;
                const T g_yy = *p_gam3;
                const T g_yz = *p_gam4;
                const T g_zz = *p_gam5;

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

                const T ko_scale = local_ko_scale(G, ko_sigma, i, j, k);

                const T adv0 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam0, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam0, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam0, inv_2dz, beta2);
                const T adv1 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam1, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam1, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam1, inv_2dz, beta2);
                const T adv2 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam2, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam2, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam2, inv_2dz, beta2);
                const T adv3 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam3, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam3, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam3, inv_2dz, beta2);
                const T adv4 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam4, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam4, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam4, inv_2dz, beta2);
                const T adv5 = beta0 * tensorium_RG::fd::Dx_upwind_ptr(p_gam5, sx, inv_2dx, beta0) +
                               beta1 * tensorium_RG::fd::Dy_upwind_ptr(p_gam5, sy, inv_2dy, beta1) +
                               beta2 * tensorium_RG::fd::Dz_upwind_ptr(p_gam5, inv_2dz, beta2);

                const T diss0 = tensorium_RG::fd::KO6_axis_ptr(p_gam0, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam0, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam0, 1);
                const T diss1 = tensorium_RG::fd::KO6_axis_ptr(p_gam1, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam1, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam1, 1);
                const T diss2 = tensorium_RG::fd::KO6_axis_ptr(p_gam2, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam2, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam2, 1);
                const T diss3 = tensorium_RG::fd::KO6_axis_ptr(p_gam3, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam3, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam3, 1);
                const T diss4 = tensorium_RG::fd::KO6_axis_ptr(p_gam4, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam4, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam4, 1);
                const T diss5 = tensorium_RG::fd::KO6_axis_ptr(p_gam5, sx) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam5, sy) +
                                tensorium_RG::fd::KO6_axis_ptr(p_gam5, 1);

                *p_rhs0 = adv0 + lie_vals[0] - T(2) * alpha * (*p_A0) + trace_vals[0] + ko_scale * diss0;
                *p_rhs1 = adv1 + lie_vals[1] - T(2) * alpha * (*p_A1) + trace_vals[1] + ko_scale * diss1;
                *p_rhs2 = adv2 + lie_vals[2] - T(2) * alpha * (*p_A2) + trace_vals[2] + ko_scale * diss2;
                *p_rhs3 = adv3 + lie_vals[3] - T(2) * alpha * (*p_A3) + trace_vals[3] + ko_scale * diss3;
                *p_rhs4 = adv4 + lie_vals[4] - T(2) * alpha * (*p_A4) + trace_vals[4] + ko_scale * diss4;
                *p_rhs5 = adv5 + lie_vals[5] - T(2) * alpha * (*p_A5) + trace_vals[5] + ko_scale * diss5;

                ++p_beta0;
                ++p_beta1;
                ++p_beta2;
                ++p_gam0;
                ++p_gam1;
                ++p_gam2;
                ++p_gam3;
                ++p_gam4;
                ++p_gam5;
                ++p_A0;
                ++p_A1;
                ++p_A2;
                ++p_A3;
                ++p_A4;
                ++p_A5;
                ++p_rhs0;
                ++p_rhs1;
                ++p_rhs2;
                ++p_rhs3;
                ++p_rhs4;
                ++p_rhs5;
                ++p_alpha;
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

} // namespace detail

template <typename T>
inline void compute_rhs_gamma_tilde(const BSSNGridSoA<T> &G, Field3D<T> rhs[6], size_t padding = 4,
                                    const GaugeParameters<T> &params = {}) {
    if (tensorium_RG::fd::max_spatial_derivative_order() == 4) {
        detail::compute_rhs_gamma_tilde_impl<4>(G, rhs, padding, params);
    } else {
        detail::compute_rhs_gamma_tilde_impl<6>(G, rhs, padding, params);
    }
}
} // namespace tensorium_RG::bssn
