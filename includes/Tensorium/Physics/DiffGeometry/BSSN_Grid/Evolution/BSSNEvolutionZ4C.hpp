#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNConformal.hpp"
#include "../TimeIntegration/BSSNPerfTimers.hpp"
#include "BSSNEvolutionGauge.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

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
constexpr int sym_row_index[6] = {0, 0, 0, 1, 1, 2};
constexpr int sym_col_index[6] = {0, 1, 2, 1, 2, 2};
} // namespace detail

/**
 * @brief RHS for the Z4c scalar \f$\Theta\f$.
 */
namespace detail {

template <int Order, bool UseZ4TraceCache, typename T>
inline void compute_rhs_Theta_impl(const BSSNGridSoA<T> &G, Field3D<T> &rhs_theta,
                                   const GaugeParameters<T> &params, size_t padding,
                                   const Field3D<T> *z4_conformal_trace_cache) {
    BSSN_PROFILE_KERNEL(Theta);

    const auto   region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
        return;

    const double inv_12dx = 1.0 / (60.0 * G.dx);
    const double inv_12dy = 1.0 / (60.0 * G.dy);
    const double inv_12dz = 1.0 / (60.0 * G.dz);
    const double inv_2dx = 1.0 / (2.0 * G.dx);
    const double inv_2dy = 1.0 / (2.0 * G.dy);
    const double inv_2dz = 1.0 / (2.0 * G.dz);

    const ptrdiff_t sx = G.Theta.st.sx;
    const ptrdiff_t sy = G.Theta.st.sy;

    const T two_plus_kappa2 = T(2) + params.kappa2;
    const T two_thirds = T(2) / T(3);
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma);

    auto loop_ij = [&](size_t i, size_t j) {
        size_t idx_start = G.Theta.idx(i, j, k0);

        const T *__restrict p_theta = G.Theta.ptr() + idx_start;
        const T *__restrict p_alpha = G.alpha.ptr() + idx_start;
        const T *__restrict p_K = G.K.ptr() + idx_start;
        const T *__restrict p_beta0 = G.beta[0].ptr() + idx_start;
        const T *__restrict p_beta1 = G.beta[1].ptr() + idx_start;
        const T *__restrict p_beta2 = G.beta[2].ptr() + idx_start;
        const T *__restrict p_tg0 = G.tildeGamma[0].ptr() + idx_start;
        const T *__restrict p_tg1 = G.tildeGamma[1].ptr() + idx_start;
        const T *__restrict p_tg2 = G.tildeGamma[2].ptr() + idx_start;
        const T *__restrict p_chi = G.chi.ptr() + idx_start;
        const T *__restrict p_g0 = G.gamma_tilde_inv[0].ptr() + idx_start;
        const T *__restrict p_g1 = G.gamma_tilde_inv[1].ptr() + idx_start;
        const T *__restrict p_g2 = G.gamma_tilde_inv[2].ptr() + idx_start;
        const T *__restrict p_g3 = G.gamma_tilde_inv[3].ptr() + idx_start;
        const T *__restrict p_g4 = G.gamma_tilde_inv[4].ptr() + idx_start;
        const T *__restrict p_g5 = G.gamma_tilde_inv[5].ptr() + idx_start;
        const T *__restrict p_A0 = G.A_tilde[0].ptr() + idx_start;
        const T *__restrict p_A1 = G.A_tilde[1].ptr() + idx_start;
        const T *__restrict p_A2 = G.A_tilde[2].ptr() + idx_start;
        const T *__restrict p_A3 = G.A_tilde[3].ptr() + idx_start;
        const T *__restrict p_A4 = G.A_tilde[4].ptr() + idx_start;
        const T *__restrict p_A5 = G.A_tilde[5].ptr() + idx_start;
        const T *__restrict p_R0 = G.Ricci[0].ptr() + idx_start;
        const T *__restrict p_R1 = G.Ricci[1].ptr() + idx_start;
        const T *__restrict p_R2 = G.Ricci[2].ptr() + idx_start;
        const T *__restrict p_R3 = G.Ricci[3].ptr() + idx_start;
        const T *__restrict p_R4 = G.Ricci[4].ptr() + idx_start;
        const T *__restrict p_R5 = G.Ricci[5].ptr() + idx_start;
        const T *__restrict p_z4_trace =
            UseZ4TraceCache ? (z4_conformal_trace_cache->ptr() + idx_start) : nullptr;
        T *__restrict p_rhs = rhs_theta.ptr() + idx_start;

#pragma omp simd
        for (size_t k = k0; k < k1; ++k) {
            const T theta = *p_theta;
            const T alpha = *p_alpha;
            const T K_val = *p_K;
            const T bx = *p_beta0;
            const T by = *p_beta1;
            const T bz = *p_beta2;
            const T chi = *p_chi;
            const T chi_guarded = guard_chi_div(chi, params.chi_div_floor);

            T z_over_chi0 = T(0);
            T z_over_chi1 = T(0);
            T z_over_chi2 = T(0);
            recover_z_over_chi_components_from_gamma_ptr_order<Order>(
                p_tg0, p_tg1, p_tg2, p_g0, p_g1, p_g2, p_g3, p_g4, p_g5, sx, sy, inv_12dx,
                inv_12dy, inv_12dz, z_over_chi0, z_over_chi1, z_over_chi2);
            const T z_phys0 = chi_guarded * z_over_chi0;
            const T z_phys1 = chi_guarded * z_over_chi1;
            const T z_phys2 = chi_guarded * z_over_chi2;

            const T d_alpha_x = tensorium_RG::fd::Dx_ptr_order<Order>(p_alpha, sx, inv_12dx);
            const T d_alpha_y = tensorium_RG::fd::Dy_ptr_order<Order>(p_alpha, sy, inv_12dy);
            const T d_alpha_z = tensorium_RG::fd::Dz_ptr_order<Order>(p_alpha, inv_12dz);

            const T adv = bx * tensorium_RG::fd::Dx_upwind_ptr(p_theta, sx, inv_2dx, bx) +
                          by * tensorium_RG::fd::Dy_upwind_ptr(p_theta, sy, inv_2dy, by) +
                          bz * tensorium_RG::fd::Dz_upwind_ptr(p_theta, inv_2dz, bz);

            const T g_xx = *p_g0;
            const T g_xy = *p_g1;
            const T g_xz = *p_g2;
            const T g_yy = *p_g3;
            const T g_yz = *p_g4;
            const T g_zz = *p_g5;

            const T R_conformal_base = g_xx * (*p_R0) + g_yy * (*p_R3) + g_zz * (*p_R5) +
                                       T(2) * (g_xy * (*p_R1) + g_xz * (*p_R2) + g_yz * (*p_R4));
            T R_conformal_z4 = T(0);
            if constexpr (UseZ4TraceCache) {
                R_conformal_z4 = *p_z4_trace;
            } else {
                // Contracting the Z4 Ricci correction with g^{ab} reduces analytically to
                // -(Z^m / chi) * d_m chi / chi, so we can avoid materializing RicciZ4[ab]
                // inside the SIMD loop.
                const T d_chi_x = tensorium_RG::fd::Dx_ptr_order<Order>(p_chi, sx, inv_12dx);
                const T d_chi_y = tensorium_RG::fd::Dy_ptr_order<Order>(p_chi, sy, inv_12dy);
                const T d_chi_z = tensorium_RG::fd::Dz_ptr_order<Order>(p_chi, inv_12dz);
                R_conformal_z4 =
                    -(z_over_chi0 * d_chi_x + z_over_chi1 * d_chi_y + z_over_chi2 * d_chi_z) /
                    chi_guarded;
            }
            const T R_scalar = chi_guarded * (R_conformal_base + R_conformal_z4);

            const T A_xx = *p_A0;
            const T A_xy = *p_A1;
            const T A_xz = *p_A2;
            const T A_yy = *p_A3;
            const T A_yz = *p_A4;
            const T A_zz = *p_A5;

            const T tmp0_x = A_xx * g_xx + A_xy * g_xy + A_xz * g_xz;
            const T tmp0_y = A_xy * g_xx + A_yy * g_xy + A_yz * g_xz;
            const T tmp0_z = A_xz * g_xx + A_yz * g_xy + A_zz * g_xz;

            const T tmp1_x = A_xx * g_xy + A_xy * g_yy + A_xz * g_yz;
            const T tmp1_y = A_xy * g_xy + A_yy * g_yy + A_yz * g_yz;
            const T tmp1_z = A_xz * g_xy + A_yz * g_yy + A_zz * g_yz;

            const T tmp2_x = A_xx * g_xz + A_xy * g_yz + A_xz * g_zz;
            const T tmp2_y = A_xy * g_xz + A_yy * g_yz + A_yz * g_zz;
            const T tmp2_z = A_xz * g_xz + A_yz * g_yz + A_zz * g_zz;

            const T A_up_xx = g_xx * tmp0_x + g_xy * tmp0_y + g_xz * tmp0_z;
            const T A_up_xy = g_xx * tmp1_x + g_xy * tmp1_y + g_xz * tmp1_z;
            const T A_up_xz = g_xx * tmp2_x + g_xy * tmp2_y + g_xz * tmp2_z;
            const T A_up_yy = g_xy * tmp1_x + g_yy * tmp1_y + g_yz * tmp1_z;
            const T A_up_yz = g_xy * tmp2_x + g_yy * tmp2_y + g_yz * tmp2_z;
            const T A_up_zz = g_xz * tmp2_x + g_yz * tmp2_y + g_zz * tmp2_z;

            const T A_contract = A_xx * A_up_xx + A_yy * A_up_yy + A_zz * A_up_zz +
                                 T(2) * (A_xy * A_up_xy + A_xz * A_up_xz + A_yz * A_up_yz);

            const T geom_source = T(0.5) * (R_scalar - A_contract + two_thirds * K_val * K_val -
                                            T(2) * theta * K_val);
            const T Z_dot_dalpha = z_phys0 * d_alpha_x + z_phys1 * d_alpha_y + z_phys2 * d_alpha_z;
            const T damping = -params.kappa1_times_lapse(alpha) * two_plus_kappa2 * theta;
            const T diss = tensorium_RG::fd::KO6_axis_ptr(p_theta, sx) +
                           tensorium_RG::fd::KO6_axis_ptr(p_theta, sy) +
                           tensorium_RG::fd::KO6_axis_ptr(p_theta, 1);

            *p_rhs = alpha * geom_source + damping + adv - Z_dot_dalpha +
                     local_ko_scale(G, ko_sigma, i, j, k) * diss;

            ++p_theta;
            ++p_alpha;
            ++p_K;
            ++p_beta0;
            ++p_beta1;
            ++p_beta2;
            ++p_tg0;
            ++p_tg1;
            ++p_tg2;
            ++p_chi;
            ++p_g0;
            ++p_g1;
            ++p_g2;
            ++p_g3;
            ++p_g4;
            ++p_g5;
            ++p_A0;
            ++p_A1;
            ++p_A2;
            ++p_A3;
            ++p_A4;
            ++p_A5;
            ++p_R0;
            ++p_R1;
            ++p_R2;
            ++p_R3;
            ++p_R4;
            ++p_R5;
            if constexpr (UseZ4TraceCache)
                ++p_z4_trace;
            ++p_rhs;
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
inline void compute_rhs_Theta(const BSSNGridSoA<T> &G, Field3D<T> &rhs_theta,
                              const GaugeParameters<T> &params = {}, size_t padding = 4,
                              const Field3D<T> *z4_conformal_trace_cache = nullptr) {
    const bool use_z4_trace_cache = (z4_conformal_trace_cache != nullptr);
    const bool use_order4 = (tensorium_RG::fd::max_spatial_derivative_order() == 4);

    if (use_z4_trace_cache) {
        if (use_order4) {
            detail::compute_rhs_Theta_impl<4, true>(G, rhs_theta, params, padding,
                                                    z4_conformal_trace_cache);
        } else {
            detail::compute_rhs_Theta_impl<6, true>(G, rhs_theta, params, padding,
                                                    z4_conformal_trace_cache);
        }
    } else {
        if (use_order4) {
            detail::compute_rhs_Theta_impl<4, false>(G, rhs_theta, params, padding,
                                                     static_cast<const Field3D<T> *>(nullptr));
        } else {
            detail::compute_rhs_Theta_impl<6, false>(G, rhs_theta, params, padding,
                                                     static_cast<const Field3D<T> *>(nullptr));
        }
    }
}

/**
 * @brief Diagnostic-only RHS for the Z4c spatial constraint vector \f$Z^i\f$.
 * @details
 * The reference GRChombo/Athenak formulations carry the spatial Z4 information through the
 * contracted conformal connection. We therefore reconstruct `G.Z` from `tildeGamma` for
 * diagnostics/exports and keep its RHS identically zero.
 */
template <typename T>
inline void compute_rhs_Z(const BSSNGridSoA<T> &G, Field3D<T> rhs_Z[3],
                          const GaugeParameters<T> &params = {}, size_t padding = 4) {
    BSSN_PROFILE_KERNEL(Z);
    (void)params;

    const auto   region = interior_bounds(G, padding);
    const size_t i0 = region.i0;
    const size_t j0 = region.j0;
    const size_t k0 = region.k0;
    const size_t i1 = region.i1;
    const size_t j1 = region.j1;
    const size_t k1 = region.k1;

    if (region.empty())
        return;

    auto loop_zero_ij = [&](size_t i, size_t j) {
        size_t idx_start = rhs_Z[0].idx(i, j, k0);
        T     *p_rhs0 = rhs_Z[0].ptr() + idx_start;
        T     *p_rhs1 = rhs_Z[1].ptr() + idx_start;
        T     *p_rhs2 = rhs_Z[2].ptr() + idx_start;
        for (size_t k = k0; k < k1; ++k) {
            *p_rhs0 = T(0);
            *p_rhs1 = T(0);
            *p_rhs2 = T(0);
            ++p_rhs0;
            ++p_rhs1;
            ++p_rhs2;
        }
    };

    if (rhs_kernel_team_mode_enabled()) {
#pragma omp for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_zero_ij(i, j);
    } else {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i)
            for (size_t j = j0; j < j1; ++j)
                loop_zero_ij(i, j);
    }
}

} // namespace tensorium_RG::bssn
