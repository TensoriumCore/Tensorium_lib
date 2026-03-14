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
template <typename T>
inline void compute_rhs_Theta(const BSSNGridSoA<T> &G, Field3D<T> &rhs_theta,
                              const GaugeParameters<T> &params = {}, size_t padding = 4,
                              const Field3D<T> *z4_conformal_trace_cache = nullptr) {
    BSSN_PROFILE_KERNEL(Theta);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma); // Adaptive KO6 strength.

    const auto region = interior_bounds(G, padding);
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
    const T ko_scale = T(ko_sigma / G.dx);

    auto loop_ij = [&](size_t i, size_t j) {
            size_t idx_start = G.Theta.idx(i, j, k0);

            const T *p_theta = G.Theta.ptr() + idx_start;
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;
            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_tildeGamma[3] = {G.tildeGamma[0].ptr() + idx_start,
                                        G.tildeGamma[1].ptr() + idx_start,
                                        G.tildeGamma[2].ptr() + idx_start};
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_ginv[6];
            const T *p_A[6];
            const T *p_R[6];
            for (int s = 0; s < 6; ++s) {
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
                p_R[s] = G.Ricci[s].ptr() + idx_start;
            }
            const T *p_z4_trace =
                z4_conformal_trace_cache ? (z4_conformal_trace_cache->ptr() + idx_start) : nullptr;

            T *p_rhs = rhs_theta.ptr() + idx_start;

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
            const T *p_A0 = p_A[0];
            const T *p_A1 = p_A[1];
            const T *p_A2 = p_A[2];
            const T *p_A3 = p_A[3];
            const T *p_A4 = p_A[4];
            const T *p_A5 = p_A[5];
            const T *p_R0 = p_R[0];
            const T *p_R1 = p_R[1];
            const T *p_R2 = p_R[2];
            const T *p_R3 = p_R[3];
            const T *p_R4 = p_R[4];
            const T *p_R5 = p_R[5];

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
                // Source Theta from the evolved contracted Gamma, not from the auxiliary Z field.
                T z_over_chi[3] = {T(0), T(0), T(0)};
                recover_z_over_chi_from_gamma_ptr(p_tg0, p_tg1, p_tg2, p_g0, p_g1, p_g2, p_g3,
                                                  p_g4, p_g5, sx, sy, inv_12dx, inv_12dy,
                                                  inv_12dz, z_over_chi);
                const T z_phys[3] = {chi_guarded * z_over_chi[0], chi_guarded * z_over_chi[1],
                                     chi_guarded * z_over_chi[2]};

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};

                const T adv = bx * Dx_upwind_ptr(p_theta, sx, inv_2dx, bx) +
                              by * Dy_upwind_ptr(p_theta, sy, inv_2dy, by) +
                              bz * Dz_upwind_ptr(p_theta, inv_2dz, bz);

                const T g_xx = *p_g0;
                const T g_xy = *p_g1;
                const T g_xz = *p_g2;
                const T g_yy = *p_g3;
                const T g_yz = *p_g4;
                const T g_zz = *p_g5;

                const T R_conformal_base = g_xx * (*p_R0) + g_yy * (*p_R3) + g_zz * (*p_R5) +
                                           T(2) * (g_xy * (*p_R1) + g_xz * (*p_R2) +
                                                   g_yz * (*p_R4));
                T       R_conformal_z4 = T(0);
                if (p_z4_trace) {
                    R_conformal_z4 = *p_z4_trace;
                } else {
                    T RicciZ4[6];
                    compute_RicciZ4_core(G, i, j, k, RicciZ4, params.chi_div_floor, inv_12dx,
                                         inv_12dy, inv_12dz, sx, sy);
                    R_conformal_z4 =
                        g_xx * RicciZ4[0] + g_yy * RicciZ4[3] + g_zz * RicciZ4[5] +
                        T(2) * (g_xy * RicciZ4[1] + g_xz * RicciZ4[2] + g_yz * RicciZ4[4]);
                }
                const T R_conformal = R_conformal_base + R_conformal_z4;
                const T R_scalar = chi_guarded * R_conformal;

                const T row0_x = g_xx;
                const T row0_y = g_xy;
                const T row0_z = g_xz;
                const T row1_x = g_xy;
                const T row1_y = g_yy;
                const T row1_z = g_yz;
                const T row2_x = g_xz;
                const T row2_y = g_yz;
                const T row2_z = g_zz;

                const T A_xx = *p_A0;
                const T A_xy = *p_A1;
                const T A_xz = *p_A2;
                const T A_yy = *p_A3;
                const T A_yz = *p_A4;
                const T A_zz = *p_A5;

                const T tmp0_x = A_xx * row0_x + A_xy * row0_y + A_xz * row0_z;
                const T tmp0_y = A_xy * row0_x + A_yy * row0_y + A_yz * row0_z;
                const T tmp0_z = A_xz * row0_x + A_yz * row0_y + A_zz * row0_z;

                const T tmp1_x = A_xx * row1_x + A_xy * row1_y + A_xz * row1_z;
                const T tmp1_y = A_xy * row1_x + A_yy * row1_y + A_yz * row1_z;
                const T tmp1_z = A_xz * row1_x + A_yz * row1_y + A_zz * row1_z;

                const T tmp2_x = A_xx * row2_x + A_xy * row2_y + A_xz * row2_z;
                const T tmp2_y = A_xy * row2_x + A_yy * row2_y + A_yz * row2_z;
                const T tmp2_z = A_xz * row2_x + A_yz * row2_y + A_zz * row2_z;

                const T A_up_xx = row0_x * tmp0_x + row0_y * tmp0_y + row0_z * tmp0_z;
                const T A_up_xy = row0_x * tmp1_x + row0_y * tmp1_y + row0_z * tmp1_z;
                const T A_up_xz = row0_x * tmp2_x + row0_y * tmp2_y + row0_z * tmp2_z;
                const T A_up_yy = row1_x * tmp1_x + row1_y * tmp1_y + row1_z * tmp1_z;
                const T A_up_yz = row1_x * tmp2_x + row1_y * tmp2_y + row1_z * tmp2_z;
                const T A_up_zz = row2_x * tmp2_x + row2_y * tmp2_y + row2_z * tmp2_z;

                const T A_contract = A_xx * A_up_xx + A_yy * A_up_yy + A_zz * A_up_zz +
                                     T(2) * (A_xy * A_up_xy + A_xz * A_up_xz + A_yz * A_up_yz);

                const T geom_source = T(0.5) * (R_scalar - A_contract + two_thirds * K_val * K_val -
                                                T(2) * theta * K_val);
                const T Z_dot_dalpha =
                    z_phys[0] * d_alpha[0] + z_phys[1] * d_alpha[1] + z_phys[2] * d_alpha[2];
                const T geom = alpha * geom_source;
                const T damping = -params.kappa1_times_lapse(alpha) * two_plus_kappa2 * theta;
                const T diss = KO6_axis_ptr(p_theta, sx) + KO6_axis_ptr(p_theta, sy) +
                               KO6_axis_ptr(p_theta, 1);

                *p_rhs = geom + damping + adv - Z_dot_dalpha + ko_scale * diss;

                ++p_theta;
                ++p_alpha;
                ++p_K;
                ++p_rhs;
                ++p_chi;
                if (p_z4_trace)
                    ++p_z4_trace;
                ++p_beta0;
                ++p_beta1;
                ++p_beta2;
                ++p_tg0;
                ++p_tg1;
                ++p_tg2;
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

    const auto region = interior_bounds(G, padding);
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
