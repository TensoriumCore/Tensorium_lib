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
    const bool use_Z_field = params.evolve_Z || params.frozen_Z_is_synced;

    auto loop_ij = [&](size_t i, size_t j) {
            size_t idx_start = G.Theta.idx(i, j, k0);

            const T *p_theta = G.Theta.ptr() + idx_start;
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;
            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_Z[3] = {G.Z[0].ptr() + idx_start, G.Z[1].ptr() + idx_start,
                               G.Z[2].ptr() + idx_start};
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

            #pragma omp simd aligned(p_theta,p_alpha,p_K,p_chi,p_beta0,p_beta1,p_beta2,p_tg0,p_tg1,p_tg2,p_g0,p_g1,p_g2,p_g3,p_g4,p_g5,p_A0,p_A1,p_A2,p_A3,p_A4,p_A5,p_R0,p_R1,p_R2,p_R3,p_R4,p_R5,p_rhs:64)
            for (size_t k = k0; k < k1; ++k) {
                const T theta = *p_theta;
                const T alpha = *p_alpha;
                const T K_val = *p_K;
                const T bx = *p_beta0;
                const T by = *p_beta1;
                const T bz = *p_beta2;
                const T chi = *p_chi;
                const T chi_guarded = guard_chi_div(chi, params.chi_div_floor);
                T       z_phys[3] = {T(0), T(0), T(0)};
                if (use_Z_field) {
                    z_phys[0] = *p_Z[0];
                    z_phys[1] = *p_Z[1];
                    z_phys[2] = *p_Z[2];
                } else {
                    T div_metric_inv[3] = {T(0), T(0), T(0)};
                    detail::metric_inverse_divergence_ptr(
                        p_g0, p_g1, p_g2, p_g3, p_g4, p_g5, sx, sy, inv_12dx, inv_12dy, inv_12dz,
                        div_metric_inv);
                    // gamma_metric is the contracted conformal Christoffel from metric derivatives.
                    const T gamma_metric[3] = {-div_metric_inv[0], -div_metric_inv[1],
                                               -div_metric_inv[2]};
                    const T z_over_chi[3] = {T(0.5) * (*p_tildeGamma[0] - gamma_metric[0]),
                                             T(0.5) * (*p_tildeGamma[1] - gamma_metric[1]),
                                             T(0.5) * (*p_tildeGamma[2] - gamma_metric[2])};
                    z_phys[0] = chi_guarded * z_over_chi[0];
                    z_phys[1] = chi_guarded * z_over_chi[1];
                    z_phys[2] = chi_guarded * z_over_chi[2];
                }

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
                ++p_Z[0];
                ++p_Z[1];
                ++p_Z[2];
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
 * @brief RHS for the Z4c spatial constraint vector \f$Z^i\f$ (contravariant).
 */
template <typename T>
inline void compute_rhs_Z(const BSSNGridSoA<T> &G, Field3D<T> rhs_Z[3],
                          const GaugeParameters<T> &params = {}, size_t padding = 4) {
    BSSN_PROFILE_KERNEL(Z);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(params.ko_sigma); // Adaptive KO6 for Z^i.

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

    if (!params.evolve_Z) {
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
        return;
    }

    const double inv_12dx = 1.0 / (60.0 * G.dx);
    const double inv_12dy = 1.0 / (60.0 * G.dy);
    const double inv_12dz = 1.0 / (60.0 * G.dz);
    const double inv_2dx = 1.0 / (2.0 * G.dx);
    const double inv_2dy = 1.0 / (2.0 * G.dy);
    const double inv_2dz = 1.0 / (2.0 * G.dz);

    const ptrdiff_t sx = G.Z[0].st.sx;
    const ptrdiff_t sy = G.Z[0].st.sy;
    const T         two_thirds = T(2) / T(3);

    auto loop_ij = [&](size_t i, size_t j) {
            size_t idx_start = G.Z[0].idx(i, j, k0);

            const T *p_Z[3] = {G.Z[0].ptr() + idx_start, G.Z[1].ptr() + idx_start,
                               G.Z[2].ptr() + idx_start};
            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_theta = G.Theta.ptr() + idx_start;
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_gcov[6];
            const T *p_ginv[6];
            const T *p_A[6];
            for (int s = 0; s < 6; ++s) {
                p_gcov[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
            }

            T *p_rhs[3] = {rhs_Z[0].ptr() + idx_start, rhs_Z[1].ptr() + idx_start,
                           rhs_Z[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {
                const T alpha = *p_alpha;
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];

                const T g_xx = *p_ginv[0];
                const T g_xy = *p_ginv[1];
                const T g_xz = *p_ginv[2];
                const T g_yy = *p_ginv[3];
                const T g_yz = *p_ginv[4];
                const T g_zz = *p_ginv[5];

                const T row0_x = g_xx;
                const T row0_y = g_xy;
                const T row0_z = g_xz;
                const T row1_x = g_xy;
                const T row1_y = g_yy;
                const T row1_z = g_yz;
                const T row2_x = g_xz;
                const T row2_y = g_yz;
                const T row2_z = g_zz;

                const T A_xx = *p_A[0];
                const T A_xy = *p_A[1];
                const T A_xz = *p_A[2];
                const T A_yy = *p_A[3];
                const T A_yz = *p_A[4];
                const T A_zz = *p_A[5];

                T A_lo[3][3];
                A_lo[0][0] = A_xx;
                A_lo[0][1] = A_lo[1][0] = A_xy;
                A_lo[0][2] = A_lo[2][0] = A_xz;
                A_lo[1][1] = A_yy;
                A_lo[1][2] = A_lo[2][1] = A_yz;
                A_lo[2][2] = A_zz;

                const T tmp0_x = A_xx * row0_x + A_xy * row0_y + A_xz * row0_z;
                const T tmp0_y = A_xy * row0_x + A_yy * row0_y + A_yz * row0_z;
                const T tmp0_z = A_xz * row0_x + A_yz * row0_y + A_zz * row0_z;

                const T tmp1_x = A_xx * row1_x + A_xy * row1_y + A_xz * row1_z;
                const T tmp1_y = A_xy * row1_x + A_yy * row1_y + A_yz * row1_z;
                const T tmp1_z = A_xz * row1_x + A_yz * row1_y + A_zz * row1_z;

                const T tmp2_x = A_xx * row2_x + A_xy * row2_y + A_xz * row2_z;
                const T tmp2_y = A_xy * row2_x + A_yy * row2_y + A_yz * row2_z;
                const T tmp2_z = A_xz * row2_x + A_yz * row2_y + A_zz * row2_z;

                T A_up_matrix[3][3];
                A_up_matrix[0][0] = row0_x * tmp0_x + row0_y * tmp0_y + row0_z * tmp0_z;
                A_up_matrix[0][1] = A_up_matrix[1][0] =
                    row0_x * tmp1_x + row0_y * tmp1_y + row0_z * tmp1_z;
                A_up_matrix[0][2] = A_up_matrix[2][0] =
                    row0_x * tmp2_x + row0_y * tmp2_y + row0_z * tmp2_z;
                A_up_matrix[1][1] = row1_x * tmp1_x + row1_y * tmp1_y + row1_z * tmp1_z;
                A_up_matrix[1][2] = A_up_matrix[2][1] =
                    row1_x * tmp2_x + row1_y * tmp2_y + row1_z * tmp2_z;
                A_up_matrix[2][2] = row2_x * tmp2_x + row2_y * tmp2_y + row2_z * tmp2_z;

                const T dK[3] = {Dx_ptr(p_K, sx, inv_12dx), Dy_ptr(p_K, sy, inv_12dy),
                                 Dz_ptr(p_K, inv_12dz)};
                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};
                const T chi_val = *p_chi;
                const T z_vals[3] = {*p_Z[0], *p_Z[1], *p_Z[2]};
                T       d6phi[3];
                grad_6phi_from_chi(d_chi, chi_val, d6phi, params.chi_div_floor);

                T beta_grad[3][3];
                for (int comp = 0; comp < 3; ++comp) {
                    beta_grad[comp][0] = Dx_ptr(p_beta[comp], sx, inv_12dx);
                    beta_grad[comp][1] = Dy_ptr(p_beta[comp], sy, inv_12dy);
                    beta_grad[comp][2] = Dz_ptr(p_beta[comp], inv_12dz);
                }

                T dA_dx[6];
                T dA_dy[6];
                T dA_dz[6];
                for (int s = 0; s < 6; ++s) {
                    dA_dx[s] = Dx_ptr(p_A[s], sx, inv_12dx);
                    dA_dy[s] = Dy_ptr(p_A[s], sy, inv_12dy);
                    dA_dz[s] = Dz_ptr(p_A[s], inv_12dz);
                }

                T dA_tensor[3][3][3] = {};
                for (int s = 0; s < 6; ++s) {
                    const int row = detail::sym_row_index[s];
                    const int col = detail::sym_col_index[s];
                    const T   dx = dA_dx[s];
                    const T   dy = dA_dy[s];
                    const T   dz = dA_dz[s];
                    dA_tensor[0][row][col] = dx;
                    dA_tensor[0][col][row] = dx;
                    dA_tensor[1][row][col] = dy;
                    dA_tensor[1][col][row] = dy;
                    dA_tensor[2][row][col] = dz;
                    dA_tensor[2][col][row] = dz;
                }

                T Gamma_tilde_vals[3][3][3];
                tensorium_RG::bssn::compute_tildeGamma_symbols_ptr(
                    p_gcov, p_ginv, sx, sy, inv_12dx, inv_12dy, inv_12dz, Gamma_tilde_vals);

                T covariant[3][3][3];
                for (int dir = 0; dir < 3; ++dir)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) {
                            T sum = dA_tensor[dir][a][b];
                            for (int m = 0; m < 3; ++m) {
                                sum -= Gamma_tilde_vals[m][a][dir] * A_lo[m][b];
                                sum -= Gamma_tilde_vals[m][b][dir] * A_lo[a][m];
                            }
                            covariant[dir][a][b] = sum;
                        }

                T ginv_matrix[3][3];
                ginv_matrix[0][0] = g_xx;
                ginv_matrix[0][1] = ginv_matrix[1][0] = g_xy;
                ginv_matrix[0][2] = ginv_matrix[2][0] = g_xz;
                ginv_matrix[1][1] = g_yy;
                ginv_matrix[1][2] = ginv_matrix[2][1] = g_yz;
                ginv_matrix[2][2] = g_zz;

                T div_vec[3] = {T(0), T(0), T(0)};
                for (int ii = 0; ii < 3; ++ii)
                    for (int dir = 0; dir < 3; ++dir)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                div_vec[ii] +=
                                    ginv_matrix[ii][a] * ginv_matrix[dir][b] * covariant[dir][a][b];

                const T dissZ_coef = ko_sigma / G.dx;

                for (int comp = 0; comp < 3; ++comp) {
                    const T adv = bx * Dx_upwind_ptr(p_Z[comp], sx, inv_2dx, bx) +
                                  by * Dy_upwind_ptr(p_Z[comp], sy, inv_2dy, by) +
                                  bz * Dz_upwind_ptr(p_Z[comp], inv_2dz, bz);

                    const T damping = -params.kappa_z * alpha * (*p_Z[comp]);
                    const T lie_drag =
                        -(z_vals[0] * beta_grad[comp][0] + z_vals[1] * beta_grad[comp][1] +
                          z_vals[2] * beta_grad[comp][2]);

                    T phi_term = T(0);
                    for (int dir = 0; dir < 3; ++dir)
                        phi_term += A_up_matrix[comp][dir] * d6phi[dir];

                    T gradK = T(0);
                    if (comp == 0)
                        gradK = row0_x * dK[0] + row0_y * dK[1] + row0_z * dK[2];
                    else if (comp == 1)
                        gradK = row1_x * dK[0] + row1_y * dK[1] + row1_z * dK[2];
                    else
                        gradK = row2_x * dK[0] + row2_y * dK[1] + row2_z * dK[2];

                    const T gradK_phys = chi_val * gradK;
                    const T momentum = div_vec[comp] + phi_term - two_thirds * gradK_phys;
                    const T diss = KO6_axis_ptr(p_Z[comp], sx) + KO6_axis_ptr(p_Z[comp], sy) +
                                   KO6_axis_ptr(p_Z[comp], 1);

                    *p_rhs[comp] = alpha * momentum + adv + lie_drag + damping + dissZ_coef * diss;
                }

                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_Z[c];
                    ++p_rhs[c];
                }
                ++p_theta;
                ++p_alpha;
                ++p_K;
                ++p_chi;
                for (int s = 0; s < 6; ++s) {
                    ++p_gcov[s];
                    ++p_ginv[s];
                    ++p_A[s];
                }
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
