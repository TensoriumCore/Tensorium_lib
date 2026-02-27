#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../TimeIntegration/BSSNPerfTimers.hpp"
#include "BSSNEvolutionCommon.hpp"
#include "BSSNEvolutionGauge.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionGamma.hpp
 * @brief RHS for the contracted connection \f$\tilde{\Gamma}^i\f$.
 * @details Implements the standard BSSN equation with advection, metric-Laplacian of the shift,
 * lapse coupling, and Ricci-driven source terms:
 * \f[
 * \begin{aligned}
 * \partial_t \tilde{\Gamma}^i &= \beta^k\partial_k\tilde{\Gamma}^i -
 * \tilde{\Gamma}^k\partial_k\beta^i
 *   + \tfrac{2}{3}\tilde{\Gamma}^i\partial_k\beta^k
 *   + \tilde{\gamma}^{jk}\partial_j\partial_k\beta^i +
 * \tfrac{1}{3}\tilde{\gamma}^{ij}\partial_j\partial_k\beta^k \\
 *   &\quad - 2\tilde{A}^{ij}\partial_j\alpha + 2\alpha\left(\tilde{\Gamma}^i_{\ jk}\tilde{A}^{jk} -
 * \tfrac{2}{3}\tilde{\gamma}^{ij}\partial_j K\right)
 *   + \mathcal{D}_6[\tilde{\Gamma}^i].
 * \end{aligned}
 * \f]
 */

namespace tensorium_RG::bssn {

/**
 * @brief Assemble \f$\partial_t\tilde{\Gamma}^i\f$ inside the interior region.
 * @param rhs Array of 3 fields that will store the RHS.
 * @details
 * - `adv` = \f$\beta^k\partial_k\tilde{\Gamma}^i\f$ uses upwind stencils.
 * - `gamma_beta` and `stretch` implement the Lie derivative terms
 * \f$-\tilde{\Gamma}^k\partial_k\beta^i\f$ and
 *   \f$\tfrac{2}{3}\tilde{\Gamma}^i\partial_k\beta^k\f$.
 * - `hess_term` and `grad_div_term` build the shift Laplacian pieces.
 * - `A_grad_alpha` and `source` encode \f$-2\tilde{A}^{ij}\partial_j\alpha\f$ and the Ricci
 * coupling.
 */
template <typename T>
inline void compute_rhs_Gamma(const BSSNGridSoA<T> &G, Field3D<T> rhs[3],
                              const Field3D<T> Z[3], const Field3D<T> &Theta,
                              const GaugeParameters<T> &gauge_params = {}, size_t padding = 4) {
    BSSN_PROFILE_KERNEL(Gamma);
    using namespace tensorium_RG::fd;
    const T ko_sigma = scaled_ko_sigma(gauge_params.ko_sigma); // Adaptive KO6 knob.
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
    constexpr size_t gamma_validation_stride = 8;
#endif

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1) {
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
        tensorium_RG::bssn::detail::finalize_contracted_warning_step();
#endif
        return;
    }

    const double inv_12dx = 1.0 / (60.0 * G.dx);
    const double inv_12dy = 1.0 / (60.0 * G.dy);
    const double inv_12dz = 1.0 / (60.0 * G.dz);
    const double inv_2dx = 1.0 / (2.0 * G.dx);
    const double inv_2dy = 1.0 / (2.0 * G.dy);
    const double inv_2dz = 1.0 / (2.0 * G.dz);
    const T      inv_third = T(1) / T(3);
    const T      two_thirds = T(2) / T(3);

    const double inv_12dx2 = 1.0 / (180.0 * G.dx * G.dx);
    const double inv_12dy2 = 1.0 / (180.0 * G.dy * G.dy);
    const double inv_12dz2 = 1.0 / (180.0 * G.dz * G.dz);
    const double inv_144dxdy = 1.0 / (144.0 * G.dx * G.dy);
    const double inv_144dxdz = 1.0 / (144.0 * G.dx * G.dz);
    const double inv_144dydz = 1.0 / (144.0 * G.dy * G.dz);

    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;
    const T         ko_scale = T(ko_sigma / G.dx);

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.alpha.idx(i, j, k0);

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};
            const T *p_Gamma[3] = {G.tildeGamma[0].ptr() + idx_start,
                                   G.tildeGamma[1].ptr() + idx_start,
                                   G.tildeGamma[2].ptr() + idx_start};
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;
            const T *p_theta = Theta.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_Z[3] = {Z[0].ptr() + idx_start, Z[1].ptr() + idx_start,
                               Z[2].ptr() + idx_start};

            const T *p_ginv[6];
            const T *p_gcov[6];
            const T *p_A[6];
            for (int s = 0; s < 6; ++s) {
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_gcov[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
            }

            T *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                           rhs[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {

                T Gamma_tilde_vals[3][3][3];
                tensorium_RG::bssn::compute_tildeGamma_symbols_ptr(
                    p_gcov, p_ginv, sx, sy, inv_12dx, inv_12dy, inv_12dz, Gamma_tilde_vals);
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
                if (((i - i0) % gamma_validation_stride == 0) &&
                    ((j - j0) % gamma_validation_stride == 0) &&
                    ((k - k0) % gamma_validation_stride == 0)) {
                    tensorium_RG::bssn::validate_tildeGamma_symbols(G, i, j, k, Gamma_tilde_vals);
                }
#endif

                const T g_xx = *p_ginv[0];
                const T g_xy = *p_ginv[1];
                const T g_xz = *p_ginv[2];
                const T g_yy = *p_ginv[3];
                const T g_yz = *p_ginv[4];
                const T g_zz = *p_ginv[5];

                const T g_cov_xx = *p_gcov[0];
                const T g_cov_xy = *p_gcov[1];
                const T g_cov_xz = *p_gcov[2];
                const T g_cov_yy = *p_gcov[3];
                const T g_cov_yz = *p_gcov[4];
                const T g_cov_zz = *p_gcov[5];

                const T cov_row0_x = g_cov_xx;
                const T cov_row0_y = g_cov_xy;
                const T cov_row0_z = g_cov_xz;
                const T cov_row1_x = g_cov_xy;
                const T cov_row1_y = g_cov_yy;
                const T cov_row1_z = g_cov_yz;
                const T cov_row2_x = g_cov_xz;
                const T cov_row2_y = g_cov_yz;
                const T cov_row2_z = g_cov_zz;

                const T A_xx = *p_A[0];
                const T A_xy = *p_A[1];
                const T A_xz = *p_A[2];
                const T A_yy = *p_A[3];
                const T A_yz = *p_A[4];
                const T A_zz = *p_A[5];

                const T row0_x = g_xx;
                const T row0_y = g_xy;
                const T row0_z = g_xz;
                const T row1_x = g_xy;
                const T row1_y = g_yy;
                const T row1_z = g_yz;
                const T row2_x = g_xz;
                const T row2_y = g_yz;
                const T row2_z = g_zz;

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

                T A_up_matrix[3][3];
                A_up_matrix[0][0] = A_up_xx;
                A_up_matrix[0][1] = A_up_matrix[1][0] = A_up_xy;
                A_up_matrix[0][2] = A_up_matrix[2][0] = A_up_xz;
                A_up_matrix[1][1] = A_up_yy;
                A_up_matrix[1][2] = A_up_matrix[2][1] = A_up_yz;
                A_up_matrix[2][2] = A_up_zz;
#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)
                if (((i - i0) % gamma_validation_stride == 0) &&
                    ((j - j0) % gamma_validation_stride == 0) &&
                    ((k - k0) % gamma_validation_stride == 0)) {
                    T ginv_matrix[3][3];
                    ginv_matrix[0][0] = g_xx;
                    ginv_matrix[0][1] = ginv_matrix[1][0] = g_xy;
                    ginv_matrix[0][2] = ginv_matrix[2][0] = g_xz;
                    ginv_matrix[1][1] = g_yy;
                    ginv_matrix[1][2] = ginv_matrix[2][1] = g_yz;
                    ginv_matrix[2][2] = g_zz;

                    T A_lo_matrix[3][3];
                    A_lo_matrix[0][0] = A_xx;
                    A_lo_matrix[0][1] = A_lo_matrix[1][0] = A_xy;
                    A_lo_matrix[0][2] = A_lo_matrix[2][0] = A_xz;
                    A_lo_matrix[1][1] = A_yy;
                    A_lo_matrix[1][2] = A_lo_matrix[2][1] = A_yz;
                    A_lo_matrix[2][2] = A_zz;

                    T A_up_ref[3][3];
                    tensorium_RG::bssn::validate_A_raising(G, i, j, k, ginv_matrix,
                                                           A_lo_matrix, A_up_matrix, A_up_ref);
                    tensorium_RG::bssn::validate_gamma_source_term(G, i, j, k, Gamma_tilde_vals,
                                                                   A_up_matrix, A_up_ref);
                }
#endif

                const T beta_vec[3] = {*p_beta[0], *p_beta[1], *p_beta[2]};
                const T gamma_vec[3] = {*p_Gamma[0], *p_Gamma[1], *p_Gamma[2]};

                T d_beta[3][3];
                for (int c = 0; c < 3; ++c) {
                    d_beta[c][0] = Dx_ptr(p_beta[c], sx, inv_12dx);
                    d_beta[c][1] = Dy_ptr(p_beta[c], sy, inv_12dy);
                    d_beta[c][2] = Dz_ptr(p_beta[c], inv_12dz);
                }
                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];

                T grad_div[3] = {0, 0, 0};
                T hess_contr[3] = {0, 0, 0};

                for (int comp = 0; comp < 3; ++comp) {
                    const T dxx = Dxx_ptr(p_beta[comp], sx, inv_12dx2);
                    const T dyy = Dyy_ptr(p_beta[comp], sy, inv_12dy2);
                    const T dzz = Dzz_ptr(p_beta[comp], inv_12dz2);
                    const T dxy = Dxy4_ptr(p_beta[comp], sx, sy, inv_144dxdy);
                    const T dxz = Dxz4_ptr(p_beta[comp], sx, inv_144dxdz);
                    const T dyz = Dyz4_ptr(p_beta[comp], sy, inv_144dydz);

                    if (comp == 0)
                        grad_div[0] += dxx;
                    else if (comp == 1)
                        grad_div[0] += dxy;
                    else
                        grad_div[0] += dxz;

                    if (comp == 0)
                        grad_div[1] += dxy;
                    else if (comp == 1)
                        grad_div[1] += dyy;
                    else
                        grad_div[1] += dyz;

                    if (comp == 0)
                        grad_div[2] += dxz;
                    else if (comp == 1)
                        grad_div[2] += dyz;
                    else
                        grad_div[2] += dzz;

                    hess_contr[comp] = g_xx * dxx + g_yy * dyy + g_zz * dzz +
                                       T(2) * (g_xy * dxy + g_xz * dxz + g_yz * dyz);
                }

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};
                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};
                const T d_theta[3] = {Dx_ptr(p_theta, sx, inv_12dx), Dy_ptr(p_theta, sy, inv_12dy),
                                      Dz_ptr(p_theta, inv_12dz)};
                const T d_K[3] = {Dx_ptr(p_K, sx, inv_12dx), Dy_ptr(p_K, sy, inv_12dy),
                                  Dz_ptr(p_K, inv_12dz)};
                const T alpha = *p_alpha;
                const T chi = *p_chi;
                const T theta_val = *p_theta;
                const T chi_guarded = guard_chi_div(chi, gauge_params.chi_div_floor);
                const T K_val = *p_K;
                const T kappa1_lapse = gauge_params.kappa1_times_lapse(alpha);

                const T metric_dot_grad_div[3] = {
                    row0_x * grad_div[0] + row0_y * grad_div[1] + row0_z * grad_div[2],
                    row1_x * grad_div[0] + row1_y * grad_div[1] + row1_z * grad_div[2],
                    row2_x * grad_div[0] + row2_y * grad_div[1] + row2_z * grad_div[2]};
                const T metric_dot_dK[3] = {row0_x * d_K[0] + row0_y * d_K[1] + row0_z * d_K[2],
                                            row1_x * d_K[0] + row1_y * d_K[1] + row1_z * d_K[2],
                                            row2_x * d_K[0] + row2_y * d_K[1] + row2_z * d_K[2]};
                const T metric_dot_dtheta[3] = {
                    row0_x * d_theta[0] + row0_y * d_theta[1] + row0_z * d_theta[2],
                    row1_x * d_theta[0] + row1_y * d_theta[1] + row1_z * d_theta[2],
                    row2_x * d_theta[0] + row2_y * d_theta[1] + row2_z * d_theta[2]};
                const T metric_dot_dalpha[3] = {
                    row0_x * d_alpha[0] + row0_y * d_alpha[1] + row0_z * d_alpha[2],
                    row1_x * d_alpha[0] + row1_y * d_alpha[1] + row1_z * d_alpha[2],
                    row2_x * d_alpha[0] + row2_y * d_alpha[1] + row2_z * d_alpha[2]};
                const T A_dot_dalpha[3] = {
                    A_up_matrix[0][0] * d_alpha[0] + A_up_matrix[0][1] * d_alpha[1] +
                        A_up_matrix[0][2] * d_alpha[2],
                    A_up_matrix[1][0] * d_alpha[0] + A_up_matrix[1][1] * d_alpha[1] +
                        A_up_matrix[1][2] * d_alpha[2],
                    A_up_matrix[2][0] * d_alpha[0] + A_up_matrix[2][1] * d_alpha[1] +
                        A_up_matrix[2][2] * d_alpha[2]};
                const T A_dot_dchi[3] = {A_up_matrix[0][0] * d_chi[0] + A_up_matrix[0][1] * d_chi[1] +
                                             A_up_matrix[0][2] * d_chi[2],
                                         A_up_matrix[1][0] * d_chi[0] + A_up_matrix[1][1] * d_chi[1] +
                                             A_up_matrix[1][2] * d_chi[2],
                                         A_up_matrix[2][0] * d_chi[0] + A_up_matrix[2][1] * d_chi[1] +
                                             A_up_matrix[2][2] * d_chi[2]};
                T gamma_metric[3] = {T(0), T(0), T(0)};
                detail::metric_inverse_divergence_ptr(p_ginv[0], p_ginv[1], p_ginv[2], p_ginv[3],
                                                      p_ginv[4], p_ginv[5], sx, sy, inv_12dx,
                                                      inv_12dy, inv_12dz, gamma_metric);
                gamma_metric[0] = -gamma_metric[0];
                gamma_metric[1] = -gamma_metric[1];
                gamma_metric[2] = -gamma_metric[2];

                T z_over_chi[3] = {T(0), T(0), T(0)};
                for (int c = 0; c < 3; ++c) {
                    if (gauge_params.evolve_Z) {
                        // Stored Z is contravariant physical Z^i; convert to Z^i/chi here.
                        z_over_chi[c] = *p_Z[c] / chi_guarded;
                    } else {
                        z_over_chi[c] = T(0.5) * (gamma_vec[c] - gamma_metric[c]);
                    }
                }

                T gamma_driver[3] = {T(0), T(0), T(0)};
                for (int c = 0; c < 3; ++c)
                    gamma_driver[c] = gamma_metric[c] + T(2) * gauge_params.kappa3 * z_over_chi[c];

                for (int comp = 0; comp < 3; ++comp) {
                    const T adv =
                        beta_vec[0] * Dx_upwind_ptr(p_Gamma[comp], sx, inv_2dx, beta_vec[0]) +
                        beta_vec[1] * Dy_upwind_ptr(p_Gamma[comp], sy, inv_2dy, beta_vec[1]) +
                        beta_vec[2] * Dz_upwind_ptr(p_Gamma[comp], inv_2dz, beta_vec[2]);

                    T gamma_beta = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        gamma_beta += gamma_driver[axis] * d_beta[comp][axis];

                    const T stretch = two_thirds * gamma_driver[comp] * div_beta;
                    const T grad_div_term = metric_dot_grad_div[comp] * inv_third;
                    const T A_grad_alpha = A_dot_dalpha[comp];

                    T GammaA = T(0);
                    for (int m = 0; m < 3; ++m)
                        for (int n = 0; n < 3; ++n)
                            GammaA += Gamma_tilde_vals[comp][m][n] * A_up_matrix[m][n];

                    const T grad_K_up = metric_dot_dK[comp];
                    const T A_grad_chi_over_chi = A_dot_dchi[comp] / chi_guarded;

                    const T source = T(2) * alpha * (GammaA - two_thirds * grad_K_up);
                    const T grad_theta_up = metric_dot_dtheta[comp];
                    const T grad_alpha_up = metric_dot_dalpha[comp];
                    const T theta_drive = T(2) * alpha * grad_theta_up - T(2) * theta_val * grad_alpha_up;
                    const T chi_drive = -T(3) * alpha * A_grad_chi_over_chi;
                    const T z4_k_drive = -T(4) / T(3) * alpha * K_val * z_over_chi[comp];
                    const T damping = -T(2) * kappa1_lapse * z_over_chi[comp];

                    const T diss = KO6_axis_ptr(p_Gamma[comp], sx) +
                                   KO6_axis_ptr(p_Gamma[comp], sy) + KO6_axis_ptr(p_Gamma[comp], 1);

                    *p_rhs[comp] = adv - gamma_beta + stretch + hess_contr[comp] + grad_div_term -
                                   T(2) * A_grad_alpha + source + theta_drive + chi_drive +
                                   z4_k_drive + damping + ko_scale * diss;
                }

                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_Gamma[c];
                    ++p_Z[c];
                    ++p_rhs[c];
                }
                ++p_alpha;
                ++p_chi;
                ++p_K;
                ++p_theta;
                for (int s = 0; s < 6; ++s) {
                    ++p_ginv[s];
                    ++p_gcov[s];
                    ++p_A[s];
                }
            }
        }
    }
}

} // namespace tensorium_RG::bssn
