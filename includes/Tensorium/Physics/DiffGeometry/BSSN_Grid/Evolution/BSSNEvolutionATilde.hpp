#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>

/**
 * @file BSSNEvolutionATilde.hpp
 * @brief RHS for the trace-free extrinsic curvature \f$\tilde{A}_{ij}\f$.
 * @details Implements the trace-free part of
 * \f[
 * (\partial_t-\mathcal{L}_\beta)\tilde{A}_{ij} = \chi\left[-D_iD_j\alpha + \alpha
 * R_{ij}\right]^{TF} + \alpha (K\tilde{A}_{ij} - 2\tilde{A}_{ik}\tilde{A}^k_{\ j})
 * \f]
 * where \f$D_i\f$ is the physical covariant derivative.  The code reconstructs the physical metric
 * \f$\gamma_{ij}=\chi^{-1}\tilde{\gamma}_{ij}\f$ and Christoffels to evaluate the covariant Hessian
 * of \f$\alpha\f$.
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

/**
 * @brief Assemble \f$\partial_t\tilde{A}_{ij}\f$ for each symmetric component.
 * @details
 * - `lie` implements the \f$\mathcal{L}_\beta\tilde{A}_{ij}\f$ term.
 * - `term_geom` corresponds to \f$\chi[-D_iD_j\alpha + \alpha R_{ij}]^{TF}\f$.
 * - `term_quad` implements \f$\alpha(K\tilde{A}_{ij} - 2\tilde{A}_{ik}\tilde{A}^k_{\ j})\f$.
 */
template <typename T>
inline void compute_rhs_A_tilde(const BSSNGridSoA<T> &G, Field3D<T> rhs[6], size_t padding = 4) {
    using namespace tensorium_RG::fd;
    const T ko_sigma = T(0.6);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.A_tilde[0].st.nx_tot * G.A_tilde[0].st.ny_tot * G.A_tilde[0].st.nz_tot;
    for (int s = 0; s < 6; ++s)
        std::fill(rhs[s].ptr(), rhs[s].ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    const T third = T(1) / T(3);
    const T two_thirds = T(2) * third;

    // Pre-compute optimization constants
    const double inv_12dx = 1.0 / (12.0 * G.dx);
    const double inv_12dy = 1.0 / (12.0 * G.dy);
    const double inv_12dz = 1.0 / (12.0 * G.dz);
    const double inv_2dx = 1.0 / (2.0 * G.dx);
    const double inv_2dy = 1.0 / (2.0 * G.dy);
    const double inv_2dz = 1.0 / (2.0 * G.dz);

    const double inv_12dx2 = 1.0 / (12.0 * G.dx * G.dx);
    const double inv_12dy2 = 1.0 / (12.0 * G.dy * G.dy);
    const double inv_12dz2 = 1.0 / (12.0 * G.dz * G.dz);
    const double inv_144dxdy = 1.0 / (144.0 * G.dx * G.dy);
    const double inv_144dxdz = 1.0 / (144.0 * G.dx * G.dz);
    const double inv_144dydz = 1.0 / (144.0 * G.dy * G.dz);

    const ptrdiff_t sx = G.A_tilde[0].st.sx;
    const ptrdiff_t sy = G.A_tilde[0].st.sy;

    // Map (row, col) -> symmetric index 0..5
    const int map_s[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.A_tilde[0].idx(i, j, k0);

            // Pointers setup
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;
            const T *p_K = G.K.ptr() + idx_start;

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};

            const T *p_gam[6];
            const T *p_gam_inv[6];
            const T *p_A[6];
            const T *p_Ricci[6];
            T       *p_rhs[6];

            for (int s = 0; s < 6; ++s) {
                p_gam[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_gam_inv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
                p_Ricci[s] = G.Ricci[s].ptr() + idx_start;
                p_rhs[s] = rhs[s].ptr() + idx_start;
            }

            for (size_t k = k0; k < k1; ++k) {
                const T alpha = *p_alpha;
                const T chi = *p_chi;
                const T inv_chi = T(1) / chi;
                const T K = *p_K;

                // --- 1. Load Local Tensors ---
                T gamma_tilde_inv[3][3];
                T gamma_phys[3][3];
                T gamma_phys_inv[3][3];
                T A_mat[3][3];
                T Ricci_mat[3][3];

                for (int s = 0; s < 6; ++s) {
                    int a, b;
                    if (s == 0) {
                        a = 0;
                        b = 0;
                    } else if (s == 1) {
                        a = 0;
                        b = 1;
                    } else if (s == 2) {
                        a = 0;
                        b = 2;
                    } else if (s == 3) {
                        a = 1;
                        b = 1;
                    } else if (s == 4) {
                        a = 1;
                        b = 2;
                    } else {
                        a = 2;
                        b = 2;
                    }

                    const T gt = *p_gam[s];
                    const T gt_inv = *p_gam_inv[s];
                    const T Aval = *p_A[s];
                    const T Rval = *p_Ricci[s];

                    gamma_tilde_inv[a][b] = gt_inv;
                    gamma_tilde_inv[b][a] = gt_inv;

                    const T g_phys = gt * inv_chi;
                    const T g_phys_inv = gt_inv * chi;

                    gamma_phys[a][b] = g_phys;
                    gamma_phys[b][a] = g_phys;
                    gamma_phys_inv[a][b] = g_phys_inv;
                    gamma_phys_inv[b][a] = g_phys_inv;

                    A_mat[a][b] = Aval;
                    A_mat[b][a] = Aval;
                    Ricci_mat[a][b] = Rval;
                    Ricci_mat[b][a] = Rval;
                }

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};

                T d_beta[3][3];
                for (int c = 0; c < 3; ++c) {
                    d_beta[c][0] = Dx_ptr(p_beta[c], sx, inv_12dx);
                    d_beta[c][1] = Dy_ptr(p_beta[c], sy, inv_12dy);
                    d_beta[c][2] = Dz_ptr(p_beta[c], inv_12dz);
                }
                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                const T beta[3] = {*p_beta[0], *p_beta[1], *p_beta[2]};

                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};

                T d_g_phys[3][3][3]; // dir, row, col

                for (int s = 0; s < 6; ++s) {
                    int row, col;
                    if (s == 0) {
                        row = 0;
                        col = 0;
                    } else if (s == 1) {
                        row = 0;
                        col = 1;
                    } else if (s == 2) {
                        row = 0;
                        col = 2;
                    } else if (s == 3) {
                        row = 1;
                        col = 1;
                    } else if (s == 4) {
                        row = 1;
                        col = 2;
                    } else {
                        row = 2;
                        col = 2;
                    }

                    const T *p_g = p_gam[s];
                    const T  d_gt_x = Dx_ptr(p_g, sx, inv_12dx);
                    const T  d_gt_y = Dy_ptr(p_g, sy, inv_12dy);
                    const T  d_gt_z = Dz_ptr(p_g, inv_12dz);

                    const T g_p = gamma_phys[row][col];

                    const T val_x = (d_gt_x - g_p * d_chi[0]) * inv_chi;
                    const T val_y = (d_gt_y - g_p * d_chi[1]) * inv_chi;
                    const T val_z = (d_gt_z - g_p * d_chi[2]) * inv_chi;

                    d_g_phys[0][row][col] = val_x;
                    d_g_phys[0][col][row] = val_x;
                    d_g_phys[1][row][col] = val_y;
                    d_g_phys[1][col][row] = val_y;
                    d_g_phys[2][row][col] = val_z;
                    d_g_phys[2][col][row] = val_z;
                }

                T Gamma_conn[3][3][3];
                for (int kidx = 0; kidx < 3; ++kidx) {
                    for (int row = 0; row < 3; ++row) {
                        for (int col = row; col < 3; ++col) {
                            T sum = T(0);
                            for (int l = 0; l < 3; ++l) {
                                sum += gamma_phys_inv[kidx][l] *
                                       (d_g_phys[row][col][l] + d_g_phys[col][row][l] -
                                        d_g_phys[l][row][col]);
                            }
                            const T val = T(0.5) * sum;
                            Gamma_conn[kidx][row][col] = val;
                            Gamma_conn[kidx][col][row] = val;
                        }
                    }
                }

                T hess[3][3];
                hess[0][0] = Dxx_ptr(p_alpha, sx, inv_12dx2);
                hess[1][1] = Dyy_ptr(p_alpha, sy, inv_12dy2);
                hess[2][2] = Dzz_ptr(p_alpha, inv_12dz2);
                hess[0][1] = Dxy4_ptr(p_alpha, sx, sy, inv_144dxdy);
                hess[0][2] = Dxz4_ptr(p_alpha, sx, inv_144dxdz);
                hess[1][2] = Dyz4_ptr(p_alpha, sy, inv_144dydz);
                hess[1][0] = hess[0][1];
                hess[2][0] = hess[0][2];
                hess[2][1] = hess[1][2];

                T cov_dd[3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        T conn_grad = T(0);
                        for (int l = 0; l < 3; ++l)
                            conn_grad += Gamma_conn[l][a][b] * d_alpha[l];

                        const T val = hess[a][b] - conn_grad;
                        cov_dd[a][b] = val;
                        cov_dd[b][a] = val;
                    }
                }

                T S_mat[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        S_mat[a][b] = -cov_dd[a][b] + alpha * Ricci_mat[a][b];

                T trace_S = T(0);
                for (int m = 0; m < 3; ++m)
                    for (int n = 0; n < 3; ++n)
                        trace_S += gamma_phys_inv[m][n] * S_mat[m][n];

                T S_tf[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        S_tf[a][b] = S_mat[a][b] - third * gamma_phys[a][b] * trace_S;

                T A_up[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            sum += gamma_tilde_inv[a][m] * A_mat[m][b];
                        A_up[a][b] = sum;
                    }

                T A_contracted[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            sum += A_mat[a][m] * A_up[m][b];
                        A_contracted[a][b] = sum;
                    }

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int s = map_s[a][b];
                        const T  *p_field = p_A[s];

                        const T bx = beta[0];
                        const T by = beta[1];
                        const T bz = beta[2];

                        const T adv = bx * Dx_upwind_ptr(p_field, sx, inv_2dx, bx) +
                                      by * Dy_upwind_ptr(p_field, sy, inv_2dy, by) +
                                      bz * Dz_upwind_ptr(p_field, inv_2dz, bz);

                        T lie = T(0);
                        for (int m = 0; m < 3; ++m) {
                            lie += A_mat[a][m] * d_beta[m][b];
                            lie += A_mat[b][m] * d_beta[m][a];
                        }
                        lie -= two_thirds * A_mat[a][b] * div_beta;

                        const T term_geom = chi * S_tf[a][b];
                        const T term_quad = alpha * (K * A_mat[a][b] - T(2) * A_contracted[a][b]);

                        const T diss = KO6_axis_ptr(p_field, sx) + KO6_axis_ptr(p_field, sy) +
                                       KO6_axis_ptr(p_field, 1);
                        const T diss_scaled = (ko_sigma / G.dx) * diss;

                        *p_rhs[s] = adv + lie + term_geom + term_quad + diss_scaled;
                    }
                }

                ++p_alpha;
                ++p_chi;
                ++p_K;
                for (int c = 0; c < 3; ++c)
                    ++p_beta[c];
                for (int s = 0; s < 6; ++s) {
                    ++p_gam[s];
                    ++p_gam_inv[s];
                    ++p_A[s];
                    ++p_Ricci[s];
                    ++p_rhs[s];
                }
            }
        }
    }
}
} // namespace tensorium_RG::bssn
