#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>

/**
 * @file BSSNEvolutionK.hpp
 * @brief RHS for the mean curvature \f$K\f$.
 * @details Combines advection, the covariant Laplacian of the lapse, and quadratic extrinsic
 * curvature terms:
 * \f[
 * \partial_t K = \beta^i\partial_i K - \gamma^{ij}D_i D_j \alpha +
 * \alpha(\tilde{A}_{ij}\tilde{A}^{ij} + \tfrac{1}{3}K^2).
 * \f]
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
 * @brief Compute \f$\partial_t K\f$ throughout the interior region.
 */
template <typename T>
inline void compute_rhs_K(const BSSNGridSoA<T> &G, Field3D<T> &rhs_K, size_t padding = 4) {
    using namespace tensorium_RG::fd;
    const T ko_sigma = T(0.6);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.K.st.nx_tot * G.K.st.ny_tot * G.K.st.nz_tot;
    std::fill(rhs_K.ptr(), rhs_K.ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    const T third = T(1) / T(3);

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

    const ptrdiff_t sx = G.K.st.sx;
    const ptrdiff_t sy = G.K.st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.K.idx(i, j, k0);

            const T *p_K = G.K.ptr() + idx_start;
            const T *p_alpha = G.alpha.ptr() + idx_start;
            const T *p_chi = G.chi.ptr() + idx_start;

            const T *p_beta[3] = {G.beta[0].ptr() + idx_start, G.beta[1].ptr() + idx_start,
                                  G.beta[2].ptr() + idx_start};

            const T *p_gam[6];
            const T *p_gam_inv[6];
            const T *p_A[6];
            T       *p_rhs = rhs_K.ptr() + idx_start;

            for (int s = 0; s < 6; ++s) {
                p_gam[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_gam_inv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
            }

            for (size_t k = k0; k < k1; ++k) {
                const T alpha = *p_alpha;
                const T chi = *p_chi;
                const T inv_chi = T(1) / chi;
                const T K_val = *p_K;
                const T bx = *p_beta[0];
                const T by = *p_beta[1];
                const T bz = *p_beta[2];

                T gamma_phys_inv[3][3];
                T gamma_phys[3][3];
                T gamma_tilde_inv[3][3];
                T A_mat[3][3];

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

                    const T gt_inv = *p_gam_inv[s];
                    const T gt = *p_gam[s];
                    const T Aval = *p_A[s];

                    gamma_tilde_inv[a][b] = gt_inv;
                    gamma_tilde_inv[b][a] = gt_inv;

                    const T gp = gt * inv_chi;
                    gamma_phys[a][b] = gp;
                    gamma_phys[b][a] = gp;

                    const T gpi = gt_inv * chi;
                    gamma_phys_inv[a][b] = gpi;
                    gamma_phys_inv[b][a] = gpi;

                    A_mat[a][b] = Aval;
                    A_mat[b][a] = Aval;
                }

                const T adv = bx * Dx_upwind_ptr(p_K, sx, inv_2dx, bx) +
                              by * Dy_upwind_ptr(p_K, sy, inv_2dy, by) +
                              bz * Dz_upwind_ptr(p_K, inv_2dz, bz);

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};
                const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                                    Dz_ptr(p_chi, inv_12dz)};

                T d_g_phys[3][3][3]; 

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

                    const T gp = gamma_phys[row][col];
                    const T val_x = (d_gt_x - gp * d_chi[0]) * inv_chi;
                    const T val_y = (d_gt_y - gp * d_chi[1]) * inv_chi;
                    const T val_z = (d_gt_z - gp * d_chi[2]) * inv_chi;

                    d_g_phys[0][row][col] = val_x;
                    d_g_phys[0][col][row] = val_x;
                    d_g_phys[1][row][col] = val_y;
                    d_g_phys[1][col][row] = val_y;
                    d_g_phys[2][row][col] = val_z;
                    d_g_phys[2][col][row] = val_z;
                }

                T Gamma_conn[3][3][3];
                for (int idx_k = 0; idx_k < 3; ++idx_k) {
                    for (int row = 0; row < 3; ++row) {
                        for (int col = row; col < 3; ++col) {
                            T sum = T(0);
                            for (int l = 0; l < 3; ++l) {
                                sum += gamma_phys_inv[idx_k][l] *
                                       (d_g_phys[row][col][l] + d_g_phys[col][row][l] -
                                        d_g_phys[l][row][col]);
                            }
                            const T val = T(0.5) * sum;
                            Gamma_conn[idx_k][row][col] = val;
                            Gamma_conn[idx_k][col][row] = val;
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

                T laplacian = T(0);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        T conn_grad = T(0);
                        for (int l = 0; l < 3; ++l)
                            conn_grad += Gamma_conn[l][row][col] * d_alpha[l];

                        const T cov_d_alpha = hess[row][col] - conn_grad;
                        laplacian += gamma_phys_inv[row][col] * cov_d_alpha;
                    }
                }

                T A_up[3][3];
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            sum += gamma_tilde_inv[row][m] * A_mat[m][col];
                        A_up[row][col] = sum;
                    }

                T A_contract = T(0);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        A_contract += A_mat[row][col] * A_up[row][col];

                for (int row = 0; row < 3; ++row) {
                    for (int col = row; col < 3; ++col) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n)
                                sum +=
                                    gamma_tilde_inv[row][m] * gamma_tilde_inv[col][n] * A_mat[m][n];
                        A_up[row][col] = sum;
                        A_up[col][row] = sum;
                    }
                }

                A_contract = T(0);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        A_contract += A_mat[row][col] * A_up[row][col];

                const T quad = alpha * (A_contract + third * K_val * K_val);

                const T diss = KO6_axis_ptr(p_K, sx) + KO6_axis_ptr(p_K, sy) + KO6_axis_ptr(p_K, 1);
                const T diss_scaled = (ko_sigma / G.dx) * diss;

                *p_rhs = adv - laplacian + quad + diss_scaled;

                ++p_K;
                ++p_alpha;
                ++p_chi;
                ++p_rhs;
                for (int c = 0; c < 3; ++c)
                    ++p_beta[c];
                for (int s = 0; s < 6; ++s) {
                    ++p_gam[s];
                    ++p_gam_inv[s];
                    ++p_A[s];
                }
            }
        }
    }
}

} // namespace tensorium_RG::bssn
