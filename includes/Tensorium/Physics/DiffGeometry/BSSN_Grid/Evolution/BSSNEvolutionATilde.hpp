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
 * (\partial_t-\mathcal{L}_\beta)\tilde{A}_{ij} = \chi\left[-D_iD_j\alpha + \alpha R_{ij}\right]^{TF} + \alpha (K\tilde{A}_{ij} - 2\tilde{A}_{ik}\tilde{A}^k_{\ j})
 * \f]
 * where \f$D_i\f$ is the physical covariant derivative.  The code reconstructs the physical metric
 * \f$\gamma_{ij}=\chi^{-1}\tilde{\gamma}_{ij}\f$ and Christoffels to evaluate the covariant Hessian of \f$\alpha\f$.
 */

namespace tensorium_RG::bssn {

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
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
    const T ko_sigma = T(0.1);

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

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.A_tilde[0].idx(i, j, k);

                const T alpha = G.alpha.ptr()[id];
                const T chi = G.chi.ptr()[id];
                const T inv_chi = T(1) / chi;
                const T K = G.K.ptr()[id];

                T gamma_tilde_inv[3][3];
                T gamma_phys[3][3];
                T gamma_phys_inv[3][3];
                T A_mat[3][3];
                T Ricci_mat[3][3];

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int s = tensorium_RG::sym6_index(a, b);
                        const T   gt = tensorium_RG::sym6_get(G.gamma_tilde, id, a, b);
                        const T   gt_inv = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);
                        const T   Aval = tensorium_RG::sym6_get(G.A_tilde, id, a, b);
                        const T   Ricci_val = tensorium_RG::sym6_get(G.Ricci, id, a, b);

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

                        Ricci_mat[a][b] = Ricci_val;
                        Ricci_mat[b][a] = Ricci_val;
                    }
                }

                const T d_alpha[3] = {T(Dx(G.alpha, i, j, k, G.dx)), T(Dy(G.alpha, i, j, k, G.dy)),
                                      T(Dz(G.alpha, i, j, k, G.dz))};

                const T d_beta[3][3] = {
                    {T(Dx(G.beta[0], i, j, k, G.dx)), T(Dy(G.beta[0], i, j, k, G.dy)),
                     T(Dz(G.beta[0], i, j, k, G.dz))},
                    {T(Dx(G.beta[1], i, j, k, G.dx)), T(Dy(G.beta[1], i, j, k, G.dy)),
                     T(Dz(G.beta[1], i, j, k, G.dz))},
                    {T(Dx(G.beta[2], i, j, k, G.dx)), T(Dy(G.beta[2], i, j, k, G.dy)),
                     T(Dz(G.beta[2], i, j, k, G.dz))}};

                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                const T beta[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id], G.beta[2].ptr()[id]};

                const T d_chi[3] = {T(Dx(G.chi, i, j, k, G.dx)), T(Dy(G.chi, i, j, k, G.dy)),
                                    T(Dz(G.chi, i, j, k, G.dz))};

                T d_g_phys[3][3][3];
                for (int dir = 0; dir < 3; ++dir)
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            d_g_phys[dir][row][col] = T(0);

                for (int dir = 0; dir < 3; ++dir) {
                    for (int row = 0; row < 3; ++row) {
                        for (int col = row; col < 3; ++col) {
                            const int s = tensorium_RG::sym6_index(row, col);
                            T         d_gt = T(0);
                            if (dir == 0)
                                d_gt = T(Dx(G.gamma_tilde[s], i, j, k, G.dx));
                            else if (dir == 1)
                                d_gt = T(Dy(G.gamma_tilde[s], i, j, k, G.dy));
                            else
                                d_gt = T(Dz(G.gamma_tilde[s], i, j, k, G.dz));

                            const T val =
                                d_gt * inv_chi - gamma_phys[row][col] * inv_chi * d_chi[dir];
                            d_g_phys[dir][row][col] = val;
                            d_g_phys[dir][col][row] = val;
                        }
                    }
                }

                T Gamma_conn[3][3][3];
                for (int kidx = 0; kidx < 3; ++kidx)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            Gamma_conn[kidx][a][b] = T(0);

                for (int kidx = 0; kidx < 3; ++kidx) {
                    for (int row = 0; row < 3; ++row) {
                        for (int col = row; col < 3; ++col) {
                            T sum = T(0);
                            for (int l = 0; l < 3; ++l) {
                                const T term = d_g_phys[row][col][l] + d_g_phys[col][row][l] -
                                               d_g_phys[l][row][col];
                                sum += gamma_phys_inv[kidx][l] * term;
                            }
                            const T val = T(0.5) * sum;
                            Gamma_conn[kidx][row][col] = val;
                            Gamma_conn[kidx][col][row] = val;
                        }
                    }
                }

                T hess[3][3];
                hess[0][0] = T(Dxx(G.alpha, i, j, k, G.dx));
                hess[1][1] = T(Dyy(G.alpha, i, j, k, G.dy));
                hess[2][2] = T(Dzz(G.alpha, i, j, k, G.dz));
                hess[0][1] = T(Dxy4(G.alpha, i, j, k, G.dx, G.dy));
                hess[0][2] = T(Dxz4(G.alpha, i, j, k, G.dx, G.dz));
                hess[1][2] = T(Dyz4(G.alpha, i, j, k, G.dy, G.dz));
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
                        const int s = tensorium_RG::sym6_index(a, b);

                        const T adv = beta[0] * T(Dx_upwind(G.A_tilde[s], i, j, k, G.dx, beta[0])) +
                                      beta[1] * T(Dy_upwind(G.A_tilde[s], i, j, k, G.dy, beta[1])) +
                                      beta[2] * T(Dz_upwind(G.A_tilde[s], i, j, k, G.dz, beta[2]));

                        T lie = T(0);
                        for (int m = 0; m < 3; ++m) {
                            lie += A_mat[a][m] * d_beta[m][b];
                            lie += A_mat[b][m] * d_beta[m][a];
                        }
                        lie -= two_thirds * A_mat[a][b] * div_beta;

                        const T term_geom = chi * S_tf[a][b];
                        const T term_quad = alpha * (K * A_mat[a][b] - T(2) * A_contracted[a][b]);

                        rhs[s].ptr()[id] = adv + lie + term_geom + term_quad +
                                           T(KO6(G.A_tilde[s], i, j, k, ko_sigma));
                    }
                }
            }
        }
    }
}

} // namespace tensorium_RG::bssn
