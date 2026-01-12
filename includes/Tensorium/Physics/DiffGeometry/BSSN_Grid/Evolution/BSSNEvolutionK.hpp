#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>

namespace tensorium_RG::bssn {

inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}

template <typename T>
inline void compute_rhs_K(const BSSNGridSoA<T> &G, Field3D<T> &rhs_K, size_t padding = 4) {
    using namespace tensorium_RG::fd;

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

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.K.idx(i, j, k);

                const T alpha = G.alpha.ptr()[id];
                const T chi = G.chi.ptr()[id];
                const T inv_chi = T(1) / chi;
                const T K_val = G.K.ptr()[id];

                T gamma_tilde_inv[3][3];
                T gamma_phys[3][3];
                T gamma_phys_inv[3][3];
                T A_mat[3][3];

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int s = tensorium_RG::sym6_index(a, b);
                        const T gt = tensorium_RG::sym6_get(G.gamma_tilde, id, a, b);
                        const T gt_inv = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);
                        const T Aval = tensorium_RG::sym6_get(G.A_tilde, id, a, b);

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
                    }
                }

                const T beta_x = G.beta[0].ptr()[id];
                const T beta_y = G.beta[1].ptr()[id];
                const T beta_z = G.beta[2].ptr()[id];

                const T adv = beta_x * T(Dx(G.K, i, j, k, G.dx)) +
                              beta_y * T(Dy(G.K, i, j, k, G.dy)) +
                              beta_z * T(Dz(G.K, i, j, k, G.dz));

                const T d_alpha[3] = {T(Dx(G.alpha, i, j, k, G.dx)), T(Dy(G.alpha, i, j, k, G.dy)),
                                      T(Dz(G.alpha, i, j, k, G.dz))};
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
                            T        d_gt = T(0);
                            if (dir == 0)
                                d_gt = T(Dx(G.gamma_tilde[s], i, j, k, G.dx));
                            else if (dir == 1)
                                d_gt = T(Dy(G.gamma_tilde[s], i, j, k, G.dy));
                            else
                                d_gt = T(Dz(G.gamma_tilde[s], i, j, k, G.dz));

                            const T val = d_gt * inv_chi - gamma_phys[row][col] * inv_chi * d_chi[dir];
                            d_g_phys[dir][row][col] = val;
                            d_g_phys[dir][col][row] = val;
                        }
                    }
                }

                T Gamma_conn[3][3][3];
                for (int idx_k = 0; idx_k < 3; ++idx_k)
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            Gamma_conn[idx_k][row][col] = T(0);

                for (int idx_k = 0; idx_k < 3; ++idx_k) {
                    for (int row = 0; row < 3; ++row) {
                        for (int col = row; col < 3; ++col) {
                            T sum = T(0);
                            for (int l = 0; l < 3; ++l) {
                                const T term = d_g_phys[row][col][l] + d_g_phys[col][row][l] -
                                               d_g_phys[l][row][col];
                                sum += gamma_phys_inv[idx_k][l] * term;
                            }
                            const T val = T(0.5) * sum;
                            Gamma_conn[idx_k][row][col] = val;
                            Gamma_conn[idx_k][col][row] = val;
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
                for (int row = 0; row < 3; ++row) {
                    for (int col = row; col < 3; ++col) {
                        T conn_grad = T(0);
                        for (int l = 0; l < 3; ++l)
                            conn_grad += Gamma_conn[l][row][col] * d_alpha[l];
                        const T val = hess[row][col] - conn_grad;
                        cov_dd[row][col] = val;
                        cov_dd[col][row] = val;
                    }
                }

                T laplacian = T(0);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        laplacian += gamma_phys_inv[row][col] * cov_dd[row][col];

                T A_up[3][3];
                for (int row = 0; row < 3; ++row)
                    for (int col = row; col < 3; ++col) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n)
                                sum += gamma_tilde_inv[row][m] * gamma_tilde_inv[col][n] * A_mat[m][n];
                        A_up[row][col] = sum;
                        A_up[col][row] = sum;
                    }

                T A_contract = T(0);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        A_contract += A_mat[row][col] * A_up[row][col];

                const T quad = alpha * (A_contract + third * K_val * K_val);

                rhs_K.ptr()[id] = adv - laplacian + quad;
            }
        }
    }
}

} // namespace tensorium_RG::bssn
