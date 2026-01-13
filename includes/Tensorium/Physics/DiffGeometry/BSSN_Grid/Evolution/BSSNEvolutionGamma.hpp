#pragma once

#include <algorithm>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "BSSNEvolutionCommon.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionGamma.hpp
 * @brief RHS for the contracted connection \f$\tilde{\Gamma}^i\f$.
 * @details Implements the standard BSSN equation with advection, metric-Laplacian of the shift, lapse
 * coupling, and Ricci-driven source terms:
 * \f[
 * \begin{aligned}
 * \partial_t \tilde{\Gamma}^i &= \beta^k\partial_k\tilde{\Gamma}^i - \tilde{\Gamma}^k\partial_k\beta^i
 *   + \tfrac{2}{3}\tilde{\Gamma}^i\partial_k\beta^k
 *   + \tilde{\gamma}^{jk}\partial_j\partial_k\beta^i + \tfrac{1}{3}\tilde{\gamma}^{ij}\partial_j\partial_k\beta^k \\
 *   &\quad - 2\tilde{A}^{ij}\partial_j\alpha + 2\alpha\left(\tilde{\Gamma}^i_{\ jk}\tilde{A}^{jk} - \tfrac{2}{3}\tilde{\gamma}^{ij}\partial_j K\right)
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
 * - `gamma_beta` and `stretch` implement the Lie derivative terms \f$-\tilde{\Gamma}^k\partial_k\beta^i\f$ and
 *   \f$\tfrac{2}{3}\tilde{\Gamma}^i\partial_k\beta^k\f$.
 * - `hess_term` and `grad_div_term` build the shift Laplacian pieces.
 * - `A_grad_alpha` and `source` encode \f$-2\tilde{A}^{ij}\partial_j\alpha\f$ and the Ricci coupling.
 */
template <typename T>
inline void compute_rhs_Gamma(const BSSNGridSoA<T> &G, Field3D<T> rhs[3], size_t padding = 4) {
    using namespace tensorium_RG::fd;
    const T ko_sigma = T(0.1);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total =
        G.tildeGamma[0].st.nx_tot * G.tildeGamma[0].st.ny_tot * G.tildeGamma[0].st.nz_tot;
    for (int c = 0; c < 3; ++c)
        std::fill(rhs[c].ptr(), rhs[c].ptr() + total, T(0));

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
                const size_t id = G.tildeGamma[0].idx(i, j, k);

                T gamma_inv[3][3];
                T A_lower[3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int s = tensorium_RG::sym6_index(a, b);
                        const T   gt_inv = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);
                        const T   Aval = tensorium_RG::sym6_get(G.A_tilde, id, a, b);
                        gamma_inv[a][b] = gt_inv;
                        gamma_inv[b][a] = gt_inv;
                        A_lower[a][b] = Aval;
                        A_lower[b][a] = Aval;
                    }
                }

                T A_up[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n)
                                sum += gamma_inv[a][m] * gamma_inv[b][n] * A_lower[m][n];
                        A_up[a][b] = sum;
                        A_up[b][a] = sum;
                    }

                const T beta_vec[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id],
                                       G.beta[2].ptr()[id]};
                const T gamma_vec[3] = {G.tildeGamma[0].ptr()[id], G.tildeGamma[1].ptr()[id],
                                        G.tildeGamma[2].ptr()[id]};

                const T d_beta[3][3] = {
                    {T(Dx(G.beta[0], i, j, k, G.dx)), T(Dy(G.beta[0], i, j, k, G.dy)),
                     T(Dz(G.beta[0], i, j, k, G.dz))},
                    {T(Dx(G.beta[1], i, j, k, G.dx)), T(Dy(G.beta[1], i, j, k, G.dy)),
                     T(Dz(G.beta[1], i, j, k, G.dz))},
                    {T(Dx(G.beta[2], i, j, k, G.dx)), T(Dy(G.beta[2], i, j, k, G.dy)),
                     T(Dz(G.beta[2], i, j, k, G.dz))}};

                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];

                T    hessians[3][3][3];
                auto fill_hessian = [&](int comp, const Field3D<T> &F) {
                    hessians[comp][0][0] = T(Dxx(F, i, j, k, G.dx));
                    hessians[comp][1][1] = T(Dyy(F, i, j, k, G.dy));
                    hessians[comp][2][2] = T(Dzz(F, i, j, k, G.dz));
                    hessians[comp][0][1] = T(Dxy4(F, i, j, k, G.dx, G.dy));
                    hessians[comp][0][2] = T(Dxz4(F, i, j, k, G.dx, G.dz));
                    hessians[comp][1][2] = T(Dyz4(F, i, j, k, G.dy, G.dz));
                    hessians[comp][1][0] = hessians[comp][0][1];
                    hessians[comp][2][0] = hessians[comp][0][2];
                    hessians[comp][2][1] = hessians[comp][1][2];
                };
                fill_hessian(0, G.beta[0]);
                fill_hessian(1, G.beta[1]);
                fill_hessian(2, G.beta[2]);

                T grad_div[3];
                for (int axis = 0; axis < 3; ++axis) {
                    T sum = T(0);
                    for (int comp = 0; comp < 3; ++comp)
                        sum += hessians[comp][axis][comp];
                    grad_div[axis] = sum;
                }

                auto contract_metric = [&](int comp) {
                    T sum = T(0);
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            sum += gamma_inv[a][b] * hessians[comp][a][b];
                    return sum;
                };

                const T d_alpha[3] = {T(Dx(G.alpha, i, j, k, G.dx)), T(Dy(G.alpha, i, j, k, G.dy)),
                                      T(Dz(G.alpha, i, j, k, G.dz))};
                const T d_K[3] = {T(Dx(G.K, i, j, k, G.dx)), T(Dy(G.K, i, j, k, G.dy)),
                                  T(Dz(G.K, i, j, k, G.dz))};

                T Gamma_conn[3][3][3];
                for (int up = 0; up < 3; ++up)
                    for (int m = 0; m < 3; ++m)
                        for (int n = 0; n < 3; ++n) {
                            const int idx_field = up * 9 + m * 3 + n;
                            Gamma_conn[up][m][n] = G.Gamma_tilde[idx_field].ptr()[id];
                        }

                const T alpha = G.alpha.ptr()[id];

                for (int comp = 0; comp < 3; ++comp) {
                    const T adv = beta_vec[0] * T(Dx_upwind(G.tildeGamma[comp], i, j, k, G.dx, beta_vec[0])) +
                                  beta_vec[1] * T(Dy_upwind(G.tildeGamma[comp], i, j, k, G.dy, beta_vec[1])) +
                                  beta_vec[2] * T(Dz_upwind(G.tildeGamma[comp], i, j, k, G.dz, beta_vec[2]));

                    T gamma_beta = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        gamma_beta += gamma_vec[axis] * d_beta[comp][axis];

                    const T stretch = two_thirds * gamma_vec[comp] * div_beta;
                    const T hess_term = contract_metric(comp);

                    T grad_div_term = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        grad_div_term += gamma_inv[comp][axis] * grad_div[axis];
                    grad_div_term *= third;

                    T A_grad_alpha = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        A_grad_alpha += A_up[comp][axis] * d_alpha[axis];

                    T GammaA = T(0);
                    for (int m = 0; m < 3; ++m)
                        for (int n = 0; n < 3; ++n)
                            GammaA += Gamma_conn[comp][m][n] * A_up[m][n];

                    T grad_K_up = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        grad_K_up += gamma_inv[comp][axis] * d_K[axis];

                    const T source = T(2) * alpha * (GammaA - two_thirds * grad_K_up);

                    rhs[comp].ptr()[id] = adv - gamma_beta + stretch + hess_term + grad_div_term -
                                          T(2) * A_grad_alpha + source +
                                          T(KO6(G.tildeGamma[comp], i, j, k, ko_sigma));
                }
            }
        }
    }
}

} // namespace tensorium_RG::bssn
