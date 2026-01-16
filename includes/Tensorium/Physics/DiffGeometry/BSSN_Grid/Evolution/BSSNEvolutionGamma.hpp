#pragma once

#include <algorithm>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "BSSNEvolutionCommon.hpp"
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
inline void compute_rhs_Gamma(const BSSNGridSoA<T> &G, Field3D<T> rhs[3], size_t padding = 4) {
    using namespace tensorium_RG::fd;
    const T ko_sigma = T(0.6);

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

    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

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

            const T *p_ginv[6];
            const T *p_A[6];
            for (int s = 0; s < 6; ++s) {
                p_ginv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_A[s] = G.A_tilde[s].ptr() + idx_start;
            }

            const T *p_Gtilde[27];
            for (int q = 0; q < 27; ++q)
                p_Gtilde[q] = G.Gamma_tilde[q].ptr() + idx_start;

            T *p_rhs[3] = {rhs[0].ptr() + idx_start, rhs[1].ptr() + idx_start,
                           rhs[2].ptr() + idx_start};

            for (size_t k = k0; k < k1; ++k) {

                T gI[3][3];
                T A_lo[3][3];

                gI[0][0] = *p_ginv[0];
                gI[0][1] = *p_ginv[1];
                gI[0][2] = *p_ginv[2];
                gI[1][0] = gI[0][1];
                gI[1][1] = *p_ginv[3];
                gI[1][2] = *p_ginv[4];
                gI[2][0] = gI[0][2];
                gI[2][1] = gI[1][2];
                gI[2][2] = *p_ginv[5];

                A_lo[0][0] = *p_A[0];
                A_lo[0][1] = *p_A[1];
                A_lo[0][2] = *p_A[2];
                A_lo[1][0] = A_lo[0][1];
                A_lo[1][1] = *p_A[3];
                A_lo[1][2] = *p_A[4];
                A_lo[2][0] = A_lo[0][2];
                A_lo[2][1] = A_lo[1][2];
                A_lo[2][2] = *p_A[5];

                T A_up[3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        T sum = T(0);
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n)
                                sum += gI[a][m] * gI[b][n] * A_lo[m][n];
                        A_up[a][b] = sum;
                        if (a != b)
                            A_up[b][a] = sum;
                    }
                }

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

                    hess_contr[comp] = gI[0][0] * dxx + gI[1][1] * dyy + gI[2][2] * dzz +
                                       2.0 * (gI[0][1] * dxy + gI[0][2] * dxz + gI[1][2] * dyz);
                }

                const T d_alpha[3] = {Dx_ptr(p_alpha, sx, inv_12dx), Dy_ptr(p_alpha, sy, inv_12dy),
                                      Dz_ptr(p_alpha, inv_12dz)};
                const T d_K[3] = {Dx_ptr(p_K, sx, inv_12dx), Dy_ptr(p_K, sy, inv_12dy),
                                  Dz_ptr(p_K, inv_12dz)};
                const T alpha = *p_alpha;

                for (int comp = 0; comp < 3; ++comp) {
                    const T adv =
                        beta_vec[0] * Dx_upwind_ptr(p_Gamma[comp], sx, inv_2dx, beta_vec[0]) +
                        beta_vec[1] * Dy_upwind_ptr(p_Gamma[comp], sy, inv_2dy, beta_vec[1]) +
                        beta_vec[2] * Dz_upwind_ptr(p_Gamma[comp], inv_2dz, beta_vec[2]);

                    T gamma_beta = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        gamma_beta += gamma_vec[axis] * d_beta[comp][axis];

                    const T stretch = (T(2) / T(3)) * gamma_vec[comp] * div_beta;

                    T grad_div_term = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        grad_div_term += gI[comp][axis] * grad_div[axis];
                    grad_div_term *= (T(1) / T(3));

                    T A_grad_alpha = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        A_grad_alpha += A_up[comp][axis] * d_alpha[axis];

                    T         GammaA = T(0);
                    const int base_g = comp * 9;
                    for (int m = 0; m < 3; ++m)
                        for (int n = 0; n < 3; ++n) {
                            GammaA += (*p_Gtilde[base_g + m * 3 + n]) * A_up[m][n];
                        }

                    T grad_K_up = T(0);
                    for (int axis = 0; axis < 3; ++axis)
                        grad_K_up += gI[comp][axis] * d_K[axis];

                    const T source = T(2) * alpha * (GammaA - (T(2) / T(3)) * grad_K_up);

                    const T diss = KO6_axis_ptr(p_Gamma[comp], sx) +
                                   KO6_axis_ptr(p_Gamma[comp], sy) + KO6_axis_ptr(p_Gamma[comp], 1);

                    *p_rhs[comp] = adv - gamma_beta + stretch + hess_contr[comp] + grad_div_term -
                                   T(2) * A_grad_alpha + source + (ko_sigma / G.dx) * diss;
                }

                for (int c = 0; c < 3; ++c) {
                    ++p_beta[c];
                    ++p_Gamma[c];
                    ++p_rhs[c];
                }
                ++p_alpha;
                ++p_K;
                for (int s = 0; s < 6; ++s) {
                    ++p_ginv[s];
                    ++p_A[s];
                }
                for (int q = 0; q < 27; ++q)
                    ++p_Gtilde[q];
            }
        }
    }
}

} // namespace tensorium_RG::bssn
