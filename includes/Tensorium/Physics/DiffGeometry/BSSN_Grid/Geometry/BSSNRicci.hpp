#pragma once
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "BSSNCHristoffelTilde.hpp"
#include "BSSNInvariants.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

/**
 * @file BSSNRicci.hpp
 * @brief Assembly of the Ricci tensor split \f$R_{ij}=\tilde{R}_{ij}+R^{\chi}_{ij}\f$.
 * @details
 * The implementation follows the standard BSSN decomposition:
 * \f[
 * \begin{aligned}
 * \tilde{R}_{ij} &= -\tfrac{1}{2}\tilde{\gamma}^{mn}\partial_m\partial_n\tilde{\gamma}_{ij}
 *  + \tilde{\gamma}_{k(i}\partial_{j)}\tilde{\Gamma}^k + \tilde{\Gamma}^k\tilde{\Gamma}_{kij}
 *  + \tilde{\gamma}^{mn}\left(2\tilde{\Gamma}^k_{m(i}\tilde{\Gamma}_{j)kn} -
 * \tilde{\Gamma}^k_{ij}\tilde{\Gamma}_{kmn}\right),\\ R^{\chi}_{ij} &=
 * \tfrac{1}{2}\chi^{-1}\left(\tilde{D}_i\tilde{D}_j\chi +
 * \tilde{\gamma}_{ij}\tilde{D}^k\tilde{D}_k\chi\right)
 *   - \tfrac{1}{4}\chi^{-2}\tilde{D}_i\chi\,\tilde{D}_j\chi
 *   - \tfrac{3}{4}\chi^{-2}\tilde{\gamma}_{ij}\tilde{D}^k\chi\,\tilde{D}_k\chi.
 * \end{aligned}
 * \f]
 * `compute_ricci_bssn` evaluates these expressions with the derivative operators documented in
 * `BSSNGridDerivatives.hpp`, caches \f$\tilde{\Gamma}^i_{\ jk}\f$ locally to avoid repeated array
 * lookups, and finally stores the packed symmetric tensor into `Ricci6`.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Populate the Ricci tensor cache \f$R_{ij}\f$ throughout the interior grid.
 * @param G Grid supplying \f$\chi\f$, \f$\tilde{\gamma}_{ij}\f$, \f$\tilde{\gamma}^{ij}\f$, and
 * cached
 *          \f$\tilde{\Gamma}^i\f$.
 * @param[out] Ricci6 Packed symmetric destination fields.
 * @param throw_on_violation If true, `assert_invariants` raises when the projector detects
 *                           determinant or trace drifts beyond tolerance.
 *
 * @details
 * **Mapping to code.**
 * - `dg` lambda → \f$\partial_m\tilde{\gamma}_{ab}\f$ samples.
 * - `Gijk` array → \f$\tilde{\Gamma}^i_{\ jk}\f$ assembled from inverse metric contractions.
 * - `cov_dchi` block → \f$\tilde{D}_i\tilde{D}_j\chi\f$ computed via subtracting connection terms
 * from second derivatives.
 * - `R_chi` / `R_tilde` blocks implement the formulas quoted above and store the sum.
 * - `dGt` corresponds to \f$\partial_i\tilde{\Gamma}^k\f$ entering the \f$\tilde{R}_{ij}\f$
 * expression.
 *
 * @warning Ricci evaluation assumes halos contain valid data for ±3 offsets.  Call
 * `apply_halos_grid` before invoking this routine.
 */
template <typename T>
void compute_ricci_bssn(BSSNGridSoA<T> &G, Field3D<T> *Ricci6, bool throw_on_violation = true) {
    using namespace tensorium_RG::fd;
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 3, i1 = I1 - 3;
    const size_t j0 = J0 + 3, j1 = J1 - 3;
    const size_t k0 = K0 + 3, k1 = K1 - 3;

    const double inv_12dx = 1.0 / (60.0 * G.dx);
    const double inv_12dy = 1.0 / (60.0 * G.dy);
    const double inv_12dz = 1.0 / (60.0 * G.dz);
    const double inv_12dx2 = 1.0 / (180.0 * G.dx * G.dx);
    const double inv_12dy2 = 1.0 / (180.0 * G.dy * G.dy);
    const double inv_12dz2 = 1.0 / (180.0 * G.dz * G.dz);
    const double inv_144dxdy = 1.0 / (144.0 * G.dx * G.dy);
    const double inv_144dxdz = 1.0 / (144.0 * G.dx * G.dz);
    const double inv_144dydz = 1.0 / (144.0 * G.dy * G.dz);

    const ptrdiff_t sx = G.gamma_tilde[0].st.sx;
    const ptrdiff_t sy = G.gamma_tilde[0].st.sy;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {

            size_t idx_start = G.gamma_tilde[XX].idx(i, j, k0);

            const T *p_gam[6];
            const T *p_gam_inv[6];
            const T *p_Gt[3];
            T       *p_Ricci[6];

            for (int s = 0; s < 6; ++s) {
                p_gam[s] = G.gamma_tilde[s].ptr() + idx_start;
                p_gam_inv[s] = G.gamma_tilde_inv[s].ptr() + idx_start;
                p_Ricci[s] = Ricci6[s].ptr() + idx_start;
            }
            for (int c = 0; c < 3; ++c)
                p_Gt[c] = G.tildeGamma[c].ptr() + idx_start;

            const T *p_chi = G.chi.ptr() + idx_start;

            for (size_t k = k0; k < k1; ++k) {

                double g[3][3], gI[3][3];
                g[0][0] = *p_gam[0];
                g[0][1] = *p_gam[1];
                g[0][2] = *p_gam[2];
                g[1][1] = *p_gam[3];
                g[1][2] = *p_gam[4];
                g[2][2] = *p_gam[5];
                g[1][0] = g[0][1];
                g[2][0] = g[0][2];
                g[2][1] = g[1][2];

                gI[0][0] = *p_gam_inv[0];
                gI[0][1] = *p_gam_inv[1];
                gI[0][2] = *p_gam_inv[2];
                gI[1][1] = *p_gam_inv[3];
                gI[1][2] = *p_gam_inv[4];
                gI[2][2] = *p_gam_inv[5];
                gI[1][0] = gI[0][1];
                gI[2][0] = gI[0][2];
                gI[2][1] = gI[1][2];

                double    dg[3][3][3];
                const int map_s[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const T     *p = p_gam[map_s[a][b]];
                        const double dx_g = Dx_ptr(p, sx, inv_12dx);
                        const double dy_g = Dy_ptr(p, sy, inv_12dy);
                        const double dz_g = Dz_ptr(p, inv_12dz);
                        dg[0][a][b] = dx_g;
                        dg[0][b][a] = dx_g;
                        dg[1][a][b] = dy_g;
                        dg[1][b][a] = dy_g;
                        dg[2][a][b] = dz_g;
                        dg[2][b][a] = dz_g;
                    }
                }

                double G_up[3][3][3];
                for (int kk = 0; kk < 3; ++kk) {
                    for (int ii = 0; ii < 3; ++ii) {
                        for (int jj = ii; jj < 3; ++jj) {
                            double sum = 0.0;
                            for (int ll = 0; ll < 3; ++ll) {
                                sum +=
                                    gI[kk][ll] * (dg[ii][jj][ll] + dg[jj][ii][ll] - dg[ll][ii][jj]);
                            }
                            double val = 0.5 * sum;
                            G_up[kk][ii][jj] = val;
                            if (ii != jj)
                                G_up[kk][jj][ii] = val;
                        }
                    }
                }

                double R_chi[3][3];
                {
                    const double chi = *p_chi;
                    const double inv_chi = 1.0 / chi;
                    const double inv_chi2 = inv_chi * inv_chi;

                    const double dchi[3] = {Dx_ptr(p_chi, sx, inv_12dx),
                                            Dy_ptr(p_chi, sy, inv_12dy), Dz_ptr(p_chi, inv_12dz)};

                    double hess_chi[3][3];
                    double lap_chi = 0.0;  
                    double grad_chi_sq = 0.0;

                    const double d2_xx = Dxx_ptr(p_chi, sx, inv_12dx2);
                    const double d2_yy = Dyy_ptr(p_chi, sy, inv_12dy2);
                    const double d2_zz = Dzz_ptr(p_chi, inv_12dz2);
                    const double d2_xy = Dxy4_ptr(p_chi, sx, sy, inv_144dxdy);
                    const double d2_xz = Dxz4_ptr(p_chi, sx, inv_144dxdz);
                    const double d2_yz = Dyz4_ptr(p_chi, sy, inv_144dydz);
                    const double d2chi_raw[3][3] = {
                        {d2_xx, d2_xy, d2_xz}, {d2_xy, d2_yy, d2_yz}, {d2_xz, d2_yz, d2_zz}};

                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            double conn = 0.0;
                            for (int m = 0; m < 3; ++m)
                                conn += G_up[m][a][b] * dchi[m];

                            double h = d2chi_raw[a][b] - conn;
                            hess_chi[a][b] = h;
                            if (a != b)
                                hess_chi[b][a] = h;
                        }
                    }

                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            lap_chi += gI[a][b] * hess_chi[a][b];
                            grad_chi_sq += gI[a][b] * dchi[a] * dchi[b];
                        }
                    }

                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            double val = 0.5 * inv_chi * (hess_chi[a][b] + g[a][b] * lap_chi) -
                                         0.25 * inv_chi2 * dchi[a] * dchi[b] -
                                         0.75 * inv_chi2 * g[a][b] * grad_chi_sq;
                            R_chi[a][b] = val;
                            if (a != b)
                                R_chi[b][a] = val;
                        }
                    }
                }

                double R_tilde[3][3];
                {
                    double dGt[3][3]; // d_j Gamma^i
                    for (int c = 0; c < 3; ++c) {
                        const T *p = p_Gt[c];
                        dGt[c][0] = Dx_ptr(p, sx, inv_12dx);
                        dGt[c][1] = Dy_ptr(p, sy, inv_12dy);
                        dGt[c][2] = Dz_ptr(p, inv_12dz);
                    }
                    const double Gt_vec[3] = {*p_Gt[0], *p_Gt[1], *p_Gt[2]};

                    double G_low[3][3][3];
                    for (int k = 0; k < 3; ++k)
                        for (int i = 0; i < 3; ++i)
                            for (int j = i; j < 3; ++j) {
                                double s = 0.0;
                                for (int m = 0; m < 3; ++m)
                                    s += g[k][m] * G_up[m][i][j];
                                G_low[k][i][j] = s;
                                if (i != j)
                                    G_low[k][j][i] = s;
                            }

                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            const T *p_gab = p_gam[map_s[a][b]];
                            double   lap_g = gI[0][0] * Dxx_ptr(p_gab, sx, inv_12dx2) +
                                           gI[1][1] * Dyy_ptr(p_gab, sy, inv_12dy2) +
                                           gI[2][2] * Dzz_ptr(p_gab, inv_12dz2) +
                                           2.0 * (gI[0][1] * Dxy4_ptr(p_gab, sx, sy, inv_144dxdy) +
                                                  gI[0][2] * Dxz4_ptr(p_gab, sx, inv_144dxdz) +
                                                  gI[1][2] * Dyz4_ptr(p_gab, sy, inv_144dydz));

                            double t2 = 0.0;
                            for (int k = 0; k < 3; ++k)
                                t2 += 0.5 * (g[a][k] * dGt[k][b] + g[b][k] * dGt[k][a]);

                            double t3 = 0.0;
                            for (int k = 0; k < 3; ++k)
                                t3 += 0.5 * Gt_vec[k] * G_low[k][a][b];

                            double t4 = 0.0;
                            for (int l = 0; l < 3; ++l) {
                                for (int m = 0; m < 3; ++m) {
                                    double g_inv_lm = gI[l][m];
                                    double inner = 0.0;
                                    for (int k = 0; k < 3; ++k) {
                                        inner += 2.0 * G_up[k][l][a] * G_low[k][b][m] -
                                                 G_up[k][a][b] * G_low[k][l][m];
                                    }
                                    t4 += g_inv_lm * inner;
                                }
                            }

                            double val = -0.5 * lap_g + t2 + t3 + t4;
                            R_tilde[a][b] = val;
                            if (a != b)
                                R_tilde[b][a] = val;
                        }
                    }
                }

                for (int s = 0; s < 6; ++s) {
                    int a = (s == 0 || s == 1 || s == 2) ? 0 : ((s == 3 || s == 4) ? 1 : 2);
                    int b = (s == 0) ? 0 : ((s == 1 || s == 3) ? 1 : 2);
                    *p_Ricci[s] = (T)(R_tilde[a][b] + R_chi[a][b]);
                }

                for (int s = 0; s < 6; ++s) {
                    ++p_gam[s];
                    ++p_gam_inv[s];
                    ++p_Ricci[s];
                }
                for (int c = 0; c < 3; ++c)
                    ++p_Gt[c];
                ++p_chi;
            }
        }
    }

    tensorium_RG::bssn::assert_invariants(G, "ricci", 4, throw_on_violation);
}

} // namespace tensorium_RG::bssn
