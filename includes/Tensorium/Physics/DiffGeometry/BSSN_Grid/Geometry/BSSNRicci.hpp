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
 *  + \tilde{\gamma}^{mn}\left(2\tilde{\Gamma}^k_{m(i}\tilde{\Gamma}_{j)kn} - \tilde{\Gamma}^k_{ij}\tilde{\Gamma}_{kmn}\right),\\
 * R^{\chi}_{ij} &= \tfrac{1}{2}\chi^{-1}\left(\tilde{D}_i\tilde{D}_j\chi + \tilde{\gamma}_{ij}\tilde{D}^k\tilde{D}_k\chi\right)
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
 * @param G Grid supplying \f$\chi\f$, \f$\tilde{\gamma}_{ij}\f$, \f$\tilde{\gamma}^{ij}\f$, and cached
 *          \f$\tilde{\Gamma}^i\f$.
 * @param[out] Ricci6 Packed symmetric destination fields.
 * @param throw_on_violation If true, `assert_invariants` raises when the projector detects
 *                           determinant or trace drifts beyond tolerance.
 *
 * @details
 * **Mapping to code.**
 * - `dg` lambda → \f$\partial_m\tilde{\gamma}_{ab}\f$ samples.
 * - `Gijk` array → \f$\tilde{\Gamma}^i_{\ jk}\f$ assembled from inverse metric contractions.
 * - `cov_dchi` block → \f$\tilde{D}_i\tilde{D}_j\chi\f$ computed via subtracting connection terms from
 *   second derivatives.
 * - `R_chi` / `R_tilde` blocks implement the formulas quoted above and store the sum.
 * - `dGt` corresponds to \f$\partial_i\tilde{\Gamma}^k\f$ entering the \f$\tilde{R}_{ij}\f$ expression.
 *
 * @warning Ricci evaluation assumes halos contain valid data for ±2 offsets.  Call
 * `apply_halos_grid` before invoking this routine.
 */
template <typename T>
void compute_ricci_bssn(BSSNGridSoA<T> &G, Field3D<T> *Ricci6, bool throw_on_violation = true) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 2, i1 = I1 - 2;
    const size_t j0 = J0 + 2, j1 = J1 - 2;
    const size_t k0 = K0 + 2, k1 = K1 - 2;

    const double dx = G.dx;
    const double dy = G.dy;
    const double dz = G.dz;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {

                const size_t id = G.gamma_tilde[XX].idx(i, j, k);

                double g[3][3], gI[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        g[a][b] = (double)tensorium_RG::sym6_get(G.gamma_tilde, id, a, b);
                        gI[a][b] = (double)tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);
                    }

                double dg[3][3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int         s = tensorium_RG::sym6_index(a, b);
                        const Field3D<T> &F = G.gamma_tilde[s];

                        const double gx = Dx(F, i, j, k, dx);
                        const double gy = Dy(F, i, j, k, dy);
                        const double gz = Dz(F, i, j, k, dz);

                        dg[0][a][b] = gx;
                        dg[0][b][a] = gx;
                        dg[1][a][b] = gy;
                        dg[1][b][a] = gy;
                        dg[2][a][b] = gz;
                        dg[2][b][a] = gz;
                    }
                }

                double Gijk[3][3][3];
                for (int k1 = 0; k1 < 3; ++k1)
                    for (int i1 = 0; i1 < 3; ++i1)
                        for (int j1 = 0; j1 < 3; ++j1)
                            Gijk[k1][i1][j1] = 0.0;

                for (int kk = 0; kk < 3; ++kk) {
                    for (int ii = 0; ii < 3; ++ii) {
                        for (int jj = ii; jj < 3; ++jj) {
                            double sum = 0.0;
                            for (int ll = 0; ll < 3; ++ll) {
                                const double term =
                                    dg[ii][jj][ll] + dg[jj][ii][ll] - dg[ll][ii][jj];
                                sum += gI[kk][ll] * term;
                            }
                            const double val = 0.5 * sum;
                            Gijk[kk][ii][jj] = val;
                            if (ii != jj)
                                Gijk[kk][jj][ii] = val;
                        }
                    }
                }

                const double chi = (double)G.chi.ptr()[id];
                const double inv_chi = 1.0 / chi;
                const double inv_chi2 = inv_chi * inv_chi;

                const double dchi_x = Dx(G.chi, i, j, k, dx);
                const double dchi_y = Dy(G.chi, i, j, k, dy);
                const double dchi_z = Dz(G.chi, i, j, k, dz);

                const double d2chi_xx = Dxx(G.chi, i, j, k, dx);
                const double d2chi_yy = Dyy(G.chi, i, j, k, dy);
                const double d2chi_zz = Dzz(G.chi, i, j, k, dz);
                const double d2chi_xy = Dxy4(G.chi, i, j, k, dx, dy);
                const double d2chi_xz = Dxz4(G.chi, i, j, k, dx, dz);
                const double d2chi_yz = Dyz4(G.chi, i, j, k, dy, dz);

                double cov_dchi[3][3];

                cov_dchi[0][0] = d2chi_xx - (Gijk[0][0][0] * dchi_x + Gijk[1][0][0] * dchi_y +
                                             Gijk[2][0][0] * dchi_z);
                cov_dchi[0][1] = d2chi_xy - (Gijk[0][0][1] * dchi_x + Gijk[1][0][1] * dchi_y +
                                             Gijk[2][0][1] * dchi_z);
                cov_dchi[0][2] = d2chi_xz - (Gijk[0][0][2] * dchi_x + Gijk[1][0][2] * dchi_y +
                                             Gijk[2][0][2] * dchi_z);

                cov_dchi[1][0] = cov_dchi[0][1];
                cov_dchi[1][1] = d2chi_yy - (Gijk[0][1][1] * dchi_x + Gijk[1][1][1] * dchi_y +
                                             Gijk[2][1][1] * dchi_z);
                cov_dchi[1][2] = d2chi_yz - (Gijk[0][1][2] * dchi_x + Gijk[1][1][2] * dchi_y +
                                             Gijk[2][1][2] * dchi_z);

                cov_dchi[2][0] = cov_dchi[0][2];
                cov_dchi[2][1] = cov_dchi[1][2];
                cov_dchi[2][2] = d2chi_zz - (Gijk[0][2][2] * dchi_x + Gijk[1][2][2] * dchi_y +
                                             Gijk[2][2][2] * dchi_z);

                const double grad2 = gI[0][0] * dchi_x * dchi_x + gI[1][1] * dchi_y * dchi_y +
                                     gI[2][2] * dchi_z * dchi_z + 2.0 * gI[0][1] * dchi_x * dchi_y +
                                     2.0 * gI[0][2] * dchi_x * dchi_z +
                                     2.0 * gI[1][2] * dchi_y * dchi_z;

                const double lap = gI[0][0] * cov_dchi[0][0] + gI[1][1] * cov_dchi[1][1] +
                                   gI[2][2] * cov_dchi[2][2] + 2.0 * gI[0][1] * cov_dchi[0][1] +
                                   2.0 * gI[0][2] * cov_dchi[0][2] +
                                   2.0 * gI[1][2] * cov_dchi[1][2];

                double       R_chi[3][3];
                const double dchi[3] = {dchi_x, dchi_y, dchi_z};

                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b) {
                        const double term1 = 0.5 * inv_chi * (cov_dchi[a][b] + g[a][b] * lap);
                        const double term2 = -0.25 * inv_chi2 * (dchi[a] * dchi[b]);
                        const double term3 = -0.75 * inv_chi2 * g[a][b] * grad2;
                        const double val = term1 + term2 + term3;
                        R_chi[a][b] = val;
                        if (a != b)
                            R_chi[b][a] = val;
                    }

                const double Gt[3] = {(double)G.tildeGamma[0].ptr()[id],
                                      (double)G.tildeGamma[1].ptr()[id],
                                      (double)G.tildeGamma[2].ptr()[id]};

                double dGt[3][3];
                dGt[0][0] = Dx(G.tildeGamma[0], i, j, k, dx);
                dGt[0][1] = Dy(G.tildeGamma[0], i, j, k, dy);
                dGt[0][2] = Dz(G.tildeGamma[0], i, j, k, dz);
                dGt[1][0] = Dx(G.tildeGamma[1], i, j, k, dx);
                dGt[1][1] = Dy(G.tildeGamma[1], i, j, k, dy);
                dGt[1][2] = Dz(G.tildeGamma[1], i, j, k, dz);
                dGt[2][0] = Dx(G.tildeGamma[2], i, j, k, dx);
                dGt[2][1] = Dy(G.tildeGamma[2], i, j, k, dy);
                dGt[2][2] = Dz(G.tildeGamma[2], i, j, k, dz);

                double R_tilde[3][3];

                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b) {

                        const int         s_ab = tensorium_RG::sym6_index(a, b);
                        const Field3D<T> &Gab = G.gamma_tilde[s_ab];

                        const double d2xx = Dxx(Gab, i, j, k, dx);
                        const double d2yy = Dyy(Gab, i, j, k, dy);
                        const double d2zz = Dzz(Gab, i, j, k, dz);
                        const double d2xy = Dxy4(Gab, i, j, k, dx, dy);
                        const double d2xz = Dxz4(Gab, i, j, k, dx, dz);
                        const double d2yz = Dyz4(Gab, i, j, k, dy, dz);

                        const double lap_gab = gI[0][0] * d2xx + gI[1][1] * d2yy + gI[2][2] * d2zz +
                                               2.0 * gI[0][1] * d2xy + 2.0 * gI[0][2] * d2xz +
                                               2.0 * gI[1][2] * d2yz;

                        const double term1 = -0.5 * lap_gab;
                        double       term2 = 0.0;
                        for (int kk = 0; kk < 3; ++kk)
                            term2 += 0.5 * (g[a][kk] * dGt[kk][b] + g[b][kk] * dGt[kk][a]);

                        const double term3 = 0.5 * (Gt[0] * Gijk[0][a][b] + Gt[1] * Gijk[1][a][b] +
                                                    Gt[2] * Gijk[2][a][b]);

                        double term4 = 0.0;
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n) {
                                double sum = 0.0;
                                for (int kk = 0; kk < 3; ++kk) {
                                    sum += 2.0 * Gijk[kk][m][a] * Gijk[kk][n][b];
                                    sum -= Gijk[kk][a][b] * Gijk[kk][m][n];
                                }
                                term4 += gI[m][n] * sum;
                            }

                        const double val = term1 + term2 + term3 + term4;
                        R_tilde[a][b] = val;
                        if (a != b)
                            R_tilde[b][a] = val;
                    }

                Ricci6[XX].ptr()[id] = (T)(R_tilde[0][0] + R_chi[0][0]);
                Ricci6[XY].ptr()[id] = (T)(R_tilde[0][1] + R_chi[0][1]);
                Ricci6[XZ].ptr()[id] = (T)(R_tilde[0][2] + R_chi[0][2]);
                Ricci6[YY].ptr()[id] = (T)(R_tilde[1][1] + R_chi[1][1]);
                Ricci6[YZ].ptr()[id] = (T)(R_tilde[1][2] + R_chi[1][2]);
                Ricci6[ZZ].ptr()[id] = (T)(R_tilde[2][2] + R_chi[2][2]);
            }
        }
    }

    tensorium_RG::bssn::assert_invariants(G, "ricci", 4, throw_on_violation);
}

} // namespace tensorium_RG::bssn
