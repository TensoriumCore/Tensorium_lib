#pragma once
#include "BSSNCHristoffelTilde.hpp"
#include "BSSNGridDerivatives.hpp"
#include "BSSNGridSoA.hpp"
#include <cmath>
#include <stdio.h>

namespace tensorium_RG::bssn {

template <typename T> void compute_ricci_bssn(BSSNGridSoA<T> &G, Field3D<T> *Ricci6) {
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

                auto d_g = [&](int m, int a, int b) -> double {
                    if (a > b)
                        std::swap(a, b);
                    const Field3D<T> &F = (a == 0 && b == 0)   ? G.gamma_tilde[XX]
                                          : (a == 0 && b == 1) ? G.gamma_tilde[XY]
                                          : (a == 0 && b == 2) ? G.gamma_tilde[XZ]
                                          : (a == 1 && b == 1) ? G.gamma_tilde[YY]
                                          : (a == 1 && b == 2) ? G.gamma_tilde[YZ]
                                                               : G.gamma_tilde[ZZ];

                    if (m == 0)
                        return Dx(F, i, j, k, dx);
                    if (m == 1)
                        return Dy(F, i, j, k, dy);
                    return Dz(F, i, j, k, dz);
                };

                auto get_ginv = [&](int a, int b) -> double {
                    if (a > b)
                        std::swap(a, b);
                    const int idx = (a == 0 && b == 0)   ? XX
                                    : (a == 0 && b == 1) ? XY
                                    : (a == 0 && b == 2) ? XZ
                                    : (a == 1 && b == 1) ? YY
                                    : (a == 1 && b == 2) ? YZ
                                                         : ZZ;
                    return G.gamma_tilde_inv[idx].ptr()[id];
                };

                double Gijk[3][3][3];

                for (int kk = 0; kk < 3; ++kk) {
                    for (int ii = 0; ii < 3; ++ii) {
                        for (int jj = ii; jj < 3; ++jj) {
                            double sum = 0.0;
                            for (int ll = 0; ll < 3; ++ll) {
                                const double ginv_kl = get_ginv(kk, ll);
                                const double term =
                                    d_g(ii, jj, ll) + d_g(jj, ii, ll) - d_g(ll, ii, jj);
                                sum += ginv_kl * term;
                            }
                            double val = 0.5 * sum;
                            Gijk[kk][ii][jj] = val;
                            if (ii != jj)
                                Gijk[kk][jj][ii] = val;
                        }
                    }
                }

                auto christoffel = [&](int idx_k, int idx_a, int idx_b) -> double {
                    return Gijk[idx_k][idx_a][idx_b];
                };

                const double chi = G.chi.ptr()[id];
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
                cov_dchi[0][0] =
                    d2chi_xx - (christoffel(0, 0, 0) * dchi_x + christoffel(1, 0, 0) * dchi_y +
                                christoffel(2, 0, 0) * dchi_z);
                cov_dchi[0][1] =
                    d2chi_xy - (christoffel(0, 0, 1) * dchi_x + christoffel(1, 0, 1) * dchi_y +
                                christoffel(2, 0, 1) * dchi_z);
                cov_dchi[0][2] =
                    d2chi_xz - (christoffel(0, 0, 2) * dchi_x + christoffel(1, 0, 2) * dchi_y +
                                christoffel(2, 0, 2) * dchi_z);
                cov_dchi[1][1] =
                    d2chi_yy - (christoffel(0, 1, 1) * dchi_x + christoffel(1, 1, 1) * dchi_y +
                                christoffel(2, 1, 1) * dchi_z);
                cov_dchi[1][2] =
                    d2chi_yz - (christoffel(0, 1, 2) * dchi_x + christoffel(1, 1, 2) * dchi_y +
                                christoffel(2, 1, 2) * dchi_z);
                cov_dchi[2][2] =
                    d2chi_zz - (christoffel(0, 2, 2) * dchi_x + christoffel(1, 2, 2) * dchi_y +
                                christoffel(2, 2, 2) * dchi_z);

                cov_dchi[1][0] = cov_dchi[0][1];
                cov_dchi[2][0] = cov_dchi[0][2];
                cov_dchi[2][1] = cov_dchi[1][2];

                double lap_chi_conf = 0.0;
                lap_chi_conf += get_ginv(0, 0) * cov_dchi[0][0] + get_ginv(1, 1) * cov_dchi[1][1] +
                                get_ginv(2, 2) * cov_dchi[2][2];
                lap_chi_conf +=
                    2.0 * (get_ginv(0, 1) * cov_dchi[0][1] + get_ginv(0, 2) * cov_dchi[0][2] +
                           get_ginv(1, 2) * cov_dchi[1][2]);

                double grad_chi_sq = 0.0;
                grad_chi_sq += get_ginv(0, 0) * dchi_x * dchi_x + get_ginv(1, 1) * dchi_y * dchi_y +
                               get_ginv(2, 2) * dchi_z * dchi_z;
                grad_chi_sq +=
                    2.0 * (get_ginv(0, 1) * dchi_x * dchi_y + get_ginv(0, 2) * dchi_x * dchi_z +
                           get_ginv(1, 2) * dchi_y * dchi_z);

                double R_chi[3][3];
                double coeff1 = 0.5 * inv_chi;
                double coeff2 = 0.25 * inv_chi2;
                double coeff3 = 0.75 * inv_chi2 * grad_chi_sq;

                auto get_g = [&](int a, int b) -> double {
                    if (a > b)
                        std::swap(a, b);
                    const int idx = (a == 0 && b == 0)   ? XX
                                    : (a == 0 && b == 1) ? XY
                                    : (a == 0 && b == 2) ? XZ
                                    : (a == 1 && b == 1) ? YY
                                    : (a == 1 && b == 2) ? YZ
                                                         : ZZ;
                    return G.gamma_tilde[idx].ptr()[id];
                };

                R_chi[0][0] = coeff1 * (cov_dchi[0][0] + get_g(0, 0) * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_x - get_g(0, 0) * coeff3;
                R_chi[0][1] = coeff1 * (cov_dchi[0][1] + get_g(0, 1) * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_y - get_g(0, 1) * coeff3;
                R_chi[0][2] = coeff1 * (cov_dchi[0][2] + get_g(0, 2) * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_z - get_g(0, 2) * coeff3;
                R_chi[1][1] = coeff1 * (cov_dchi[1][1] + get_g(1, 1) * lap_chi_conf) -
                              coeff2 * dchi_y * dchi_y - get_g(1, 1) * coeff3;
                R_chi[1][2] = coeff1 * (cov_dchi[1][2] + get_g(1, 2) * lap_chi_conf) -
                              coeff2 * dchi_y * dchi_z - get_g(1, 2) * coeff3;
                R_chi[2][2] = coeff1 * (cov_dchi[2][2] + get_g(2, 2) * lap_chi_conf) -
                              coeff2 * dchi_z * dchi_z - get_g(2, 2) * coeff3;

                double d2g[6][6];
                d2g[XX][XX] = Dxx(G.gamma_tilde[XX], i, j, k, dx);
                d2g[XX][YY] = Dyy(G.gamma_tilde[XX], i, j, k, dy);
                d2g[XX][ZZ] = Dzz(G.gamma_tilde[XX], i, j, k, dz);
                d2g[XY][XX] = Dxx(G.gamma_tilde[XY], i, j, k, dx);
                d2g[XY][YY] = Dyy(G.gamma_tilde[XY], i, j, k, dy);
                d2g[XY][ZZ] = Dzz(G.gamma_tilde[XY], i, j, k, dz);
                d2g[XZ][XX] = Dxx(G.gamma_tilde[XZ], i, j, k, dx);
                d2g[XZ][YY] = Dyy(G.gamma_tilde[XZ], i, j, k, dy);
                d2g[XZ][ZZ] = Dzz(G.gamma_tilde[XZ], i, j, k, dz);
                d2g[YY][XX] = Dxx(G.gamma_tilde[YY], i, j, k, dx);
                d2g[YY][YY] = Dyy(G.gamma_tilde[YY], i, j, k, dy);
                d2g[YY][ZZ] = Dzz(G.gamma_tilde[YY], i, j, k, dz);
                d2g[YZ][XX] = Dxx(G.gamma_tilde[YZ], i, j, k, dx);
                d2g[YZ][YY] = Dyy(G.gamma_tilde[YZ], i, j, k, dy);
                d2g[YZ][ZZ] = Dzz(G.gamma_tilde[YZ], i, j, k, dz);
                d2g[ZZ][XX] = Dxx(G.gamma_tilde[ZZ], i, j, k, dx);
                d2g[ZZ][YY] = Dyy(G.gamma_tilde[ZZ], i, j, k, dy);
                d2g[ZZ][ZZ] = Dzz(G.gamma_tilde[ZZ], i, j, k, dz);

                d2g[XX][XY] = Dxy4(G.gamma_tilde[XX], i, j, k, dx, dy);
                d2g[XX][XZ] = Dxz4(G.gamma_tilde[XX], i, j, k, dx, dz);
                d2g[XX][YZ] = Dyz4(G.gamma_tilde[XX], i, j, k, dy, dz);
                d2g[XY][XY] = Dxy4(G.gamma_tilde[XY], i, j, k, dx, dy);
                d2g[XY][XZ] = Dxz4(G.gamma_tilde[XY], i, j, k, dx, dz);
                d2g[XY][YZ] = Dyz4(G.gamma_tilde[XY], i, j, k, dy, dz);
                d2g[XZ][XY] = Dxy4(G.gamma_tilde[XZ], i, j, k, dx, dy);
                d2g[XZ][XZ] = Dxz4(G.gamma_tilde[XZ], i, j, k, dx, dz);
                d2g[XZ][YZ] = Dyz4(G.gamma_tilde[XZ], i, j, k, dy, dz);
                d2g[YY][XY] = Dxy4(G.gamma_tilde[YY], i, j, k, dx, dy);
                d2g[YY][XZ] = Dxz4(G.gamma_tilde[YY], i, j, k, dx, dz);
                d2g[YY][YZ] = Dyz4(G.gamma_tilde[YY], i, j, k, dy, dz);
                d2g[YZ][XY] = Dxy4(G.gamma_tilde[YZ], i, j, k, dx, dy);
                d2g[YZ][XZ] = Dxz4(G.gamma_tilde[YZ], i, j, k, dx, dz);
                d2g[YZ][YZ] = Dyz4(G.gamma_tilde[YZ], i, j, k, dy, dz);
                d2g[ZZ][XY] = Dxy4(G.gamma_tilde[ZZ], i, j, k, dx, dy);
                d2g[ZZ][XZ] = Dxz4(G.gamma_tilde[ZZ], i, j, k, dx, dz);
                d2g[ZZ][YZ] = Dyz4(G.gamma_tilde[ZZ], i, j, k, dy, dz);

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

                const double Gt[3] = {G.tildeGamma[0].ptr()[id], G.tildeGamma[1].ptr()[id],
                                      G.tildeGamma[2].ptr()[id]};

                double R_tilde[3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        int idx_ab = (a == 0 && b == 0)   ? XX
                                     : (a == 0 && b == 1) ? XY
                                     : (a == 0 && b == 2) ? XZ
                                     : (a == 1 && b == 1) ? YY
                                     : (a == 1 && b == 2) ? YZ
                                                          : ZZ;

                        double term1 = 0.0;
                        term1 += get_ginv(0, 0) * d2g[idx_ab][XX] +
                                 get_ginv(1, 1) * d2g[idx_ab][YY] +
                                 get_ginv(2, 2) * d2g[idx_ab][ZZ];
                        term1 += 2.0 * (get_ginv(0, 1) * d2g[idx_ab][XY] +
                                        get_ginv(0, 2) * d2g[idx_ab][XZ] +
                                        get_ginv(1, 2) * d2g[idx_ab][YZ]);
                        term1 *= -0.5;

                        double term2 = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            term2 += 0.5 * (get_g(a, k) * dGt[k][b] + get_g(b, k) * dGt[k][a]);
                        }

                        double term3 =
                            0.5 * (Gt[0] * christoffel(0, a, b) + Gt[1] * christoffel(1, a, b) +
                                   Gt[2] * christoffel(2, a, b));

                        double term4 = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                double sum_Gam = 0.0;
                                for (int k = 0; k < 3; ++k) {
                                    sum_Gam += 2.0 * christoffel(k, m, a) * christoffel(k, n, b);
                                    sum_Gam -= christoffel(k, a, b) * christoffel(k, m, n);
                                }
                                term4 += get_ginv(m, n) * sum_Gam;
                            }
                        }

                        R_tilde[a][b] = term1 + term2 + term3 + term4;
                    }
                }

                R_tilde[1][0] = R_tilde[0][1];
                R_tilde[2][0] = R_tilde[0][2];
                R_tilde[2][1] = R_tilde[1][2];

                Ricci6[XX].ptr()[id] = R_tilde[0][0] + R_chi[0][0];
                Ricci6[XY].ptr()[id] = R_tilde[0][1] + R_chi[0][1];
                Ricci6[XZ].ptr()[id] = R_tilde[0][2] + R_chi[0][2];
                Ricci6[YY].ptr()[id] = R_tilde[1][1] + R_chi[1][1];
                Ricci6[YZ].ptr()[id] = R_tilde[1][2] + R_chi[1][2];
                Ricci6[ZZ].ptr()[id] = R_tilde[2][2] + R_chi[2][2];
            }
        }
    }
}
} // namespace tensorium_RG::bssn
