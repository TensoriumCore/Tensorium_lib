#pragma once
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

                const double gxx = G.gamma_tilde[XX].ptr()[id];
                const double gxy = G.gamma_tilde[XY].ptr()[id];
                const double gxz = G.gamma_tilde[XZ].ptr()[id];
                const double gyy = G.gamma_tilde[YY].ptr()[id];
                const double gyz = G.gamma_tilde[YZ].ptr()[id];
                const double gzz = G.gamma_tilde[ZZ].ptr()[id];

                const double gixx = G.gamma_tilde_inv[XX].ptr()[id];
                const double gixy = G.gamma_tilde_inv[XY].ptr()[id];
                const double gixz = G.gamma_tilde_inv[XZ].ptr()[id];
                const double giyy = G.gamma_tilde_inv[YY].ptr()[id];
                const double giyz = G.gamma_tilde_inv[YZ].ptr()[id];
                const double gizz = G.gamma_tilde_inv[ZZ].ptr()[id];

                const double Gt_x = G.tildeGamma[0].ptr()[id];
                const double Gt_y = G.tildeGamma[1].ptr()[id];
                const double Gt_z = G.tildeGamma[2].ptr()[id];

                const double G_xxx = G.Gamma_tilde[0 * 9 + 0 * 3 + 0].ptr()[id];
                const double G_xxy = G.Gamma_tilde[0 * 9 + 0 * 3 + 1].ptr()[id];
                const double G_xxz = G.Gamma_tilde[0 * 9 + 0 * 3 + 2].ptr()[id];
                const double G_xyy = G.Gamma_tilde[0 * 9 + 1 * 3 + 1].ptr()[id];
                const double G_xyz = G.Gamma_tilde[0 * 9 + 1 * 3 + 2].ptr()[id];
                const double G_xzz = G.Gamma_tilde[0 * 9 + 2 * 3 + 2].ptr()[id];
                const double G_yxx = G.Gamma_tilde[1 * 9 + 0 * 3 + 0].ptr()[id];
                const double G_yxy = G.Gamma_tilde[1 * 9 + 0 * 3 + 1].ptr()[id];
                const double G_yxz = G.Gamma_tilde[1 * 9 + 0 * 3 + 2].ptr()[id];
                const double G_yyy = G.Gamma_tilde[1 * 9 + 1 * 3 + 1].ptr()[id];
                const double G_yyz = G.Gamma_tilde[1 * 9 + 1 * 3 + 2].ptr()[id];
                const double G_yzz = G.Gamma_tilde[1 * 9 + 2 * 3 + 2].ptr()[id];
                const double G_zxx = G.Gamma_tilde[2 * 9 + 0 * 3 + 0].ptr()[id];
                const double G_zxy = G.Gamma_tilde[2 * 9 + 0 * 3 + 1].ptr()[id];
                const double G_zxz = G.Gamma_tilde[2 * 9 + 0 * 3 + 2].ptr()[id];
                const double G_zyy = G.Gamma_tilde[2 * 9 + 1 * 3 + 1].ptr()[id];
                const double G_zyz = G.Gamma_tilde[2 * 9 + 1 * 3 + 2].ptr()[id];
                const double G_zzz = G.Gamma_tilde[2 * 9 + 2 * 3 + 2].ptr()[id];

                auto christoffel = [&](int k, int a, int b) -> double {
                    if (a > b)
                        std::swap(a, b);
                    return G.Gamma_tilde[k * 9 + a * 3 + b].ptr()[id];
                };

                double cov_dchi[3][3];
                cov_dchi[0][0] = d2chi_xx - (G_xxx * dchi_x + G_yxx * dchi_y + G_zxx * dchi_z);
                cov_dchi[0][1] = d2chi_xy - (G_xxy * dchi_x + G_yxy * dchi_y + G_zxy * dchi_z);
                cov_dchi[0][2] = d2chi_xz - (G_xxz * dchi_x + G_yxz * dchi_y + G_zxz * dchi_z);
                cov_dchi[1][1] = d2chi_yy - (G_xyy * dchi_x + G_yyy * dchi_y + G_zyy * dchi_z);
                cov_dchi[1][2] = d2chi_yz - (G_xyz * dchi_x + G_yyz * dchi_y + G_zyz * dchi_z);
                cov_dchi[2][2] = d2chi_zz - (G_xzz * dchi_x + G_yzz * dchi_y + G_zzz * dchi_z);

                cov_dchi[1][0] = cov_dchi[0][1];
                cov_dchi[2][0] = cov_dchi[0][2];
                cov_dchi[2][1] = cov_dchi[1][2];

                double lap_chi_conf = 0.0;
                lap_chi_conf +=
                    gixx * cov_dchi[0][0] + giyy * cov_dchi[1][1] + gizz * cov_dchi[2][2];
                lap_chi_conf +=
                    2.0 * (gixy * cov_dchi[0][1] + gixz * cov_dchi[0][2] + giyz * cov_dchi[1][2]);

                double grad_chi_sq = 0.0;
                grad_chi_sq +=
                    gixx * dchi_x * dchi_x + giyy * dchi_y * dchi_y + gizz * dchi_z * dchi_z;
                grad_chi_sq += 2.0 * (gixy * dchi_x * dchi_y + gixz * dchi_x * dchi_z +
                                      giyz * dchi_y * dchi_z);

                double R_chi[3][3];
                double coeff1 = 0.5 * inv_chi;
                double coeff2 = 0.25 * inv_chi2;
                double coeff3 = 0.75 * inv_chi2 * grad_chi_sq;

                R_chi[0][0] = coeff1 * (cov_dchi[0][0] + gxx * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_x - gxx * coeff3;
                R_chi[0][1] = coeff1 * (cov_dchi[0][1] + gxy * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_y - gxy * coeff3;
                R_chi[0][2] = coeff1 * (cov_dchi[0][2] + gxz * lap_chi_conf) -
                              coeff2 * dchi_x * dchi_z - gxz * coeff3;
                R_chi[1][1] = coeff1 * (cov_dchi[1][1] + gyy * lap_chi_conf) -
                              coeff2 * dchi_y * dchi_y - gyy * coeff3;
                R_chi[1][2] = coeff1 * (cov_dchi[1][2] + gyz * lap_chi_conf) -
                              coeff2 * dchi_y * dchi_z - gyz * coeff3;
                R_chi[2][2] = coeff1 * (cov_dchi[2][2] + gzz * lap_chi_conf) -
                              coeff2 * dchi_z * dchi_z - gzz * coeff3;

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

                if (i == I0 + 5 && j == J0 + 5 && k == K0 + 5) {
                    const double a = Dxy(G.chi, i, j, k, dx, dy);
                    const double b = Dxy4(G.chi, i, j, k, dx, dy);
                    printf("[DBG] Dxy vs Dxy4 at (%zu,%zu,%zu): %.17e %.17e diff=%.3e\n", i, j, k,
                           a, b, std::abs(a - b));
                }
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
                        term1 += gixx * d2g[idx_ab][XX] + giyy * d2g[idx_ab][YY] +
                                 gizz * d2g[idx_ab][ZZ];
                        term1 += 2.0 * (gixy * d2g[idx_ab][XY] + gixz * d2g[idx_ab][XZ] +
                                        giyz * d2g[idx_ab][YZ]);
                        term1 *= -0.5;

                        double term2 = 0.0;
                        double gab_vals[3] = {(a == 0)   ? gxx
                                              : (a == 1) ? gxy
                                                         : gxz,
                                              (a == 0)   ? gxy
                                              : (a == 1) ? gyy
                                                         : gyz,
                                              (a == 0)   ? gxz
                                              : (a == 1) ? gyz
                                                         : gzz};
                        double gbb_vals[3] = {(b == 0)   ? gxx
                                              : (b == 1) ? gxy
                                                         : gxz,
                                              (b == 0)   ? gxy
                                              : (b == 1) ? gyy
                                                         : gyz,
                                              (b == 0)   ? gxz
                                              : (b == 1) ? gyz
                                                         : gzz};

                        for (int k = 0; k < 3; ++k) {
                            term2 += 0.5 * (gab_vals[k] * dGt[k][b] + gbb_vals[k] * dGt[k][a]);
                        }

                        double term3 =
                            0.5 * (Gt_x * christoffel(0, a, b) + Gt_y * christoffel(1, a, b) +
                                   Gt_z * christoffel(2, a, b));

                        double term4 = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                double ginv_mn = (m == 0 && n == 0)                           ? gixx
                                                 : (m == 1 && n == 1)                         ? giyy
                                                 : (m == 2 && n == 2)                         ? gizz
                                                 : ((m == 0 && n == 1) || (m == 1 && n == 0)) ? gixy
                                                 : ((m == 0 && n == 2) || (m == 2 && n == 0))
                                                     ? gixz
                                                     : giyz;

                                double sum_Gam = 0.0;
                                for (int k = 0; k < 3; ++k) {
                                    sum_Gam += 2.0 * christoffel(k, m, a) * christoffel(k, n, b);
                                    sum_Gam -= christoffel(k, a, b) * christoffel(k, m, n);
                                }
                                term4 += ginv_mn * sum_Gam;
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
