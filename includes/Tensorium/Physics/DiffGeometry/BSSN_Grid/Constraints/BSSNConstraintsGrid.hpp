#pragma once
#include "../Geometry/BSSNConformal.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tensorium_RG::bssn {

template <typename T>
static inline void compute_bssn_constraints(BSSNGridSoA<T> &G, const Field3D<T> *Ricci6,
                                            Field3D<T> &H_out, Field3D<T> M_out[3],
                                            Field3D<T> C_out[3], double r_min, double r_max,
                                            double xc, double yc, double zc) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 4, i1 = I1 - 4;
    const size_t j0 = J0 + 4, j1 = J1 - 4;
    const size_t k0 = K0 + 4, k1 = K1 - 4;

    const double dx = G.dx, dy = G.dy, dz = G.dz;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.chi.idx(i, j, k);

                const double chi = (double)G.chi.ptr()[id];
                const double inv_chi = 1.0 / chi;

                const double dchi[3] = {(double)Dx(G.chi, i, j, k, dx),
                                        (double)Dy(G.chi, i, j, k, dy),
                                        (double)Dz(G.chi, i, j, k, dz)};

                double d6phi[3];
                tensorium_RG::bssn::grad_6phi_from_chi(dchi, chi, d6phi);

                const double Ktr = (double)G.K.ptr()[id];
                const double dK[3] = {(double)Dx(G.K, i, j, k, dx), (double)Dy(G.K, i, j, k, dy),
                                      (double)Dz(G.K, i, j, k, dz)};

                double gtI[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        gtI[a][b] = (double)tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);

                double Atd[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        Atd[a][b] = (double)tensorium_RG::sym6_get(G.A_tilde, id, a, b);

                double Atu[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double s = 0.0;
                        for (int k1 = 0; k1 < 3; ++k1)
                            for (int k2 = 0; k2 < 3; ++k2)
                                s += gtI[a][k1] * gtI[b][k2] * Atd[k1][k2];
                        Atu[a][b] = s;
                    }

                double A2 = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        A2 += Atd[a][b] * Atu[a][b];

                const double Rxx = (double)Ricci6[XX].ptr()[id];
                const double Rxy = (double)Ricci6[XY].ptr()[id];
                const double Rxz = (double)Ricci6[XZ].ptr()[id];
                const double Ryy = (double)Ricci6[YY].ptr()[id];
                const double Ryz = (double)Ricci6[YZ].ptr()[id];
                const double Rzz = (double)Ricci6[ZZ].ptr()[id];

                const double gI_xx = chi * gtI[0][0];
                const double gI_xy = chi * gtI[0][1];
                const double gI_xz = chi * gtI[0][2];
                const double gI_yy = chi * gtI[1][1];
                const double gI_yz = chi * gtI[1][2];
                const double gI_zz = chi * gtI[2][2];

                const double Rsc = gI_xx * Rxx + 2.0 * gI_xy * Rxy + 2.0 * gI_xz * Rxz +
                                   gI_yy * Ryy + 2.0 * gI_yz * Ryz + gI_zz * Rzz;

                const double H = Rsc + (2.0 / 3.0) * Ktr * Ktr - A2;
                H_out.ptr()[id] = (T)H;

                double d_gI[3][3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        const int idx = tensorium_RG::sym6_index(a, b);

                        if (a > b)
                            continue;
                        const Field3D<T> &F = G.gamma_tilde_inv[idx];
                        d_gI[a][b][0] = (double)Dx(F, i, j, k, dx);
                        d_gI[a][b][1] = (double)Dy(F, i, j, k, dy);
                        d_gI[a][b][2] = (double)Dz(F, i, j, k, dz);
                        d_gI[b][a][0] = d_gI[a][b][0];
                        d_gI[b][a][1] = d_gI[a][b][1];
                        d_gI[b][a][2] = d_gI[a][b][2];
                    }
                }

                for (int ii = 0; ii < 3; ++ii) {
                    double div = 0.0;
                    for (int jj = 0; jj < 3; ++jj) {
                        for (int kk = 0; kk < 3; ++kk) {
                            const int         idx_jk = tensorium_RG::sym6_index(jj, kk);
                            const Field3D<T> &Fjk = G.A_tilde[idx_jk];

                            double dA_jk = 0.0;
                            if (jj == 0)
                                dA_jk = (double)Dx(Fjk, i, j, k, dx);
                            else if (jj == 1)
                                dA_jk = (double)Dy(Fjk, i, j, k, dy);
                            else
                                dA_jk = (double)Dz(Fjk, i, j, k, dz);

                            double dAtu = 0.0;
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b) {
                                    dAtu += d_gI[ii][a][jj] * gtI[kk][b] * Atd[a][b];
                                    dAtu += gtI[ii][a] * d_gI[kk][b][jj] * Atd[a][b];
                                    dAtu += gtI[ii][a] * gtI[kk][b] * dA_jk;
                                }
                            div += dAtu;
                        }
                    }

                    double phi_term = 0.0;
                    for (int jdir = 0; jdir < 3; ++jdir) {
                        double s = 0.0;
                        for (int j = 0; j < 3; ++j)
                            s += Atu[ii][j] * (j == jdir ? 1.0 : 0.0);
                        phi_term += s * d6phi[jdir];
                    }

                    double DK = 0.0;
                    for (int j = 0; j < 3; ++j)
                        DK += gtI[ii][j] * dK[j];

                    const double Mi = div + phi_term - (2.0 / 3.0) * DK;
                    M_out[ii].ptr()[id] = (T)Mi;
                }

                for (int iC = 0; iC < 3; ++iC) {
                    double div_gI = 0.0;
                    for (int jC = 0; jC < 3; ++jC)
                        div_gI += d_gI[iC][jC][jC];
                    const double Ci = (double)G.tildeGamma[iC].ptr()[id] + div_gI;
                    C_out[iC].ptr()[id] = (T)Ci;
                }
            }
        }
    }
}

static inline void print_constraint_norms(BSSNGridSoA<double> &G, const Field3D<double> &H,
                                          const Field3D<double> M[3], const Field3D<double> C[3],
                                          double r_min, double r_max, double xc, double yc,
                                          double zc) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 4, i1 = I1 - 4;
    const size_t j0 = J0 + 4, j1 = J1 - 4;
    const size_t k0 = K0 + 4, k1 = K1 - 4;

    double maxH = 0, maxM = 0, maxC = 0;
    double l2H = 0, l2M = 0, l2C = 0;
    size_t n = 0;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.chi.idx(i, j, k);

                const double h = std::abs(H.ptr()[id]);
                const double m =
                    std::sqrt(M[0].ptr()[id] * M[0].ptr()[id] + M[1].ptr()[id] * M[1].ptr()[id] +
                              M[2].ptr()[id] * M[2].ptr()[id]);
                const double c =
                    std::sqrt(C[0].ptr()[id] * C[0].ptr()[id] + C[1].ptr()[id] * C[1].ptr()[id] +
                              C[2].ptr()[id] * C[2].ptr()[id]);

                maxH = std::max(maxH, h);
                maxM = std::max(maxM, m);
                maxC = std::max(maxC, c);

                l2H += h * h;
                l2M += m * m;
                l2C += c * c;
                n++;
            }

    l2H = std::sqrt(l2H / std::max<size_t>(1, n));
    l2M = std::sqrt(l2M / std::max<size_t>(1, n));
    l2C = std::sqrt(l2C / std::max<size_t>(1, n));

    std::printf("[CONS] r in (%.3f, %.3f) samples=%zu\n", r_min, r_max, n);
    std::printf("[CONS] H: Linf=%.3e  L2=%.3e\n", maxH, l2H);
    std::printf("[CONS] M: Linf=%.3e  L2=%.3e\n", maxM, l2M);
    std::printf("[CONS] C: Linf=%.3e  L2=%.3e\n", maxC, l2C);
}

} // namespace tensorium_RG::bssn
