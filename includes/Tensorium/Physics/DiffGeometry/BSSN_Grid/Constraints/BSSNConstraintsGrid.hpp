#pragma once
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNGamma.hpp"
#include "../Geometry/BSSNInvariants.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

/**
 * @file BSSNConstraintsGrid.hpp
 * @brief Hamiltonian, momentum, and Gamma-constraint evaluation on the structured grid.
 * @details
 * The constraints follow the vacuum ADM definitions expressed in terms of the conformal variables:
 * \f[
 * \begin{aligned}
 * H &= R + K^2 - K_{ij}K^{ij},\\
 * M_i &= D_j K^j_{\ i} - D_i K,\\
 * C_i &= \tilde{\Gamma}^i + \partial_j\tilde{\gamma}^{ij}.
 * \end{aligned}
 * \f]
 * `compute_bssn_constraints` first reconstructs physical-metric quantities
 * (\f$\gamma_{ij}=\chi^{-1}\tilde{\gamma}_{ij}\f$, \f$K_{ij}=\chi^{-1}(\tilde{A}_{ij}+\tfrac{1}{3}\gamma_{ij}K)\f$) and then
 * uses the derivative operators to evaluate the above expressions on every interior grid point.
 */

namespace tensorium_RG::bssn {

namespace detail {
constexpr int sym_row_from_index[6] = {0, 0, 0, 1, 1, 2};
constexpr int sym_col_from_index[6] = {0, 1, 2, 1, 2, 2};
} // namespace detail

template <typename T>
inline void compute_momentum_constraint_Mi(const BSSNGridSoA<T> &G, Field3D<T> Mi[3],
                                           size_t padding = 4) {
    using namespace tensorium_RG::fd;

    const size_t total = G.chi.st.nx_tot * G.chi.st.ny_tot * G.chi.st.nz_tot;
    for (int comp = 0; comp < 3; ++comp)
        std::fill_n(Mi[comp].ptr(), total, T(0));

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    auto deriv = [&](const Field3D<T> &F, size_t ii, size_t jj, size_t kk, int dir) -> double {
        return (dir == 0)   ? Dx(F, ii, jj, kk, G.dx)
               : (dir == 1) ? Dy(F, ii, jj, kk, G.dy)
                            : Dz(F, ii, jj, kk, G.dz);
    };

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.chi.idx(i, j, k);

                double g_inv[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        g_inv[a][b] = (double)tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);

                double A_lo[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        A_lo[a][b] = (double)tensorium_RG::sym6_get(G.A_tilde, id, a, b);

                double d_g_inv[3][3][3] = {};
                double dA_lo[3][3][3] = {};
                for (int dir = 0; dir < 3; ++dir) {
                    for (int s = 0; s < 6; ++s) {
                        const int row = detail::sym_row_from_index[s];
                        const int col = detail::sym_col_from_index[s];
                        const Field3D<T> &F_inv = G.gamma_tilde_inv[s];
                        const Field3D<T> &F_A = G.A_tilde[s];
                        const double dginv = deriv(F_inv, i, j, k, dir);
                        const double dA = deriv(F_A, i, j, k, dir);
                        d_g_inv[dir][row][col] = dginv;
                        d_g_inv[dir][col][row] = dginv;
                        dA_lo[dir][row][col] = dA;
                        dA_lo[dir][col][row] = dA;
                    }
                }

                const double chi = (double)G.chi.ptr()[id];
                const double inv_chi = (chi != 0.0) ? (1.0 / chi) : 0.0;
                const double d_chi[3] = {deriv(G.chi, i, j, k, 0), deriv(G.chi, i, j, k, 1),
                                        deriv(G.chi, i, j, k, 2)};
                const double dK[3] = {deriv(G.K, i, j, k, 0), deriv(G.K, i, j, k, 1),
                                      deriv(G.K, i, j, k, 2)};

                double A_mix[3][3];
                for (int up = 0; up < 3; ++up)
                    for (int lo = 0; lo < 3; ++lo) {
                        double sum = 0.0;
                        for (int m = 0; m < 3; ++m)
                            sum += g_inv[up][m] * A_lo[m][lo];
                        A_mix[up][lo] = sum;
                    }

                double dA_mix[3][3][3];
                for (int dir = 0; dir < 3; ++dir)
                    for (int up = 0; up < 3; ++up)
                        for (int lo = 0; lo < 3; ++lo) {
                            double sum = 0.0;
                            for (int m = 0; m < 3; ++m)
                                sum += d_g_inv[dir][up][m] * A_lo[m][lo] +
                                        g_inv[up][m] * dA_lo[dir][m][lo];
                            dA_mix[dir][up][lo] = sum;
                        }

                T Gamma_vals_T[3][3][3];
                tensorium_RG::bssn::compute_tildeGamma_symbols(G, i, j, k, Gamma_vals_T);
                double Gamma_vals[3][3][3];
                for (int up = 0; up < 3; ++up)
                    for (int lo1 = 0; lo1 < 3; ++lo1)
                        for (int lo2 = 0; lo2 < 3; ++lo2)
                            Gamma_vals[up][lo1][lo2] = (double)Gamma_vals_T[up][lo1][lo2];

                double traceGamma[3] = {0.0, 0.0, 0.0};
                for (int kidx = 0; kidx < 3; ++kidx)
                    for (int jidx = 0; jidx < 3; ++jidx)
                        traceGamma[kidx] += Gamma_vals[jidx][jidx][kidx];

                const double grad_phi_factor[3] = {-(1.5) * d_chi[0] * inv_chi,
                                                   -(1.5) * d_chi[1] * inv_chi,
                                                   -(1.5) * d_chi[2] * inv_chi};

                for (int lo = 0; lo < 3; ++lo) {
                    double divA = dA_mix[0][0][lo] + dA_mix[1][1][lo] + dA_mix[2][2][lo];
                    for (int kidx = 0; kidx < 3; ++kidx)
                        divA += traceGamma[kidx] * A_mix[kidx][lo];

                    double conn_term = 0.0;
                    for (int jidx = 0; jidx < 3; ++jidx)
                        for (int kidx = 0; kidx < 3; ++kidx)
                            conn_term += Gamma_vals[kidx][jidx][lo] * A_mix[jidx][kidx];
                    divA -= conn_term;

                    double phi_term = 0.0;
                    for (int jidx = 0; jidx < 3; ++jidx)
                        phi_term += A_mix[jidx][lo] * grad_phi_factor[jidx];

                    const double Mi_val = divA + phi_term - (2.0 / 3.0) * dK[lo];
                    Mi[lo].ptr()[id] = (T)Mi_val;
                }
            }
        }
    }
}

/**
 * @brief Fill the Hamiltonian \f$H\f$, momentum \f$M_i\f$, and Gamma \f$C_i\f$ constraint fields.
 * @param Ricci6 Packed \f$R_{ij}\f$ produced by `compute_ricci_bssn`.
 * @param H_out,M_out,C_out Destinations that must be allocated like the grid.
 * @param r_min,r_max Optional radial filter to skip interior/exterior regions (useful around punctures).
 * @details
 * - Hamiltonian block maps `gI_*` multiplications to \f$R=\gamma^{ij}R_{ij}\f$.
 * - Momentum block forms \f$M_i\f$ via `compute_momentum_constraint_Mi`, which evaluates the lower
 *   index expression \f$\tilde{D}_j\tilde{A}^j{}_{\ i} + 6\tilde{A}^j{}_{\ i}\partial_j\phi - \tfrac{2}{3}\partial_i K\f$.
 * - Gamma constraint reuses `metric_inverse_divergence` to compare the evolved \f$\tilde{\Gamma}^i\f$ against
 *   \f$-\partial_j\tilde{\gamma}^{ij}\f$.
 */
template <typename T>
static inline void compute_bssn_constraints(BSSNGridSoA<T> &G, const Field3D<T> *Ricci6,
                                            Field3D<T> &H_out, Field3D<T> M_out[3],
                                            Field3D<T> C_out[3], double r_min, double r_max,
                                            double xc, double yc, double zc,
                                            bool throw_on_violation = true) {
    using namespace tensorium_RG::fd;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total = G.chi.st.nx_tot * G.chi.st.ny_tot * G.chi.st.nz_tot;
    std::fill_n(H_out.ptr(), total, T(0));
    for (int a = 0; a < 3; ++a)
        std::fill_n(C_out[a].ptr(), total, T(0));

    compute_momentum_constraint_Mi(G, M_out, 4);

    const size_t i0 = I0 + 4, i1 = I1 - 4;
    const size_t j0 = J0 + 4, j1 = J1 - 4;
    const size_t k0 = K0 + 4, k1 = K1 - 4;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;
                const size_t id = G.chi.idx(i, j, k);
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r <= r_min || r >= r_max) {
                    for (int comp = 0; comp < 3; ++comp)
                        M_out[comp].ptr()[id] = T(0);
                    continue;
                }

                const double chi = (double)G.chi.ptr()[id];

                const double Ktr = (double)G.K.ptr()[id];

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

                T div_gamma[3];
                tensorium_RG::bssn::metric_inverse_divergence(G, i, j, k, div_gamma);
                for (int iC = 0; iC < 3; ++iC) {
                    const double Ci = (double)G.tildeGamma[iC].ptr()[id] + (double)div_gamma[iC];
                    C_out[iC].ptr()[id] = (T)Ci;
                }
            }
        }
    }

    tensorium_RG::bssn::assert_invariants(G, "constraints", 4,
                                          tensorium_RG::bssn::compute_invariant_tolerances(G),
                                          throw_on_violation);
}

template <typename T>
inline void compute_z4c_constraints(BSSNGridSoA<T> &G, const Field3D<T> *Ricci6, Field3D<T> &H_out,
                                    Field3D<T> M_out[3], Field3D<T> C_out[3], double r_min,
                                    double r_max, double xc, double yc, double zc,
                                    bool throw_on_violation = true) {
    compute_bssn_constraints(G, Ricci6, H_out, M_out, C_out, r_min, r_max, xc, yc, zc,
                             throw_on_violation);
}

/**
 * @brief Convenience logger that prints Linf/L2 norms of \f$H\f$, \f$|\vec{M}|\f$, and \f$|\vec{C}|\f$ to stdout.
 */
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
