#pragma once

#include "../Grid/BSSNGridOperations.hpp"
#include <cstddef>

/**
 * @file BSSNGridDerivatives.hpp
 * @brief High-order finite-difference stencils and dissipation operators used by the BSSN grid.
 * @details
 * All evolution, geometry, and constraint kernels call into this header to evaluate spatial
 * derivatives on the structure-of-arrays fields defined in `Fields/BSSNGridSoA.hpp`.  Unless noted
 * otherwise, operators are 4th-order accurate centered differences that require two guard cells on
 * each side:
 * \f[
 * \partial_x f_{i,j,k} = \frac{-f_{i+2,j,k} + 8 f_{i+1,j,k} - 8 f_{i-1,j,k} + f_{i-2,j,k}}
 *                             {12\,\Delta x}.
 * \f]
 * Second derivatives follow the standard \f$(-1,16,-30,16,-1)/12\f$ stencil, and mixed derivatives
 * use either the classical 2nd-order cross operator (`Dxy`, `Dxz`, `Dyz`) or the 4th-order
 * tensor-product version (`Dxy4`, `Dxz4`, `Dyz4`) when Ricci evaluations demand higher accuracy.
 * Upwinded advection relies on biased 3-point formulas activated according to the sign of the local
 * shift. Kreiss–Oliger dissipation implements the 6th-order filter used to stabilize high-frequency
 * numerical noise in hyperbolic systems.
 *
 * @note These operators assume uniform spacing along each axis and share the units and conventions
 * from @ref BSSN_Grid.  Callers are expected to clamp loop bounds so the stencil footprints never
 * cross uninitialized halos.
 */

namespace tensorium_RG::fd {

enum Variance { Up, Down };
enum Axis { X = 0, Y = 1, Z = 2 };

/// @brief Utility accessor that forwards to `Field3D::idx` / raw pointer storage.
template <typename Field> inline double get(const Field &f, size_t i, size_t j, size_t k) {
    return f.ptr()[f.idx(i, j, k)];
}

/**
 * @brief 4th-order centered derivative along x.
 * @details Implements the stencil shown in the file-level documentation.
 * @param f Field to differentiate.
 * @param dx Grid spacing \f$\Delta x\f$.
 */
template <typename Field>
inline double Dx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    return (-get(f, i + 2, j, k) + 8.0 * get(f, i + 1, j, k) - 8.0 * get(f, i - 1, j, k) +
            get(f, i - 2, j, k)) /
           (12.0 * dx);
}

/// @brief 4th-order centered derivative along y.
template <typename Field>
inline double Dy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    return (-get(f, i, j + 2, k) + 8.0 * get(f, i, j + 1, k) - 8.0 * get(f, i, j - 1, k) +
            get(f, i, j - 2, k)) /
           (12.0 * dy);
}

/// @brief 4th-order centered derivative along z.
template <typename Field>
inline double Dz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    return (-get(f, i, j, k + 2) + 8.0 * get(f, i, j, k + 1) - 8.0 * get(f, i, j, k - 1) +
            get(f, i, j, k - 2)) /
           (12.0 * dz);
}

/**
 * @brief 4th-order approximation to \f$\partial_{xx} f\f$.
 * @note Uses the classic \f$(-1,16,-30,16,-1)/12\f$ stencil scaled by \f$1/\Delta x^2\f$.
 */
template <typename Field>
inline double Dxx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    double inv_dx2 = 1.0 / (dx * dx);
    return (-get(f, i + 2, j, k) + 16.0 * get(f, i + 1, j, k) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i - 1, j, k) - get(f, i - 2, j, k)) *
           (1.0 / 12.0) * inv_dx2;
}

template <typename Field>
inline double Dyy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    double inv_dy2 = 1.0 / (dy * dy);
    return (-get(f, i, j + 2, k) + 16.0 * get(f, i, j + 1, k) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i, j - 1, k) - get(f, i, j - 2, k)) *
           (1.0 / 12.0) * inv_dy2;
}

template <typename Field>
inline double Dzz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    double inv_dz2 = 1.0 / (dz * dz);
    return (-get(f, i, j, k + 2) + 16.0 * get(f, i, j, k + 1) - 30.0 * get(f, i, j, k) +
            16.0 * get(f, i, j, k - 1) - get(f, i, j, k - 2)) *
           (1.0 / 12.0) * inv_dz2;
}

/**
 * @brief Mixed second derivative using the compact second-order stencil.
 * @details Useful near where 4th-order cross derivatives would require larger halos.
 */
template <typename Field>
inline double Dxy(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    return (get(f, i + 1, j + 1, k) - get(f, i + 1, j - 1, k) - get(f, i - 1, j + 1, k) +
            get(f, i - 1, j - 1, k)) /
           (4.0 * dx * dy);
}

template <typename Field>
inline double Dxz(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    return (get(f, i + 1, j, k + 1) - get(f, i + 1, j, k - 1) - get(f, i - 1, j, k + 1) +
            get(f, i - 1, j, k - 1)) /
           (4.0 * dx * dz);
}

template <typename Field>
inline double Dyz(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    return (get(f, i, j + 1, k + 1) - get(f, i, j + 1, k - 1) - get(f, i, j - 1, k + 1) +
            get(f, i, j - 1, k - 1)) /
           (4.0 * dy * dz);
}

/**
 * @brief Tensor-product 4th-order approximation to \f$\partial_{xy} f\f$.
 * @details Expands the 1D \f$(-1,8,-8,1)/12\f$ derivative in both axes and rescales by
 *          \f$144\,\Delta x\,\Delta y\f$ so Ricci builders can reuse it for the
 * \f$\tilde{R}_{ij}\f$ terms.
 */
template <typename Field>
inline double Dxy4(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    double sum = 0.0;

    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i + idx[a], j + idx[b], k);
        }
    }

    return sum / (144.0 * dx * dy);
}

template <typename Field>
inline double Dxz4(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    double       sum = 0.0;
    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i + idx[a], j, k + idx[b]);
        }
    }
    return sum / (144.0 * dx * dz);
}

template <typename Field>
inline double Dyz4(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    double       sum = 0.0;
    const int    idx[4] = {-2, -1, 1, 2};
    const double w[4] = {-1.0, 8.0, -8.0, 1.0};
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * get(f, i, j + idx[a], k + idx[b]);
        }
    }
    return sum / (144.0 * dy * dz);
}

/// @brief Axis selector that forwards to `Dx/Dy/Dz` for compile-time loops.
template <typename Field>
inline double D(const Field &f, size_t i, size_t j, size_t k, Axis a, double dx, double dy,
                double dz) {
    if (a == X)
        return Dx(f, i, j, k, dx);
    if (a == Y)
        return Dy(f, i, j, k, dy);
    return Dz(f, i, j, k, dz);
}

/**
 * @brief One-dimensional slice of the 6th-order Kreiss–Oliger filter.
 * @details Implements
 * \f$\mathcal{D}_6[f]_j = (f_{j-3} - 6f_{j-2} + 15 f_{j-1} - 20 f_j + 15 f_{j+1} - 6 f_{j+2} +
 * f_{j+3})/64\f$.
 */
template <typename Field>
inline double KO6_axis(const Field &f, size_t i, size_t j, size_t k, Axis axis) {
    auto sample = [&](int offset) {
        if (axis == X)
            return get(f, i + offset, j, k);
        if (axis == Y)
            return get(f, i, j + offset, k);
        return get(f, i, j, k + offset);
    };
    const double s = (sample(-3) - 6.0 * sample(-2) + 15.0 * sample(-1) - 20.0 * sample(0) +
                      15.0 * sample(1) - 6.0 * sample(2) + sample(3)) /
                     64.0;
    return s;
}

/**
 * @brief Dimension-summed Kreiss–Oliger dissipation scaled by \f$\sigma\f$.
 * @note RHS kernels typically set \f$\sigma=0.1\f$ following the module conventions.
 */

inline double &fd_dx() {
    static double v = 1.0;
    return v;
}

inline void set_fd_dx(double dx) { fd_dx() = dx; }

template <typename Field>
inline double KO6(const Field &f, size_t i, size_t j, size_t k, double sigma, double dx) {
    const double sum = KO6_axis(f, i, j, k, X) + KO6_axis(f, i, j, k, Y) + KO6_axis(f, i, j, k, Z);
    return (sigma / dx) * sum;
}

template <typename Field>
inline double KO6(const Field &f, size_t i, size_t j, size_t k, double sigma) {
    return KO6(f, i, j, k, sigma, fd_dx());
}

/**
 * @brief Third-order upwind derivative along x used for shift advection terms.
 * @param beta Sign of the advecting velocity (typically \f$\beta^x\f$) determines the biased
 * stencil.
 */
template <typename Field>
inline double Dx_upwind(const Field &f, size_t i, size_t j, size_t k, double dx, double beta) {
    if (beta >= 0.0)
        return (3.0 * get(f, i, j, k) - 4.0 * get(f, i - 1, j, k) + get(f, i - 2, j, k)) /
               (2.0 * dx);
    return (-3.0 * get(f, i, j, k) + 4.0 * get(f, i + 1, j, k) - get(f, i + 2, j, k)) / (2.0 * dx);
}

template <typename Field>
inline double Dy_upwind(const Field &f, size_t i, size_t j, size_t k, double dy, double beta) {
    if (beta >= 0.0)
        return (3.0 * get(f, i, j, k) - 4.0 * get(f, i, j - 1, k) + get(f, i, j - 2, k)) /
               (2.0 * dy);
    return (-3.0 * get(f, i, j, k) + 4.0 * get(f, i, j + 1, k) - get(f, i, j + 2, k)) / (2.0 * dy);
}

template <typename Field>
inline double Dz_upwind(const Field &f, size_t i, size_t j, size_t k, double dz, double beta) {
    if (beta >= 0.0)
        return (3.0 * get(f, i, j, k) - 4.0 * get(f, i, j, k - 1) + get(f, i, j, k - 2)) /
               (2.0 * dz);
    return (-3.0 * get(f, i, j, k) + 4.0 * get(f, i, j, k + 1) - get(f, i, j, k + 2)) / (2.0 * dz);
}

inline int sym6(int a, int b) {
    if (a == 0 && b == 0)
        return XX;
    if ((a == 0 && b == 1) || (a == 1 && b == 0))
        return XY;
    if ((a == 0 && b == 2) || (a == 2 && b == 0))
        return XZ;
    if (a == 1 && b == 1)
        return YY;
    if ((a == 1 && b == 2) || (a == 2 && b == 1))
        return YZ;
    return ZZ;
}

inline int g27(int up, int lo1, int lo2) { return up * 9 + lo1 * 3 + lo2; }

template <typename Grid, typename Field3>
inline void covariant_D_vec(const Grid &G, const Field3 Vfield[3], size_t i, size_t j, size_t k,
                            double DV[3][3], Variance in_var, Variance out_var) {
    const size_t id = G.alpha.idx(i, j, k);

    const double chi = G.chi.ptr()[id];
    const double inv_chi = 1.0 / chi;

    double tgamma[3][3], tginv[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            tgamma[a][b] = G.gamma_tilde[sym6(a, b)].ptr()[id];
            tginv[a][b] = G.gamma_tilde_inv[sym6(a, b)].ptr()[id];
        }

    const double dchi[3] = {Dx(G.chi, i, j, k, G.dx), Dy(G.chi, i, j, k, G.dy),
                            Dz(G.chi, i, j, k, G.dz)};

    double V_up[3], V_dn[3];
    if (in_var == Variance::Up) {
        V_up[0] = Vfield[0].ptr()[id];
        V_up[1] = Vfield[1].ptr()[id];
        V_up[2] = Vfield[2].ptr()[id];
        for (int a = 0; a < 3; ++a)
            V_dn[a] = inv_chi *
                      (tgamma[a][0] * V_up[0] + tgamma[a][1] * V_up[1] + tgamma[a][2] * V_up[2]);
    } else {
        V_dn[0] = Vfield[0].ptr()[id];
        V_dn[1] = Vfield[1].ptr()[id];
        V_dn[2] = Vfield[2].ptr()[id];
        for (int a = 0; a < 3; ++a)
            V_up[a] = chi * (tginv[a][0] * V_dn[0] + tginv[a][1] * V_dn[1] + tginv[a][2] * V_dn[2]);
    }

    double dV_in[3][3];
    for (int m = 0; m < 3; ++m)
        for (int a = 0; a < 3; ++a) {
            const auto  &f = Vfield[a];
            const double dv = (m == 0)   ? Dx(f, i, j, k, G.dx)
                              : (m == 1) ? Dy(f, i, j, k, G.dy)
                                         : Dz(f, i, j, k, G.dz);
            dV_in[m][a] = dv;
        }

    double Gamma_phys[3][3][3];
    for (int up = 0; up < 3; ++up)
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {

                const double Gt = G.Gamma_tilde[g27(up, lo1, lo2)].ptr()[id];

                const double term1 = (up == lo1) ? dchi[lo2] : 0.0;
                const double term2 = (up == lo2) ? dchi[lo1] : 0.0;

                double contracted = 0.0;
                for (int ell = 0; ell < 3; ++ell)
                    contracted += tginv[up][ell] * dchi[ell];

                const double term3 = tgamma[lo1][lo2] * contracted;

                const double C = -(0.5 / chi) * (term1 + term2 - term3);

                Gamma_phys[up][lo1][lo2] = Gt + C;
            }

    for (int m = 0; m < 3; ++m)
        for (int a = 0; a < 3; ++a)
            DV[m][a] = 0.0;

    if (in_var == Variance::Up && out_var == Variance::Up) {
        for (int m = 0; m < 3; ++m)
            for (int jidx = 0; jidx < 3; ++jidx) {
                double val = dV_in[m][jidx];
                for (int kidx = 0; kidx < 3; ++kidx)
                    val += Gamma_phys[jidx][m][kidx] * V_up[kidx];
                DV[m][jidx] = val;
            }
        return;
    }

    if (in_var == Variance::Down && out_var == Variance::Down) {
        for (int m = 0; m < 3; ++m)
            for (int jidx = 0; jidx < 3; ++jidx) {
                double val = dV_in[m][jidx];
                for (int kidx = 0; kidx < 3; ++kidx)
                    val -= Gamma_phys[kidx][m][jidx] * V_dn[kidx];
                DV[m][jidx] = val;
            }
        return;
    }

    if (in_var == Variance::Up && out_var == Variance::Down) {
        const double inv_chi2 = inv_chi * inv_chi;

        auto dgamma_phys = [&](int m, int a, int b) -> double {
            const auto  &F = G.gamma_tilde[sym6(a, b)];
            const double dgt = (m == 0)   ? Dx(F, i, j, k, G.dx)
                               : (m == 1) ? Dy(F, i, j, k, G.dy)
                                          : Dz(F, i, j, k, G.dz);
            const double gt = F.ptr()[id];
            return inv_chi * dgt - inv_chi2 * dchi[m] * gt;
        };

        for (int m = 0; m < 3; ++m)
            for (int jidx = 0; jidx < 3; ++jidx) {

                double partial = 0.0;
                for (int kidx = 0; kidx < 3; ++kidx) {
                    partial += dgamma_phys(m, jidx, kidx) * V_up[kidx];
                    partial += (inv_chi * tgamma[jidx][kidx]) * dV_in[m][kidx];
                }

                double corr = 0.0;
                for (int kidx = 0; kidx < 3; ++kidx)
                    corr += Gamma_phys[kidx][m][jidx] * V_dn[kidx];

                DV[m][jidx] = partial - corr;
            }
        return;
    }

    if (in_var == Variance::Down && out_var == Variance::Up) {
        double D_dn[3];
        for (int m = 0; m < 3; ++m) {
            for (int jidx = 0; jidx < 3; ++jidx) {
                double val = dV_in[m][jidx];
                for (int kidx = 0; kidx < 3; ++kidx)
                    val -= Gamma_phys[kidx][m][jidx] * V_dn[kidx];
                D_dn[jidx] = val;
            }
            for (int jidx = 0; jidx < 3; ++jidx) {
                const double gij = chi * tginv[jidx][0];
                (void)gij;
                DV[m][jidx] = chi * (tginv[jidx][0] * D_dn[0] + tginv[jidx][1] * D_dn[1] +
                                     tginv[jidx][2] * D_dn[2]);
            }
        }
        return;
    }
}

} // namespace tensorium_RG::fd
