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

inline double get_offset(const double *ptr, ptrdiff_t offset) { return ptr[offset]; }

// First Derivatives (Order 4)
template <typename T> inline double Dx_ptr(const T *p, ptrdiff_t sx, double inv_12dx) {
    return (-p[2 * sx] + 8.0 * p[sx] - 8.0 * p[-sx] + p[-2 * sx]) * inv_12dx;
}

template <typename T> inline double Dy_ptr(const T *p, ptrdiff_t sy, double inv_12dy) {
    return (-p[2 * sy] + 8.0 * p[sy] - 8.0 * p[-sy] + p[-2 * sy]) * inv_12dy;
}

template <typename T> inline double Dz_ptr(const T *p, double inv_12dz) {
    // Stride Z is always 1 in this layout
    return (-p[2] + 8.0 * p[1] - 8.0 * p[-1] + p[-2]) * inv_12dz;
}

// Second Derivatives (Order 4)
template <typename T> inline double Dxx_ptr(const T *p, ptrdiff_t sx, double inv_12dx2) {
    return (-p[2 * sx] + 16.0 * p[sx] - 30.0 * p[0] + 16.0 * p[-sx] - p[-2 * sx]) * inv_12dx2;
}

template <typename T> inline double Dyy_ptr(const T *p, ptrdiff_t sy, double inv_12dy2) {
    return (-p[2 * sy] + 16.0 * p[sy] - 30.0 * p[0] + 16.0 * p[-sy] - p[-2 * sy]) * inv_12dy2;
}

template <typename T> inline double Dzz_ptr(const T *p, double inv_12dz2) {
    return (-p[2] + 16.0 * p[1] - 30.0 * p[0] + 16.0 * p[-1] - p[-2]) * inv_12dz2;
}

// Mixed Derivatives (Order 2 - Compact)
template <typename T>
inline double Dxy_ptr(const T *p, ptrdiff_t sx, ptrdiff_t sy, double inv_4dxdy) {
    return (p[sx + sy] - p[sx - sy] - p[-sx + sy] + p[-sx - sy]) * inv_4dxdy;
}

template <typename T> inline double Dxz_ptr(const T *p, ptrdiff_t sx, double inv_4dxdz) {
    return (p[sx + 1] - p[sx - 1] - p[-sx + 1] + p[-sx - 1]) * inv_4dxdz;
}

template <typename T> inline double Dyz_ptr(const T *p, ptrdiff_t sy, double inv_4dydz) {
    return (p[sy + 1] - p[sy - 1] - p[-sy + 1] + p[-sy - 1]) * inv_4dydz;
}

// Mixed Derivatives (Order 4)
template <typename T>
inline double Dxy4_ptr(const T *p, ptrdiff_t sx, ptrdiff_t sy, double inv_144dxdy) {
    double          sum = 0.0;
    const ptrdiff_t idx_x[4] = {-2 * sx, -sx, sx, 2 * sx};
    const ptrdiff_t idx_y[4] = {-2 * sy, -sy, sy, 2 * sy};
    const double    w[4] = {-1.0, 8.0, -8.0, 1.0};

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * p[idx_x[a] + idx_y[b]];
        }
    }
    return sum * inv_144dxdy;
}

template <typename T> inline double Dxz4_ptr(const T *p, ptrdiff_t sx, double inv_144dxdz) {
    double          sum = 0.0;
    const ptrdiff_t idx_x[4] = {-2 * sx, -sx, sx, 2 * sx};
    const ptrdiff_t idx_z[4] = {-2, -1, 1, 2};
    const double    w[4] = {-1.0, 8.0, -8.0, 1.0};

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * p[idx_x[a] + idx_z[b]];
        }
    }
    return sum * inv_144dxdz;
}

template <typename T> inline double Dyz4_ptr(const T *p, ptrdiff_t sy, double inv_144dydz) {
    double          sum = 0.0;
    const ptrdiff_t idx_y[4] = {-2 * sy, -sy, sy, 2 * sy};
    const ptrdiff_t idx_z[4] = {-2, -1, 1, 2};
    const double    w[4] = {-1.0, 8.0, -8.0, 1.0};

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            sum += w[a] * w[b] * p[idx_y[a] + idx_z[b]];
        }
    }
    return sum * inv_144dydz;
}

// Upwind Derivatives
template <typename T>
inline double Dx_upwind_ptr(const T *p, ptrdiff_t sx, double inv_2dx, double beta) {
    if (beta >= 0.0)
        return (3.0 * p[0] - 4.0 * p[-sx] + p[-2 * sx]) * inv_2dx;
    return (-3.0 * p[0] + 4.0 * p[sx] - p[2 * sx]) * inv_2dx;
}

template <typename T>
inline double Dy_upwind_ptr(const T *p, ptrdiff_t sy, double inv_2dy, double beta) {
    if (beta >= 0.0)
        return (3.0 * p[0] - 4.0 * p[-sy] + p[-2 * sy]) * inv_2dy;
    return (-3.0 * p[0] + 4.0 * p[sy] - p[2 * sy]) * inv_2dy;
}

template <typename T> inline double Dz_upwind_ptr(const T *p, double inv_2dz, double beta) {
    if (beta >= 0.0)
        return (3.0 * p[0] - 4.0 * p[-1] + p[-2]) * inv_2dz;
    return (-3.0 * p[0] + 4.0 * p[1] - p[2]) * inv_2dz;
}

// KO6 Dissipation
template <typename T> inline double KO6_axis_ptr(const T *p, ptrdiff_t stride) {
    return (p[-3 * stride] - 6.0 * p[-2 * stride] + 15.0 * p[-stride] - 20.0 * p[0] +
            15.0 * p[stride] - 6.0 * p[2 * stride] + p[3 * stride]) *
           (1.0 / 64.0);
}

/// @brief Utility accessor that forwards to `Field3D::idx` / raw pointer storage.
template <typename Field> inline double get(const Field &f, size_t i, size_t j, size_t k) {
    return f.ptr()[f.idx(i, j, k)];
}

template <typename Field>
inline double Dx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    return Dx_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, 1.0 / (12.0 * dx));
}

template <typename Field>
inline double Dy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    return Dy_ptr(f.ptr() + f.idx(i, j, k), f.st.sy, 1.0 / (12.0 * dy));
}

template <typename Field>
inline double Dz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    return Dz_ptr(f.ptr() + f.idx(i, j, k), 1.0 / (12.0 * dz));
}

template <typename Field>
inline double Dxx(const Field &f, size_t i, size_t j, size_t k, double dx) {
    return Dxx_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, 1.0 / (12.0 * dx * dx));
}

template <typename Field>
inline double Dyy(const Field &f, size_t i, size_t j, size_t k, double dy) {
    return Dyy_ptr(f.ptr() + f.idx(i, j, k), f.st.sy, 1.0 / (12.0 * dy * dy));
}

template <typename Field>
inline double Dzz(const Field &f, size_t i, size_t j, size_t k, double dz) {
    return Dzz_ptr(f.ptr() + f.idx(i, j, k), 1.0 / (12.0 * dz * dz));
}

template <typename Field>
inline double Dxy(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    return Dxy_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, f.st.sy, 1.0 / (4.0 * dx * dy));
}

template <typename Field>
inline double Dxz(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    return Dxz_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, 1.0 / (4.0 * dx * dz));
}

template <typename Field>
inline double Dyz(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    return Dyz_ptr(f.ptr() + f.idx(i, j, k), f.st.sy, 1.0 / (4.0 * dy * dz));
}

template <typename Field>
inline double Dxy4(const Field &f, size_t i, size_t j, size_t k, double dx, double dy) {
    return Dxy4_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, f.st.sy, 1.0 / (144.0 * dx * dy));
}

template <typename Field>
inline double Dxz4(const Field &f, size_t i, size_t j, size_t k, double dx, double dz) {
    return Dxz4_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, 1.0 / (144.0 * dx * dz));
}

template <typename Field>
inline double Dyz4(const Field &f, size_t i, size_t j, size_t k, double dy, double dz) {
    return Dyz4_ptr(f.ptr() + f.idx(i, j, k), f.st.sy, 1.0 / (144.0 * dy * dz));
}

template <typename Field>
inline double D(const Field &f, size_t i, size_t j, size_t k, Axis a, double dx, double dy,
                double dz) {
    if (a == X)
        return Dx(f, i, j, k, dx);
    if (a == Y)
        return Dy(f, i, j, k, dy);
    return Dz(f, i, j, k, dz);
}

template <typename Field>
inline double KO6_axis(const Field &f, size_t i, size_t j, size_t k, Axis axis) {
    const auto *p = f.ptr() + f.idx(i, j, k);
    if (axis == X)
        return KO6_axis_ptr(p, f.st.sx);
    if (axis == Y)
        return KO6_axis_ptr(p, f.st.sy);
    return KO6_axis_ptr(p, ptrdiff_t(1));
}

inline double &fd_dx() {
    static double v = 1.0;
    return v;
}

inline void set_fd_dx(double dx) { fd_dx() = dx; }

template <typename Field>
inline double KO6(const Field &f, size_t i, size_t j, size_t k, double sigma, double dx) {
    const auto  *p = f.ptr() + f.idx(i, j, k);
    const double sum =
        KO6_axis_ptr(p, f.st.sx) + KO6_axis_ptr(p, f.st.sy) + KO6_axis_ptr(p, ptrdiff_t(1));
    return (sigma / dx) * sum;
}

template <typename Field>
inline double KO6(const Field &f, size_t i, size_t j, size_t k, double sigma) {
    return KO6(f, i, j, k, sigma, fd_dx());
}

template <typename Field>
inline double Dx_upwind(const Field &f, size_t i, size_t j, size_t k, double dx, double beta) {
    return Dx_upwind_ptr(f.ptr() + f.idx(i, j, k), f.st.sx, 1.0 / (2.0 * dx), beta);
}

template <typename Field>
inline double Dy_upwind(const Field &f, size_t i, size_t j, size_t k, double dy, double beta) {
    return Dy_upwind_ptr(f.ptr() + f.idx(i, j, k), f.st.sy, 1.0 / (2.0 * dy), beta);
}

template <typename Field>
inline double Dz_upwind(const Field &f, size_t i, size_t j, size_t k, double dz, double beta) {
    return Dz_upwind_ptr(f.ptr() + f.idx(i, j, k), 1.0 / (2.0 * dz), beta);
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

// =========================================================
// Complex Kernels (Optimized to reuse pointers)
// =========================================================

template <typename Grid, typename Field3>
inline void covariant_D_vec(const Grid &G, const Field3 Vfield[3], size_t i, size_t j, size_t k,
                            double DV[3][3], Variance in_var, Variance out_var) {
    const size_t id = G.alpha.idx(i, j, k);

    // Optimization: Pre-calculate inverses
    const double    inv_12dx = 1.0 / (12.0 * G.dx);
    const double    inv_12dy = 1.0 / (12.0 * G.dy);
    const double    inv_12dz = 1.0 / (12.0 * G.dz);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

    const double chi = G.chi.ptr()[id];
    const double inv_chi = 1.0 / chi;

    double tgamma[3][3], tginv[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            tgamma[a][b] = G.gamma_tilde[sym6(a, b)].ptr()[id];
            tginv[a][b] = G.gamma_tilde_inv[sym6(a, b)].ptr()[id];
        }

    const double *p_chi = G.chi.ptr() + id;
    const double  dchi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                             Dz_ptr(p_chi, inv_12dz)};

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
            const auto  *p_v = Vfield[a].ptr() + id;
            const double dv = (m == 0)   ? Dx_ptr(p_v, sx, inv_12dx)
                              : (m == 1) ? Dy_ptr(p_v, sy, inv_12dy)
                                         : Dz_ptr(p_v, inv_12dz);
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

        for (int m = 0; m < 3; ++m)
            for (int jidx = 0; jidx < 3; ++jidx) {
                double partial = 0.0;
                for (int kidx = 0; kidx < 3; ++kidx) {
                    // Inline dgamma_phys logic using pointers
                    const auto  *p_gam = G.gamma_tilde[sym6(jidx, kidx)].ptr() + id;
                    const double gt = *p_gam;
                    const double dgt = (m == 0)   ? Dx_ptr(p_gam, sx, inv_12dx)
                                       : (m == 1) ? Dy_ptr(p_gam, sy, inv_12dy)
                                                  : Dz_ptr(p_gam, inv_12dz);

                    const double dgamma_val = inv_chi * dgt - inv_chi2 * dchi[m] * gt;

                    partial += dgamma_val * V_up[kidx];
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
                DV[m][jidx] = chi * (tginv[jidx][0] * D_dn[0] + tginv[jidx][1] * D_dn[1] +
                                     tginv[jidx][2] * D_dn[2]);
            }
        }
        return;
    }
}

} // namespace tensorium_RG::fd
