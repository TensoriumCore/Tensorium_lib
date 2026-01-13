
#pragma once

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
 * Second derivatives follow the standard \f$(-1,16,-30,16,-1)/12\f$ stencil, and mixed derivatives use
 * either the classical 2nd-order cross operator (`Dxy`, `Dxz`, `Dyz`) or the 4th-order tensor-product
 * version (`Dxy4`, `Dxz4`, `Dyz4`) when Ricci evaluations demand higher accuracy.  Upwinded
 * advection relies on biased 3-point formulas activated according to the sign of the local shift.
 * Kreiss–Oliger dissipation implements the 6th-order filter used to stabilize high-frequency
 * numerical noise in hyperbolic systems.
 *
 * @note These operators assume uniform spacing along each axis and share the units and conventions
 * from @ref BSSN_Grid.  Callers are expected to clamp loop bounds so the stencil footprints never
 * cross uninitialized halos.
 */

namespace tensorium_RG::fd {

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
 *          \f$144\,\Delta x\,\Delta y\f$ so Ricci builders can reuse it for the \f$\tilde{R}_{ij}\f$ terms.
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
 * \f$\mathcal{D}_6[f]_j = (f_{j-3} - 6f_{j-2} + 15 f_{j-1} - 20 f_j + 15 f_{j+1} - 6 f_{j+2} + f_{j+3})/64\f$.
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
template <typename Field>
inline double KO6(const Field &f, size_t i, size_t j, size_t k, double sigma) {
    const double sum = KO6_axis(f, i, j, k, X) + KO6_axis(f, i, j, k, Y) + KO6_axis(f, i, j, k, Z);
    return sigma * sum;
}

/**
 * @brief Third-order upwind derivative along x used for shift advection terms.
 * @param beta Sign of the advecting velocity (typically \f$\beta^x\f$) determines the biased stencil.
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

} // namespace tensorium_RG::fd
