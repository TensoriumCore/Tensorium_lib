#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <cmath>
#include <limits>

/**
 * @file BSSNAlgebraic.hpp
 * @brief Determinant renormalization and trace-free projection used to keep the Z4c system stable.
 */

namespace tensorium_RG::bssn {

namespace algebraic_detail {
inline double det3(double gxx, double gxy, double gxz, double gyy, double gyz, double gzz) {
    return gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
           gxz * (gxy * gyz - gxz * gyy);
}
} // namespace algebraic_detail

template <typename T> inline void enforce_algebraic_constraints(BSSNGridSoA<T> &grid) {
    const size_t total = grid.gamma_tilde[XX].st.nx_tot * grid.gamma_tilde[XX].st.ny_tot *
                         grid.gamma_tilde[XX].st.nz_tot;

    T *gxx_ptr = grid.gamma_tilde[XX].ptr();
    T *gxy_ptr = grid.gamma_tilde[XY].ptr();
    T *gxz_ptr = grid.gamma_tilde[XZ].ptr();
    T *gyy_ptr = grid.gamma_tilde[YY].ptr();
    T *gyz_ptr = grid.gamma_tilde[YZ].ptr();
    T *gzz_ptr = grid.gamma_tilde[ZZ].ptr();

    T *gixx_ptr = grid.gamma_tilde_inv[XX].ptr();
    T *gixy_ptr = grid.gamma_tilde_inv[XY].ptr();
    T *gixz_ptr = grid.gamma_tilde_inv[XZ].ptr();
    T *giyy_ptr = grid.gamma_tilde_inv[YY].ptr();
    T *giyz_ptr = grid.gamma_tilde_inv[YZ].ptr();
    T *gizz_ptr = grid.gamma_tilde_inv[ZZ].ptr();

    T *Axx_ptr = grid.A_tilde[XX].ptr();
    T *Axy_ptr = grid.A_tilde[XY].ptr();
    T *Axz_ptr = grid.A_tilde[XZ].ptr();
    T *Ayy_ptr = grid.A_tilde[YY].ptr();
    T *Ayz_ptr = grid.A_tilde[YZ].ptr();
    T *Azz_ptr = grid.A_tilde[ZZ].ptr();

#pragma omp parallel for
    for (size_t idx = 0; idx < total; ++idx) {
        double gxx = static_cast<double>(gxx_ptr[idx]);
        double gxy = static_cast<double>(gxy_ptr[idx]);
        double gxz = static_cast<double>(gxz_ptr[idx]);
        double gyy = static_cast<double>(gyy_ptr[idx]);
        double gyz = static_cast<double>(gyz_ptr[idx]);
        double gzz = static_cast<double>(gzz_ptr[idx]);

        double det = algebraic_detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
        if (det <= std::numeric_limits<double>::min())
            det = std::numeric_limits<double>::min();
        const double renorm = 1.0 / std::cbrt(det);
        gxx *= renorm;
        gxy *= renorm;
        gxz *= renorm;
        gyy *= renorm;
        gyz *= renorm;
        gzz *= renorm;

        const double det_scaled = algebraic_detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
        const double inv_det =
            (det_scaled > std::numeric_limits<double>::min()) ? (1.0 / det_scaled) : 0.0;

        const double gixx = (gyy * gzz - gyz * gyz) * inv_det;
        const double gixy = (gxz * gyz - gxy * gzz) * inv_det;
        const double gixz = (gxy * gyz - gxz * gyy) * inv_det;
        const double giyy = (gxx * gzz - gxz * gxz) * inv_det;
        const double giyz = (gxy * gxz - gxx * gyz) * inv_det;
        const double gizz = (gxx * gyy - gxy * gxy) * inv_det;

        double Axx = static_cast<double>(Axx_ptr[idx]);
        double Axy = static_cast<double>(Axy_ptr[idx]);
        double Axz = static_cast<double>(Axz_ptr[idx]);
        double Ayy = static_cast<double>(Ayy_ptr[idx]);
        double Ayz = static_cast<double>(Ayz_ptr[idx]);
        double Azz = static_cast<double>(Azz_ptr[idx]);

        const double trace = gixx * Axx + giyy * Ayy + gizz * Azz +
                             2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
        const double one_third_trace = trace / 3.0;
        Axx -= gxx * one_third_trace;
        Axy -= gxy * one_third_trace;
        Axz -= gxz * one_third_trace;
        Ayy -= gyy * one_third_trace;
        Ayz -= gyz * one_third_trace;
        Azz -= gzz * one_third_trace;

        gxx_ptr[idx] = static_cast<T>(gxx);
        gxy_ptr[idx] = static_cast<T>(gxy);
        gxz_ptr[idx] = static_cast<T>(gxz);
        gyy_ptr[idx] = static_cast<T>(gyy);
        gyz_ptr[idx] = static_cast<T>(gyz);
        gzz_ptr[idx] = static_cast<T>(gzz);

        gixx_ptr[idx] = static_cast<T>(gixx);
        gixy_ptr[idx] = static_cast<T>(gixy);
        gixz_ptr[idx] = static_cast<T>(gixz);
        giyy_ptr[idx] = static_cast<T>(giyy);
        giyz_ptr[idx] = static_cast<T>(giyz);
        gizz_ptr[idx] = static_cast<T>(gizz);

        Axx_ptr[idx] = static_cast<T>(Axx);
        Axy_ptr[idx] = static_cast<T>(Axy);
        Axz_ptr[idx] = static_cast<T>(Axz);
        Ayy_ptr[idx] = static_cast<T>(Ayy);
        Ayz_ptr[idx] = static_cast<T>(Ayz);
        Azz_ptr[idx] = static_cast<T>(Azz);
    }

    const T alpha_floor = T(1e-4);
    const T chi_floor = T(1e-4);
    T      *alpha_ptr = grid.alpha.ptr();
    T      *chi_ptr = grid.chi.ptr();
#pragma omp parallel for
    for (size_t idx = 0; idx < total; ++idx) {
        if (alpha_ptr[idx] < alpha_floor)
            alpha_ptr[idx] = alpha_floor;
        if (chi_ptr[idx] < chi_floor)
            chi_ptr[idx] = chi_floor;
    }
}

} // namespace tensorium_RG::bssn
