#pragma once

#include "../../../Core/Tensor.hpp"
#include "../../../Core/Vector.hpp"
#include "../Metric.hpp"
#include <cassert>
#include <cstddef>

namespace tensorium_RG {

/**
 * @brief Generate the conformal 3-metric field \f$ \tilde{\gamma}_{ij}(x^i) \f$ on a 3D grid.
 *
 * For each spatial grid point \f$ (x, y, z) \f$, the spatial metric \f$ \gamma_{ij} \f$ is
 * extracted from the given metric, the conformal factor \f$ \chi \f$ is computed, and the conformal
 * metric is defined as:
 * \f[
 * \tilde{\gamma}_{ij} = \chi \, \gamma_{ij}
 * \f]
 *
 * @tparam T Scalar type (typically `double`)
 * @param metric     A metric object providing `BSSN(X, α, β, γ)` and `compute_conformal_factor(γ)`
 * @param Nx,Ny,Nz   Grid resolution in x, y, z directions
 * @param dx,dy,dz   Grid spacings in x, y, z directions
 * @return A 5D tensor with shape \f$ (N_x, N_y, N_z, 3, 3) \f$ containing \f$ \tilde{\gamma}_{ij}
 * \f$ at each grid point
 */
template <typename T>
tensorium::Tensor<T, 5> generate_conformal_metric_field(const tensorium_RG::Metric<T> &metric,
                                                        size_t Nx, size_t Ny, size_t Nz, T dx, T dy,
                                                        T dz) {
    tensorium::Tensor<T, 5> gamma_tilde_field({Nx, Ny, Nz, 3, 3});

    tensorium::Vector<T> X(4); ///< Coordinates X^μ = (t, x, y, z)
    X(0) = T(0);               ///< Time is fixed at t = 0

    for (size_t i = 0; i < Nx; ++i) {
        T x = i * dx;
        for (size_t j = 0; j < Ny; ++j) {
            T y = j * dy;
            for (size_t k = 0; k < Nz; ++k) {
                T z = k * dz;

                X(1) = x;
                X(2) = y;
                X(3) = z;

                tensorium::Tensor<T, 2> gamma(3, 3), gamma_tilde(3, 3);
                tensorium::Vector<T>    beta(3);
                T                       alpha, chi;

                metric.BSSN(X, alpha, beta, gamma);
                chi = metric.compute_conformal_factor(gamma);
                metric.compute_conformal_metric(gamma, chi, gamma_tilde);

                for (size_t a = 0; a < 3; ++a)
                    for (size_t b = 0; b < 3; ++b)
                        gamma_tilde_field(i, j, k, a, b) = gamma_tilde(a, b);
            }
        }
    }

    return gamma_tilde_field;
}

} // namespace tensorium_RG
