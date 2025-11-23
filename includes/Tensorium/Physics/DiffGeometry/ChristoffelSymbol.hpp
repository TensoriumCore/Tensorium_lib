#pragma once

#include "../../Backend/SIMD/Allocator.hpp"
#include "../../Backend/SIMD/SIMD.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../../Functional/Functional.hpp"
#include "../../Functional/FunctionnalRG.hpp"
#include "Metric.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tensorium_RG {
/**
 * @brief Stores and computes Christoffel symbols \f$ \Gamma^\lambda_{\mu\nu} \f$
 *
 * This class represents a tensor of Christoffel symbols for a 4D metric and supports:
 * - Storage in a flattened aligned vector
 * - Numerical computation from the metric via centered finite differences
 *
 * The Christoffel symbols are given by:
 * \f[
 * \Gamma^\lambda_{\mu\nu} =
 * \frac{1}{2} g^{\lambda\kappa} \left(
 * \partial_\mu g_{\nu\kappa} +
 * \partial_\nu g_{\mu\kappa} -
 * \partial_\kappa g_{\mu\nu}
 * \right)
 * \f]
 *
 * @tparam T Scalar type (e.g., float or double)
 */
template <typename T> class ChristoffelSym {
  public:
    aligned_vector<T>       data;
    static constexpr size_t rank = 4;
    size_t                  dim;
    /**
     * @brief Construct a Christoffel symbol tensor
     * @param dim Dimensionality of the space
     */
    ChristoffelSym(size_t dim) : dim(dim), data(dim * dim * dim * dim, T(0)) {}

    /**
     * @brief Mutable access to component \f$ \Gamma^\lambda_{\mu\nu} \f$
     */
    T &operator()(size_t i, size_t j, size_t k, size_t l) {
        assert(i < dim && j < dim && k < dim && l < dim);
        return data[i * dim * dim * dim + j * dim * dim + k * dim + l];
    }

    /**
     * @brief Const access to component \f$ \Gamma^\lambda_{\mu\nu} \f$
     */
    const T &operator()(size_t i, size_t j, size_t k, size_t l) const {
        return data[i * dim * dim * dim + j * dim * dim + k * dim + l];
    }
    /**
     * @brief Fill all components with a constant value
     * @param value Value to fill
     */
    void fill(T value) { std::fill(data.begin(), data.end(), value); }

    /**
     * @brief Print all non-zero Christoffel components to stdout
     */

    void print() const {
        for (size_t l = 0; l < dim; ++l) {
            if (l != 0)
                continue;
            std::cout << "Γ^" << l << "_{μν} :\n";
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    std::cout << std::setw(12) << std::setprecision(6) << std::fixed
                              << (*this)(l, i, j, 0) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    /**
     * @brief Compute Christoffel symbols numerically from a metric
     *
     * Uses centered finite differences on the metric tensor and contracts with \f$ g^{\mu\nu} \f$.
     *
     * @param X Coordinates \f$ X^\mu \f$
     * @param h Step size
     * @param g Metric tensor \f$ g_{\mu\nu} \f$
     * @param g_inv Inverse metric \f$ g^{\mu\nu} \f$
     * @param metric_generator A callable that generates \f$ g_{\mu\nu}(X) \f$
     * @return Christoffel symbol tensor \f$ \Gamma^\lambda_{\mu\nu} \f$
     */
    __attribute__((always_inline, hot, flatten)) static inline ChristoffelSym<T>
    compute_christoffel(const tensorium::Vector<T> &X, T h, const tensorium::Tensor<T, 2> &g,
                        const tensorium::Tensor<T, 2>                        &g_inv,
                        const std::function<void(const tensorium::Vector<T> &,
                                                 tensorium::Tensor<T, 2> &)> &metric_generator) {
        const size_t      dim = X.size();
        ChristoffelSym<T> gamma(dim);
        ChristoffelSym<T> d_metric(dim);
        gamma.fill(T(0));
        d_metric.fill(T(0));

        tensorium::Vector<T>    Xh = X, Xl = X;
        tensorium::Tensor<T, 2> gh({dim, dim}), gl({dim, dim});

        for (size_t mu = 0; mu < dim; ++mu) {
            Xh = X;
            Xl = X;
            Xh(mu) += h;
            Xl(mu) -= h;

            metric_generator(Xh, gh);
            metric_generator(Xl, gl);

            for (size_t nu = 0; nu < dim; ++nu)
                for (size_t lam = 0; lam < dim; ++lam)
                    d_metric(lam, nu, mu, 0) = (gh(lam, nu) - gl(lam, nu)) / (T(2) * h);
        }

        ChristoffelSym<T> tmp(dim);
        tmp.fill(T(0));
        for (size_t lam = 0; lam < dim; ++lam)
            for (size_t nu = 0; nu < dim; ++nu)
                for (size_t mu = 0; mu < dim; ++mu)
                    tmp(lam, nu, mu, 0) =
                        T(0.5) * (d_metric(nu, lam, mu, 0) + d_metric(mu, lam, nu, 0) -
                                  d_metric(mu, nu, lam, 0));

        for (size_t lam = 0; lam < dim; ++lam)
            for (size_t nu = 0; nu < dim; ++nu)
                for (size_t mu = 0; mu < dim; ++mu) {
                    T sum = T(0);
                    for (size_t kap = 0; kap < dim; ++kap)
                        sum += g_inv(lam, kap) * tmp(kap, nu, mu, 0);
                    gamma(lam, nu, mu, 0) = sum;
                }

        return gamma;
    }
};
/**
 * @brief Compute the inverse of a 2D metric tensor using local matrix inversion
 *
 * Converts the tensor to a matrix, computes the inverse, and converts back.
 *
 * @tparam T Scalar type
 * @param g Metric tensor \f$ g_{\mu\nu} \f$
 * @return Inverse tensor \f$ g^{\mu\nu} \f$
 */
template <typename T>
__attribute__((always_inline, hot, flatten)) inline tensorium::Tensor<T, 2>
inv_mat_tensor_local(const tensorium::Tensor<T, 2> &g) {
    const size_t         d0 = g.dimensions[0];
    const size_t         d1 = g.dimensions[1];
    tensorium::Matrix<T> mat({d0, d1});

    for (size_t i = 0; i < d0; ++i)
        for (size_t j = 0; j < d1; ++j)
            mat(i, j) = g(i, j);

    tensorium::Matrix<T> inv = mat.inverse();

    tensorium::Tensor<T, 2> out({d0, d1});
    for (size_t i = 0; i < d0; ++i)
        for (size_t j = 0; j < d1; ++j)
            out(i, j) = inv(i, j);

    return out;
}
/**
 * @brief Compute Christoffel symbols at an offset position along one direction
 *
 * Used in numerical relativity to evaluate derivatives of Christoffel symbols.
 *
 * @tparam T Scalar type
 * @param X Base coordinate
 * @param direction Axis of offset
 * @param offset Value to add (positive or negative)
 * @param h Finite difference step size
 * @param g Output metric tensor
 * @param g_inv Output inverse metric
 * @param Gamma_out Output Christoffel tensor
 * @param metric Metric object
 */
template <typename T>
__attribute__((always_inline, hot, flatten)) inline void
calculate_christoffel_at_offset(const tensorium::Vector<T> &X, int direction, T offset, T h,
                                tensorium::Tensor<T, 2> &g, tensorium::Tensor<T, 2> &g_inv,
                                tensorium_RG::ChristoffelSym<T> &Gamma_out,
                                const tensorium_RG::Metric<T>   &metric) {
    tensorium::Vector<T> X_offset = X;
    X_offset(direction) += offset;

    metric(X_offset, g);
    g_inv = tensorium_RG::inv_mat_tensor_local(g);

    Gamma_out = tensorium_RG::ChristoffelSym<T>::compute_christoffel(X_offset, h, g, g_inv, metric);
}
} // namespace tensorium_RG
