/**
 * @file BSSNRicciConformalTensor.hpp
 * @brief Computes the conformal Ricci tensor contributions from the conformal factor \f$ \chi \f$
 * in the BSSN formalism.
 */
#pragma once

#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNAutoDiff.hpp"
#include "BSSNChiContext.hpp"
#include "BSSNChristoffel.hpp"
#include "BSSNContractedChristoffel.hpp"
#include "BSSNDerivatives.hpp"

namespace tensorium_RG {

/**
 * @class RicciConformalTensor
 * @brief Provides methods to compute the \f$ \chi \f$-dependent part of the Ricci tensor in the
 * BSSN formalism.
 *
 * In the BSSN decomposition, the Ricci tensor \f$ R_{ij} \f$ is split into two parts:
 * \f[
 * R_{ij} = \tilde{R}_{ij} + R_{ij}^\chi
 * \f]
 * where \f$ \tilde{R}_{ij} \f$ is the Ricci tensor associated with the conformal metric \f$
 * \tilde{\gamma}_{ij} \f$, and \f$ R_{ij}^\chi \f$ contains all contributions from the conformal
 * factor \f$ \chi \f$:
 * \f[
 * R_{ij}^\chi = \frac{1}{\chi} \left( \nabla_i \nabla_j \chi - \Gamma^k_{ij} \partial_k \chi
 * \right)
 *              + \frac{1}{2 \chi} \tilde{\gamma}_{ij} \tilde{\nabla}^k \partial_k \chi
 *              - \frac{3}{2 \chi^2} \partial_i \chi \partial_j \chi
 *              + \frac{1}{2 \chi^2} \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} \partial_k \chi
 * \partial_l \chi
 * \f]
 *
 * This class evaluates each term independently and combines them.
 *
 * @tparam T Scalar type (typically float or double)
 */
template <typename T> class RicciConformalTensor {
  public:
    /**
     * @brief Computes the Hessian contribution:
     * \f[
     * \frac{1}{\chi} \left( \nabla_i \nabla_j \chi - \Gamma^k_{ij} \partial_k \chi \right)
     * \f]
     */
    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_Hessian(const ChiContext<T>           &chi_context,
                              const tensorium::Tensor<T, 2> &gamma_tilde,
                              const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                              const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;
        Tensor<T, 2> result({3, 3});

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T corrected_hessian = chi_context.hessian_chi(i, j);
                for (int k = 0; k < 3; ++k)
                    corrected_hessian -= christoffel_tilde(k, i, j) * chi_context.grad_chi(k);
                result(i, j) = corrected_hessian / chi_context.chi;
            }
        }

        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j) {
                T avg = 0.5 * (result(i, j) + result(j, i));
                result(i, j) = result(j, i) = avg;
            }

        return result;
    }
    /**
     * @brief Computes the Laplacian contribution:
     * \f[
     * \frac{1}{2 \chi} \tilde{\gamma}_{ij} \tilde{\nabla}^k \partial_k \chi
     * \f]
     */
    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_Laplacian(const ChiContext<T>           &chi_context,
                                const tensorium::Tensor<T, 2> &gamma_tilde,
                                const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                                const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const auto &hessian_chi = chi_context.hessian_chi;
        const T     chi = chi_context.chi;

        T laplacian_chi = 0.0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                T term = hessian_chi(k, l);
                for (int m = 0; m < 3; ++m)
                    term -= christoffel_tilde(m, k, l) * grad_chi(m);
                laplacian_chi += gamma_tilde_inv(k, l) * term;
            }
        }

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = 0.5 * gamma_tilde(i, j) * laplacian_chi / chi;

        return result;
    }

    /**
     * @brief Computes the squared gradient term:
     * \f[
     * -\frac{3}{2 \chi^2} \partial_i \chi \partial_j \chi
     * \f]
     */
    static tensorium::Tensor<T, 2> compute_Ricci_chi_GradGrad(const ChiContext<T> &chi_context) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const T     chi = chi_context.chi;

        const T inv_chi2 = 1.0 / (chi * chi);
        const T coeff = -1.5 * inv_chi2;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = coeff * grad_chi(i) * grad_chi(j);

        return result;
    }
    /**
     * @brief Computes the gradient norm term:
     * \f[
     * \frac{1}{2 \chi^2} \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} \partial_k \chi \partial_l \chi
     * \f]
     */
    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_GradNorm(const ChiContext<T>           &chi_context,
                               const tensorium::Tensor<T, 2> &gamma_tilde,
                               const tensorium::Tensor<T, 2> &gamma_tilde_inv) {
        using namespace tensorium;

        Tensor<T, 2> result({3, 3});

        const auto &grad_chi = chi_context.grad_chi;
        const T     chi = chi_context.chi;
        const T     inv_chi2 = 1.0 / (chi * chi);

        T grad_norm = 0.0;
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                grad_norm += gamma_tilde_inv(k, l) * grad_chi(k) * grad_chi(l);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = 0.5 * inv_chi2 * gamma_tilde(i, j) * grad_norm;

        return result;
    }
    /**
     * @brief Computes the total \f$ R^\chi_{ij} \f$ contribution as:
     * \f[
     * R_{ij}^\chi = \text{Hessian} + \text{Laplacian term} + \text{Gradient term} + \text{Gradient norm term}
     * \f]
     */
    static tensorium::Tensor<T, 2>
    compute_Ricci_chi_total(const ChiContext<T>           &chi_context,
                            const tensorium::Tensor<T, 2> &gamma_tilde,
                            const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                            const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> term1 =
            compute_Ricci_chi_Hessian(chi_context, gamma_tilde, gamma_tilde_inv, christoffel_tilde);
        Tensor<T, 2> term2 = compute_Ricci_chi_Laplacian(chi_context, gamma_tilde, gamma_tilde_inv,
                                                         christoffel_tilde);
        Tensor<T, 2> term3 = compute_Ricci_chi_GradGrad(chi_context);
        Tensor<T, 2> term4 = compute_Ricci_chi_GradNorm(chi_context, gamma_tilde, gamma_tilde_inv);

        Tensor<T, 2> result({3, 3});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j) = term1(i, j) + term2(i, j) + term3(i, j) + term4(i, j);

        return result;
    }
};
} // namespace tensorium_RG
