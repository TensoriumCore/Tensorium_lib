/**
 * @file ExtrinsicCurvature.hpp
 * @brief Computes the extrinsic curvature tensor \f$ K_{ij} \f$ in the BSSN formalism.
 *
 * This file defines the class `ExtrinsicCurvature` used to compute the symmetric extrinsic
 * curvature tensor
 * \f$ K_{ij} \f$ from the time derivative of the spatial metric and the shift vector in the BSSN
 * decomposition.
 */

#pragma once

#include "../../Core/Derivate.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNSetup.hpp"

namespace tensorium_RG {
/**
 * @class ExtrinsicCurvature
 * @brief Computes the extrinsic curvature tensor \f$ K_{ij} \f$ from BSSN variables
 *
 * The extrinsic curvature is computed from the Lie derivative of the metric with respect
 * to the shift vector and the lapse function according to the BSSN equation:
 *
 * \f[
 * K_{ij} = -\frac{1}{2\alpha} \left( \partial_t \gamma_{ij}
 * - \nabla_i \beta_j - \nabla_j \beta_i \right)
 * \f]
 *
 * where:
 * - \f$ \gamma_{ij} \f$ is the spatial metric
 * - \f$ \alpha \f$ is the lapse function
 * - \f$ \beta^i \f$ is the shift vector
 * - \f$ \nabla_i \beta_j = \partial_i \beta_j - \Gamma^k_{ij} \beta_k \f$
 *
 * The implementation uses:
 * - `dgt` = \f$ \partial_t \gamma_{ij} \f$
 * - `partial_beta(i, j)` = \f$ \partial_i \beta_j \f$
 * - `christoffel(i, j, k)` = \f$ \Gamma^k_{ij} \f$
 *
 * @tparam K The scalar type used (e.g., double)
 */
template <typename K> class ExtrinsicCurvature {
  public:
    using Vec = tensorium::Vector<K>;
    using Mat = tensorium::Tensor<K, 2>;

    /**
     * @brief Computes the extrinsic curvature tensor \f$ K_{ij} \f$
     *
     * \f[
     * K_{ij} = -\frac{1}{2\alpha} \left(
     * \partial_t \gamma_{ij}
     * - (\partial_i \beta_j + \partial_j \beta_i - 2 \Gamma^k_{ij} \beta_k)
     * \right)
     * \f]
     *
     * The result is symmetrized to ensure \f$ K_{ij} = K_{ji} \f$.
     *
     * @param dgt Time derivative \f$ \partial_t \gamma_{ij} \f$
     * @param gamma The 3×3 spatial metric \f$ \gamma_{ij} \f$
     * @param beta The shift vector \f$ \beta^i \f$
     * @param partial_beta The partial derivatives \f$ \partial_i \beta_j \f$
     * @param christoffel The 3D Christoffel symbols \f$ \Gamma^k_{ij} \f$
     * @param alpha The lapse function \f$ \alpha \f$
     * @return The extrinsic curvature tensor \f$ K_{ij} \f$
     */
    template <typename T>
    tensorium::Tensor<T, 2>
    compute_Kij(const tensorium::Tensor<T, 2> &dgt, const tensorium::Tensor<T, 2> &gamma,
                const tensorium::Vector<T> &beta, const tensorium::Tensor<T, 2> &partial_beta,
                const tensorium::Tensor<T, 3> &christoffel, const T alpha) {
        tensorium::Tensor<T, 2> Kij({3, 3});

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                T sym_grad_beta = partial_beta(i, j) + partial_beta(j, i);
                T gamma_beta = T(0.0);

                for (size_t k = 0; k < 3; ++k)
                    gamma_beta += 2.0 * christoffel(k, i, j) * beta(k);

                Kij(i, j) = -0.5 / alpha * (dgt(i, j) - (sym_grad_beta - gamma_beta));
            }
        }
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = i + 1; j < 3; ++j) {
                T sym_val = 0.5 * (Kij(i, j) + Kij(j, i));
                Kij(i, j) = Kij(j, i) = sym_val;
            }

        return Kij;
    }
};

} // namespace tensorium_RG
