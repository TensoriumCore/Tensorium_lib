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

#include "../../../Core/Derivate.hpp"
#include "../../../Core/Matrix.hpp"
#include "../../../Core/Tensor.hpp"
#include "../../../Core/Vector.hpp"
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
     * \partial_t \gamma_{ij} - \mathcal{L}_\beta \gamma_{ij}
     * \right), \qquad \mathcal{L}_\beta \gamma_{ij} = \nabla_i \beta_j + \nabla_j \beta_i
     * \f]
     *
     * The result is symmetrized to ensure \f$ K_{ij} = K_{ji} \f$.
     *
     * @param dgt Time derivative \f$ \partial_t \gamma_{ij} \f$
     * @param beta_cov Covariant shift components \f$ \beta_j \f$
     * @param partial_beta Covariant partial derivatives \f$ \partial_i \beta_j \f$
     * @param christoffel The 3D Christoffel symbols \f$ \Gamma^k_{ij} \f$
     * @param alpha The lapse function \f$ \alpha \f$
     * @return The extrinsic curvature tensor \f$ K_{ij} \f$
     */
    template <typename T>
    tensorium::Tensor<T, 2>
    compute_Kij(const tensorium::Tensor<T, 2> &dgt, const tensorium::Vector<T> &beta_cov,
                const tensorium::Tensor<T, 2> &partial_beta,
                const tensorium::Tensor<T, 3> &christoffel, const T alpha) {
        tensorium::Tensor<T, 2> Kij({3, 3});

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                T Di_bj = partial_beta(i, j);
                T Dj_bi = partial_beta(j, i);
                for (size_t k = 0; k < 3; ++k) {
                    Di_bj -= christoffel(k, i, j) * beta_cov(k);
                    Dj_bi -= christoffel(k, j, i) * beta_cov(k);
                }
                T Lie_ij = Di_bj + Dj_bi;
                Kij(i, j) = -0.5 / alpha * (dgt(i, j) - Lie_ij);
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
