#pragma once

#include "../../Core/Derivate.hpp"
#include "../../Core/Matrix.hpp"
#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "../Metric.hpp"
#include "BSSNSetup.hpp"

namespace tensorium_RG {
/**
 * @class BSSNAtildeTensor
 * @brief Computes the trace-free conformal extrinsic curvature tensor \f$ \tilde{A}_{ij} \f$ in the
 * BSSN formalism.
 *
 * This tensor is defined as:
 * \f[
 * \tilde{A}_{ij} = \chi \left( K_{ij} - \frac{1}{3} \gamma_{ij} K \right)
 * \f]
 * where:
 * - \f$ K_{ij} \f$ is the extrinsic curvature tensor
 * - \f$ K = \gamma^{ij} K_{ij} \f$ is its trace
 * - \f$ \chi \f$ is the conformal factor
 * - \f$ \gamma_{ij} \f$ is the spatial 3-metric
 */
template <typename K> class BSSNAtildeTensor {
  public:
    using Vec = tensorium::Vector<K>;
    using Mat = tensorium::Tensor<K, 2>;
    /**
     * @brief Computes \f$ \tilde{A}_{ij} \f$ from \f$ K_{ij} \f$, \f$ \gamma_{ij} \f$, and \f$ \chi
     * \f$.
     *
     * \f[
     * \tilde{A}_{ij} = \chi \left( K_{ij} - \frac{1}{3} \gamma_{ij} K \right), \quad
     * K = \gamma^{ij} K_{ij}
     * \f]
     *
     * @param Kij        Extrinsic curvature tensor \f$ K_{ij} \f$
     * @param gamma_inv  Inverse spatial metric \f$ \gamma^{ij} \f$
     * @param gamma      Spatial metric \f$ \gamma_{ij} \f$
     * @param chi        Conformal factor \f$ \chi \f$
     * @return Trace-free conformal extrinsic curvature tensor \f$ \tilde{A}_{ij} \f$
     */
    tensorium::Tensor<K, 2> compute_Atilde_tensor(const tensorium::Tensor<K, 2> &Kij,
                                                  const tensorium::Tensor<K, 2> &gamma_inv,
                                                  const tensorium::Tensor<K, 2> &gamma, K chi) {
        K trace_K = 0.;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                trace_K += gamma_inv(i, j) * Kij(i, j);

        tensorium::Tensor<K, 2> Atilde({3, 3});
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                Atilde(i, j) = chi * (Kij(i, j) - (1. / 3.) * gamma(i, j) * trace_K);

        // Ensure symmetry: \f$ \tilde{A}_{ij} = \tilde{A}_{ji} \f$
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = i + 1; j < 3; ++j) {
                K sym = (Atilde(i, j) + Atilde(j, i)) * 0.5;
                Atilde(i, j) = Atilde(j, i) = sym;
            }

        return Atilde;
    }
};

} // namespace tensorium_RG
