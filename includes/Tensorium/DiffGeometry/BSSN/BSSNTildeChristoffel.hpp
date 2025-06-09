/**
 * @file TildeGamma.hpp
 * @brief Computes the contracted conformal Christoffel symbols \f$ \tilde{\Gamma}^i \f$
 *
 * This header defines the `TildeGamma` class used to compute the vector \f$ \tilde{\Gamma}^i \f$
 * from the conformal metric and its associated Christoffel symbols, as used in the BSSN
 * formulation.
 */

#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace tensorium_RG {
/**
 * @class TildeGamma
 * @brief Computes the contracted conformal Christoffel vector \f$ \tilde{\Gamma}^i \f$
 *
 * The BSSN formalism requires computing the vector:
 *
 * \f[
 * \tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
 * \f]
 *
 * where:
 * - \f$ \tilde{\gamma}^{jk} \f$ is the inverse of the conformal 3-metric
 * - \f$ \tilde{\Gamma}^i_{jk} \f$ are the Christoffel symbols computed from the conformal metric
 *
 * This quantity is key in the evolution of the conformal connection functions in the BSSN system.
 *
 * @tparam T Type used for floating-point computations (usually `double`)
 */

template <typename T> class TildeGamma {
  public:
    /**
     * @brief Computes \f$ \tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk} \f$
     *
     * This function contracts the conformal Christoffel symbols \f$ \tilde{\Gamma}^i_{jk} \f$
     * with the inverse conformal metric \f$ \tilde{\gamma}^{jk} \f$ to yield the vector:
     *
     * \f[
     * \tilde{\Gamma}^i = \tilde{\gamma}^{jk} \tilde{\Gamma}^i_{jk}
     * \f]
     *
     * @param gamma_tilde_inv A 3×3 tensor representing \f$ \tilde{\gamma}^{jk} \f$
     * @param christoffel A 3×3×3 tensor containing the Christoffel symbols \f$
     * \tilde{\Gamma}^i_{jk} \f$
     * @param tildeGamma Output vector of size 3, storing \f$ \tilde{\Gamma}^i \f$
     *
     * @throws std::invalid_argument if tensor dimensions are incorrect
     */
    static void compute(const tensorium::Tensor<T, 2> &gamma_tilde_inv,
                        const tensorium::Tensor<T, 3> &christoffel,
                        tensorium::Vector<T>          &tildeGamma) {
        if (gamma_tilde_inv.shape() != std::array<size_t, 2>{3, 3})
            throw std::invalid_argument("gamma_tilde_inv must be 3x3");
        if (christoffel.shape() != std::array<size_t, 3>{3, 3, 3})
            throw std::invalid_argument("christoffel must be 3x3x3");

        tildeGamma.resize(3);
        for (size_t i = 0; i < 3; ++i) {
            T sum = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    sum += gamma_tilde_inv(j, k) * christoffel(i, j, k);
                }
            }
            tildeGamma[i] = sum;
        }
    }
};
} // namespace tensorium_RG
