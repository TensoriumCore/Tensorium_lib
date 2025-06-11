/**
 * @file BSSNRicciPhysicalTensor.hpp
 * @brief Computes the full physical Ricci tensor \f$ R_{ij} \f$ from BSSN variables.
 */

#pragma once

#include "../../Core/Tensor.hpp"
#include "../../Core/Vector.hpp"
#include "BSSNRicciConformalTensor.hpp"
#include "BSSNRicciTildeTensor.hpp"

namespace tensorium_RG {

/**
 * @class RicciPhysicalTensor
 * @brief Computes the physical 3-Ricci tensor \f$ R_{ij} \f$ as the sum of conformal and conformal-factor contributions.
 *
 * In the BSSN formulation of general relativity, the physical Ricci tensor \f$ R_{ij} \f$ is decomposed as:
 * \f[
 * R_{ij} = \tilde{R}_{ij} + R^{\chi}_{ij}
 * \f]
 * where:
 *
 * - \f$ \tilde{R}_{ij} \f$ is the Ricci tensor of the conformal metric \f$ \tilde{\gamma}_{ij} \f$
 * - \f$ R^{\chi}_{ij} \f$ encodes the contributions from the conformal factor \f$ \chi \f$
 *
 * Expanding all terms in partial derivatives:
 *
 * \f[
 * R_{ij} =
 * -\frac{1}{2} \tilde{\gamma}^{kl} \partial_k \partial_l \tilde{\gamma}_{ij}
 * + \frac{1}{2} \left( \partial_i \tilde{\Gamma}_j + \partial_j \tilde{\Gamma}_i \right)
 * + \frac{1}{2} \tilde{\Gamma}^k \left( \tilde{\gamma}_{ki} \tilde{\Gamma}^l_{jl} + \tilde{\gamma}_{kj} \tilde{\Gamma}^l_{il} \right)
 * + \tilde{\gamma}^{\ell m} \left( 2 \tilde{\Gamma}^k_{\ell(i} \tilde{\Gamma}_{j)km} + \tilde{\Gamma}^k_{im} \tilde{\Gamma}_{k\ell j} \right)
 * + \frac{1}{\chi} \left( \partial_i \partial_j \chi - \tilde{\Gamma}^k_{ij} \partial_k \chi \right)
 * + \frac{1}{2\chi} \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} \left( \partial_k \partial_l \chi - \tilde{\Gamma}^m_{kl} \partial_m \chi \right)
 * - \frac{3}{2\chi^2} \partial_i \chi \partial_j \chi
 * + \frac{1}{2\chi^2} \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} \partial_k \chi \partial_l \chi
 * \f]
 *
 * This class computes each term independently and sums them to obtain the total physical Ricci tensor.
 *
 * @tparam T Scalar type (e.g., float or double)
 */
template <typename T> class RicciPhysicalTensor {
  public:
    /**
     * @brief Compute the full Ricci tensor \f$ R_{ij} \f$ as the sum of \f$ \tilde{R}_{ij} \f$ and \f$ R^\chi_{ij} \f$.
     *
     * This method calls:
     * - RicciTildeTensor::compute_Ricci_Tilde_tensor()
     * - RicciConformalTensor::compute_Ricci_chi_total()
     *
     * and returns:
     * \f[
     * R_{ij} = \tilde{R}_{ij} + R^{\chi}_{ij}
     * \f]
     *
     * Also prints the Frobenius norm of \f$ R_{ij} \f$ for diagnostics.
     *
     * @param chi_context          ChiContext object containing metric, grid spacing, and position
     * @param gamma_tilde          Conformal metric \f$ \tilde{\gamma}_{ij} \f$
     * @param gamma_tilde_inv      Inverse conformal metric \f$ \tilde{\gamma}^{ij} \f$
     * @param tilde_Gamma          Contracted conformal Christoffel symbols \f$ \tilde{\Gamma}^i \f$
     * @param christoffel_tilde    Full Christoffel symbols \f$ \tilde{\Gamma}^i_{jk} \f$
     * @return Physical Ricci tensor \f$ R_{ij} \f$
     */
    static tensorium::Tensor<T, 2> compute_Ricci_total(
        const ChiContext<T> &chi_context, const tensorium::Tensor<T, 2> &gamma_tilde,
        const tensorium::Tensor<T, 2> &gamma_tilde_inv, const tensorium::Vector<T> &tilde_Gamma,
        const tensorium::Tensor<T, 3> &christoffel_tilde) {
        using namespace tensorium;

        Tensor<T, 2> Ricci_tilde = RicciTildeTensor<T>::compute_Ricci_Tilde_tensor(
            chi_context, gamma_tilde_inv, tilde_Gamma, christoffel_tilde, gamma_tilde);

        Tensor<T, 2> Ricci_chi = RicciConformalTensor<T>::compute_Ricci_chi_total(
            chi_context, gamma_tilde, gamma_tilde_inv, christoffel_tilde);

        Tensor<T, 2> Ricci_phys({3, 3});
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Ricci_phys(i, j) = Ricci_tilde(i, j) + Ricci_chi(i, j);

        T norm = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                norm += Ricci_phys(i, j) * Ricci_phys(i, j);

        norm = std::sqrt(norm);
        std::cout << "||R_phys||_F = " << norm << std::endl;

        return Ricci_phys;
    }
};

} // namespace tensorium_RG
