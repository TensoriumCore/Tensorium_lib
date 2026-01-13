
#pragma once
#include <cstddef>

/**
 * @file BSSNConformal.hpp
 * @brief Helpers that convert conformal-factor gradients between \f$\chi\f$ and \f$\phi\f$.
 * @details
 * The BSSN module stores \f$\chi=e^{-4\phi}\f$.  When constraint or Ricci computations require
 * \f$\nabla\phi\f$ or \f$\nabla(6\phi)\f$, these helpers differentiate \f$\phi=-(1/4)\ln\chi\f$ and reuse
 * the existing \f$\partial_i\chi\f$ samples instead of recomputing logarithms.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Convert \f$\partial_i\chi\f$ to \f$\partial_i\phi\f$ using \f$\phi=-(1/4)\ln\chi\f$.
 * @details The relation is \f$\partial_i\phi = -(1/4)\chi^{-1}\partial_i\chi\f.
 * @param dchi Cartesian gradient of \f$\chi\f$.
 * @param chi Scalar conformal factor.
 * @param[out] dphi Populated with \f$\partial_i\phi\f$.
 */
template <typename T> inline void grad_phi_from_chi(const T dchi[3], T chi, T dphi[3]) noexcept {
    const T inv_chi = T(1) / chi;
    const T c = T(-0.25) * inv_chi;
    dphi[0] = c * dchi[0];
    dphi[1] = c * dchi[1];
    dphi[2] = c * dchi[2];
}

/**
 * @brief Compute \f$\partial_i(6\phi) = -\tfrac{3}{2}\chi^{-1}\partial_i\chi\f$ for the momentum constraint.
 */
template <typename T> inline void grad_6phi_from_chi(const T dchi[3], T chi, T d6phi[3]) noexcept {
    const T inv_chi = T(1) / chi;
    const T c = T(-1.5) * inv_chi;
    d6phi[0] = c * dchi[0];
    d6phi[1] = c * dchi[1];
    d6phi[2] = c * dchi[2];
}

} // namespace tensorium_RG::bssn
