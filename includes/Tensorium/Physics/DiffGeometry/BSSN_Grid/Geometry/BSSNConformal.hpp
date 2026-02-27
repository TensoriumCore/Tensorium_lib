
#pragma once
#include <algorithm>
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
 * @brief Guard helper for conformal-factor divisions.
 * @details Uses `max(chi, chi_div_floor)` when a positive floor is provided,
 * otherwise falls back to machine epsilon to avoid singular divisions.
 */
template <typename T> inline T guarded_conformal_chi(T chi, T chi_div_floor) noexcept {
    if (chi_div_floor > T(0))
        return std::max(chi, chi_div_floor);
    return std::max(chi, T(1e-16));
}

/**
 * @brief Convert \f$\partial_i\chi\f$ to \f$\partial_i\phi\f$ using \f$\phi=-(1/4)\ln\chi\f$.
 * @details The relation is \f$\partial_i\phi = -(1/4)\chi^{-1}\partial_i\chi\f$.
 * @param dchi Cartesian gradient of \f$\chi\f$.
 * @param chi Scalar conformal factor.
 * @param[out] dphi Populated with \f$\partial_i\phi\f$.
 */
template <typename T>
inline void grad_phi_from_chi(const T dchi[3], T chi, T dphi[3],
                              T chi_div_floor = T(0)) noexcept {
    const T chi_guarded = guarded_conformal_chi(chi, chi_div_floor);
    const T inv_chi = T(1) / chi_guarded;
    const T c = T(-0.25) * inv_chi;
    dphi[0] = c * dchi[0];
    dphi[1] = c * dchi[1];
    dphi[2] = c * dchi[2];
}

/**
 * @brief Compute \f$\partial_i(6\phi) = -\tfrac{3}{2}\chi^{-1}\partial_i\chi\f$ for the momentum constraint.
 */
template <typename T>
inline void grad_6phi_from_chi(const T dchi[3], T chi, T d6phi[3],
                               T chi_div_floor = T(0)) noexcept {
    const T chi_guarded = guarded_conformal_chi(chi, chi_div_floor);
    const T inv_chi = T(1) / chi_guarded;
    const T c = T(-1.5) * inv_chi;
    d6phi[0] = c * dchi[0];
    d6phi[1] = c * dchi[1];
    d6phi[2] = c * dchi[2];
}

} // namespace tensorium_RG::bssn
