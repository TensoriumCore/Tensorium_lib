
#pragma once
#include <cstddef>

namespace tensorium_RG::bssn {

template <typename T> inline void grad_phi_from_chi(const T dchi[3], T chi, T dphi[3]) noexcept {
    const T inv_chi = T(1) / chi;
    const T c = T(-0.25) * inv_chi;
    dphi[0] = c * dchi[0];
    dphi[1] = c * dchi[1];
    dphi[2] = c * dchi[2];
}

template <typename T> inline void grad_6phi_from_chi(const T dchi[3], T chi, T d6phi[3]) noexcept {
    const T inv_chi = T(1) / chi;
    const T c = T(-1.5) * inv_chi;
    d6phi[0] = c * dchi[0];
    d6phi[1] = c * dchi[1];
    d6phi[2] = c * dchi[2];
}

} // namespace tensorium_RG::bssn
