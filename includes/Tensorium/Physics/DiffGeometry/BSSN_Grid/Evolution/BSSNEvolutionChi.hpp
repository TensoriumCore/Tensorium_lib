#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

namespace tensorium_RG::bssn {

inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}

template <typename T>
inline void compute_rhs_chi(const BSSNGridSoA<T> &G, Field3D<T> &rhs_chi, size_t padding = 4) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    std::fill(rhs_chi.ptr(), rhs_chi.ptr() + G.chi.st.nx_tot * G.chi.st.ny_tot * G.chi.st.nz_tot,
              T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.chi.idx(i, j, k);
                const T      chi = G.chi.ptr()[id];
                const T      alpha = G.alpha.ptr()[id];
                const T      K = G.K.ptr()[id];
                const T      beta_x = G.beta[0].ptr()[id];
                const T      beta_y = G.beta[1].ptr()[id];
                const T      beta_z = G.beta[2].ptr()[id];
                const T      d_chi = beta_x * tensorium_RG::fd::Dx(G.chi, i, j, k, G.dx) +
                                beta_y * tensorium_RG::fd::Dy(G.chi, i, j, k, G.dy) +
                                beta_z * tensorium_RG::fd::Dz(G.chi, i, j, k, G.dz);
                const T div_beta = tensorium_RG::fd::Dx(G.beta[0], i, j, k, G.dx) +
                                   tensorium_RG::fd::Dy(G.beta[1], i, j, k, G.dy) +
                                   tensorium_RG::fd::Dz(G.beta[2], i, j, k, G.dz);
                rhs_chi.ptr()[id] = d_chi + (T(2.0 / 3.0)) * chi * (alpha * K - div_beta);
            }
        }
    }
}

} // namespace tensorium_RG::bssn
