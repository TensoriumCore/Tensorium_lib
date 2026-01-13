#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionChi.hpp
 * @brief RHS for the conformal factor equation \f$(\partial_t-\mathcal{L}_\beta)\chi = \frac{2}{3}\chi(\alpha K-\partial_i\beta^i)\f$.
 * @details The implementation evaluates
 * \f[
 * \partial_t\chi = \beta^i\partial_i\chi + \tfrac{2}{3}\chi(\alpha K-\partial_i\beta^i) + \mathcal{D}_6[\chi]
 * \f]
 * where advection uses third-order upwind stencils tied to the sign of each \f$\beta^i\f$ component and
 * \f$\mathcal{D}_6\f$ is the KO6 dissipation.
 */

namespace tensorium_RG::bssn {

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#    define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}
#endif

/**
 * @brief Fill the RHS buffer for \f$\chi\f$ inside the padded interior region.
 * @param padding Number of guard cells skipped on each side before looping.
 *
 * **Mapping to code.**
 * - `d_chi` corresponds to the advective term \f$\beta^i\partial_i\chi\f$.
 * - `div_beta` uses centered derivatives to compute \f$\partial_i\beta^i\f$.
 * - The scalar multiplier `T(2/3)*chi*(alpha*K - div_beta)` encodes the source term.
 * - KO6 call adds dissipation with \f$\sigma=0.1\f$.
 */
template <typename T>
inline void compute_rhs_chi(const BSSNGridSoA<T> &G, Field3D<T> &rhs_chi, size_t padding = 4) {
    const T ko_sigma = T(0.1);
    size_t  I0, I1, J0, J1, K0, K1;
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
                const T d_chi = beta_x * tensorium_RG::fd::Dx_upwind(G.chi, i, j, k, G.dx, beta_x) +
                                beta_y * tensorium_RG::fd::Dy_upwind(G.chi, i, j, k, G.dy, beta_y) +
                                beta_z * tensorium_RG::fd::Dz_upwind(G.chi, i, j, k, G.dz, beta_z);
                const T div_beta = tensorium_RG::fd::Dx(G.beta[0], i, j, k, G.dx) +
                                   tensorium_RG::fd::Dy(G.beta[1], i, j, k, G.dy) +
                                   tensorium_RG::fd::Dz(G.beta[2], i, j, k, G.dz);
                rhs_chi.ptr()[id] = d_chi + (T(2.0 / 3.0)) * chi * (alpha * K - div_beta) +
                                    T(tensorium_RG::fd::KO6(G.chi, i, j, k, ko_sigma));
            }
        }
    }
}

} // namespace tensorium_RG::bssn
