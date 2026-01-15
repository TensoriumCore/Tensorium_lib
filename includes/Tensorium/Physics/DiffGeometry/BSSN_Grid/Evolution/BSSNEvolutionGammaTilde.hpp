#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNEvolutionGammaTilde.hpp
 * @brief RHS for the conformal metric \f$\tilde{\gamma}_{ij}\f$.
 * @details Evaluates
 * \f[
 * \partial_t\tilde{\gamma}_{ij} = -2\alpha\tilde{A}_{ij} + \mathcal{L}_\beta \tilde{\gamma}_{ij} -
 * \tfrac{2}{3}\tilde{\gamma}_{ij}\partial_k\beta^k + \mathcal{D}_6[\tilde{\gamma}_{ij}].
 * \f]
 * Lie derivatives expand into \f$\tilde{\gamma}_{ik}\partial_j\beta^k +
 * \tilde{\gamma}_{jk}\partial_i\beta^k\f$ plus the advection term
 * \f$\beta^k\partial_k\tilde{\gamma}_{ij}\f$.
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
 * @brief Fill \f$\partial_t\tilde{\gamma}_{ij}\f$ for all six symmetric components.
 */
template <typename T>
inline void compute_rhs_gamma_tilde(const BSSNGridSoA<T> &G, Field3D<T> rhs[6],
                                    size_t padding = 4) {
    using namespace tensorium_RG::fd;
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t total =
        G.gamma_tilde[0].st.nx_tot * G.gamma_tilde[0].st.ny_tot * G.gamma_tilde[0].st.nz_tot;
    for (int s = 0; s < 6; ++s)
        std::fill(rhs[s].ptr(), rhs[s].ptr() + total, T(0));

    const size_t i0 = clamped_lower(I0, padding, I1);
    const size_t j0 = clamped_lower(J0, padding, J1);
    const size_t k0 = clamped_lower(K0, padding, K1);
    const size_t i1 = clamped_upper(I1, padding, I0);
    const size_t j1 = clamped_upper(J1, padding, J0);
    const size_t k1 = clamped_upper(K1, padding, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    const T ko_sigma = T(0.6);

#pragma omp parallel for collapse(3)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.gamma_tilde[0].idx(i, j, k);
                const T      alpha = G.alpha.ptr()[id];

                T d_beta[3][3];
                d_beta[0][0] = Dx(G.beta[0], i, j, k, G.dx);
                d_beta[0][1] = Dy(G.beta[0], i, j, k, G.dy);
                d_beta[0][2] = Dz(G.beta[0], i, j, k, G.dz);
                d_beta[1][0] = Dx(G.beta[1], i, j, k, G.dx);
                d_beta[1][1] = Dy(G.beta[1], i, j, k, G.dy);
                d_beta[1][2] = Dz(G.beta[1], i, j, k, G.dz);
                d_beta[2][0] = Dx(G.beta[2], i, j, k, G.dx);
                d_beta[2][1] = Dy(G.beta[2], i, j, k, G.dy);
                d_beta[2][2] = Dz(G.beta[2], i, j, k, G.dz);

                const T div_beta = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                const T beta_vec[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id],
                                       G.beta[2].ptr()[id]};

                T gam[3][3];
                for (int u = 0; u < 3; ++u)
                    for (int v = u; v < 3; ++v) {
                        T val = G.gamma_tilde[tensorium_RG::sym6_index(u, v)].ptr()[id];
                        gam[u][v] = val;
                        gam[v][u] = val;
                    }

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const int s = tensorium_RG::sym6_index(a, b);

                        T adv = beta_vec[0] * tensorium_RG::fd::Dx_upwind(G.gamma_tilde[s], i, j, k,
                                                                          G.dx, beta_vec[0]) +
                                beta_vec[1] * tensorium_RG::fd::Dy_upwind(G.gamma_tilde[s], i, j, k,
                                                                          G.dy, beta_vec[1]) +
                                beta_vec[2] * tensorium_RG::fd::Dz_upwind(G.gamma_tilde[s], i, j, k,
                                                                          G.dz, beta_vec[2]);

                        T lie = T(0);
                        for (int m = 0; m < 3; ++m) {
                            lie += gam[a][m] * d_beta[m][b];
                            lie += gam[b][m] * d_beta[m][a];
                        }

                        const T source = -T(2) * alpha * G.A_tilde[s].ptr()[id];
                        const T trace_rem = -T(2.0 / 3.0) * gam[a][b] * div_beta;
                        const T diss =
                            T(tensorium_RG::fd::KO6(G.gamma_tilde[s], i, j, k, ko_sigma));

                        rhs[s].ptr()[id] = adv + lie + source + trace_rem + diss;
                    }
                }
            }
        }
    }
}
} // namespace tensorium_RG::bssn
