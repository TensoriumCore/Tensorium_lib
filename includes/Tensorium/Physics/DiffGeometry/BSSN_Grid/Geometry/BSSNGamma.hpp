#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"

namespace tensorium_RG::bssn {

namespace detail {

template <typename T>
inline T diff_gamma_tilde_inv_component(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                        int row, int col, int dir) {
    const int idx = tensorium_RG::sym6_index(row, col);
    const auto &F = G.gamma_tilde_inv[idx];
    if (dir == 0)
        return (T)tensorium_RG::fd::Dx(F, i, j, k, G.dx);
    if (dir == 1)
        return (T)tensorium_RG::fd::Dy(F, i, j, k, G.dy);
    return (T)tensorium_RG::fd::Dz(F, i, j, k, G.dz);
}

} // namespace detail

// Computes ∂_j γ̃^{ij} without mutating G.tildeGamma.  The Γ constraint compares
// this diagnostic derivative against the stateful vector field stored in the
// grid, so that violations remain observable even if Γ̃^i is evolved
// independently.
template <typename T>
inline void metric_inverse_divergence(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                      T out[3]) {
    for (int a = 0; a < 3; ++a) {
        T sum = T(0);
        sum += detail::diff_gamma_tilde_inv_component(G, i, j, k, a, 0, 0);
        sum += detail::diff_gamma_tilde_inv_component(G, i, j, k, a, 1, 1);
        sum += detail::diff_gamma_tilde_inv_component(G, i, j, k, a, 2, 2);
        out[a] = sum;
    }
}

// Convenience helper for initial-data construction.  Do not call this function
// just prior to evaluating the Γ̃ constraint, otherwise the check becomes
// tautological.
template <typename T>
inline void compute_contracted_gamma_from_metric(BSSNGridSoA<T> &G, size_t i, size_t j,
                                                 size_t k, T out[3]) {
    metric_inverse_divergence(G, i, j, k, out);
    out[0] = -out[0];
    out[1] = -out[1];
    out[2] = -out[2];
}

} // namespace tensorium_RG::bssn
