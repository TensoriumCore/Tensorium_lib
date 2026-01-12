#pragma once

#include "../Fields/BSSNGridSoA.hpp"

namespace tensorium_RG::bssn {

template <typename Boundary, typename T> inline void apply_halos_grid(BSSNGridSoA<T> &G) {
    const auto &D = G.dims;

    Boundary::apply(G.alpha, D);
    Boundary::apply(G.chi, D);
    Boundary::apply(G.K, D);

    for (int i = 0; i < 3; ++i) {
        Boundary::apply(G.beta[i], D);
        Boundary::apply(G.B[i], D);
        Boundary::apply(G.tildeGamma[i], D);
    }

    for (int s = 0; s < 6; ++s) {
        Boundary::apply(G.gamma_tilde[s], D);
        Boundary::apply(G.gamma_tilde_inv[s], D);
        Boundary::apply(G.A_tilde[s], D);
    }
    for (int q = 0; q < 27; ++q)
        Boundary::apply(G.Gamma_tilde[q], D);
}

} // namespace tensorium_RG::bssn
