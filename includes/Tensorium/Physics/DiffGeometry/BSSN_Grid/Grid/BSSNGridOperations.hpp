#pragma once

#include "../Fields/BSSNGridSoA.hpp"

/**
 * @file BSSNGridOperations.hpp
 * @brief Grid-wide helpers such as halo application shared by evolution and initialization code.
 * @details
 * The Runge–Kutta driver templates over a `Boundary` functor that satisfies the `apply(Field3D,
 * GridDims)` interface.  This indirection centralizes halo synchronization (or boundary clamping)
 * so the physics kernels only see fully-populated guard zones before computing derivatives.  Every
 * field contained in `BSSNGridSoA` is forwarded to the boundary functor in a consistent order to
 * keep caches coherent and to support boundary conditions that need coupled updates (e.g., for
 * shift/torsion).
 */

namespace tensorium_RG::bssn {

/**
 * @brief Apply the boundary functor to every scalar, vector, and tensor field in the grid.
 * @tparam Boundary Functor type that exposes `static void apply(Field3D<T>&, const GridDims&)`.
 * @details
 * Mapping to code:
 * - The first loop handles scalar gauges \f$\alpha,\chi,K\f$.
 * - The second loop applies the functor to each component of \f$\beta^i, B^i, \tilde{\Gamma}^i\f$.
 * - Symmetric tensors (metric, inverse, \f\tilde{A}_{ij}\f) reuse the packed 6-component arrays.
 * - Cached geometry (`Gamma_tilde`) follows so the Ricci builder can be invoked immediately after
 *   halo application.
 *
 * @note Guard updates must precede any derivative or KO6 evaluation because the latter access up to
 * three off-domain samples.  Refer to @ref BSSN_Grid for halo width requirements.
 */
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
