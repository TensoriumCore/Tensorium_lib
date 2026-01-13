#pragma once
#include "BSSNGamma.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"

/**
 * @file BSSNCHristoffelTilde.hpp
 * @brief Builders for the Christoffel symbols \f$\tilde{\Gamma}^i_{\ jk}\f$ and their contraction.
 * @details
 * Both helpers restrict loops to interior points so that the 4th-order derivative operators have
 * valid data.  `compute_tildeGamma_full` implements the textbook formula
 * \f$\tilde{\Gamma}^i_{\ jk} = \tfrac{1}{2}\tilde{\gamma}^{i\ell}(\partial_j\tilde{\gamma}_{\ell k} +
 * \partial_k\tilde{\gamma}_{\ell j} - \partial_\ell \tilde{\gamma}_{jk})\f$ and stores the result in the
 * 27-component cache (flattened as `[i][j][k]`).  `compute_tildeGamma_contracted` recomputes the
 * evolved \f$\tilde{\Gamma}^i\f$ from the metric divergence to enforce the algebraic definition after
 * projection steps.
 */

namespace tensorium_RG::bssn {

/**
 * @brief Refresh the evolved \f$\tilde{\Gamma}^i\f$ from the metric inverse divergence.
 * @details Mapping to code:
 * 1. Sample `metric_inverse_divergence` (see `BSSNGamma.hpp`).
 * 2. Store \f$\tilde{\Gamma}^i = -\partial_j\tilde{\gamma}^{ij}\f$ in the SoA vector fields.
 * @see metric_inverse_divergence
 */
template <typename T> inline void compute_tildeGamma_contracted(BSSNGridSoA<T> &G) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 2, i1 = I1 - 2;
    const size_t j0 = J0 + 2, j1 = J1 - 2;
    const size_t k0 = K0 + 2, k1 = K1 - 2;

#pragma omp parallel for collapse(3)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.gamma_tilde[XX].idx(i, j, k);
                T            div[3];
                metric_inverse_divergence(G, i, j, k, div);
                G.tildeGamma[0].ptr()[id] = -div[0];
                G.tildeGamma[1].ptr()[id] = -div[1];
                G.tildeGamma[2].ptr()[id] = -div[2];
            }
}

/**
 * @brief Compute the full \f$\tilde{\Gamma}^i_{\ jk}\f$ tensor at every interior grid point.
 * @param G Source grid holding \f$\tilde{\gamma}_{ij}\f$ and \f$\tilde{\gamma}^{ij}\f$.
 * @param[out] Gamma27 Array of 27 `Field3D` components stored as `i*9 + j*3 + k`.
 * @details
 * The derivative lambda `d_g` enforces symmetry \f$\tilde{\gamma}_{ij}=\tilde{\gamma}_{ji}\f$ by choosing the
 * packed component once before applying `Dx/Dy/Dz`.  Each Christoffel element is assembled by
 * contracting with \f$\tilde{\gamma}^{i\ell}\f` using the precomputed inverse metric cache.  The
 * resulting tensor feeds `BSSNRicci.hpp` and the constraint monitor's covariant derivatives.  Halo
 * padding of at least two cells is assumed.
 */
template <typename T> inline void compute_tildeGamma_full(BSSNGridSoA<T> &G, Field3D<T> *Gamma27) {
    using tensorium_RG::fd::Dx;
    using tensorium_RG::fd::Dy;
    using tensorium_RG::fd::Dz;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = I0 + 2, i1 = I1 - 2;
    const size_t j0 = J0 + 2, j1 = J1 - 2;
    const size_t k0 = K0 + 2, k1 = K1 - 2;

#pragma omp parallel for collapse(3)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {

                const size_t id = G.gamma_tilde[XX].idx(i, j, k);

                auto d_g = [&](int m, int a, int b) -> T {
                    if (a > b) {
                        int t = a;
                        a = b;
                        b = t;
                    }
                    const Field3D<T> &F = (a == 0 && b == 0)   ? G.gamma_tilde[XX]
                                          : (a == 0 && b == 1) ? G.gamma_tilde[XY]
                                          : (a == 0 && b == 2) ? G.gamma_tilde[XZ]
                                          : (a == 1 && b == 1) ? G.gamma_tilde[YY]
                                          : (a == 1 && b == 2) ? G.gamma_tilde[YZ]
                                                               : G.gamma_tilde[ZZ];

                    if (m == 0)
                        return Dx(F, i, j, k, G.dx);
                    if (m == 1)
                        return Dy(F, i, j, k, G.dy);
                    return Dz(F, i, j, k, G.dz);
                };

                for (int kk = 0; kk < 3; ++kk)
                    for (int ii = 0; ii < 3; ++ii)
                        for (int jj = 0; jj < 3; ++jj) {

                            T sum = T(0);
                            for (int ll = 0; ll < 3; ++ll) {
                                const T ginv_kl = sym6_inv_get(G.gamma_tilde_inv, id, kk, ll);
                                const T term = d_g(ii, jj, ll) + d_g(jj, ii, ll) - d_g(ll, ii, jj);
                                sum += ginv_kl * term;
                            }
                            const T Gkij = T(0.5) * sum;
                            Gamma27[kk * 9 + ii * 3 + jj].ptr()[id] = Gkij;
                        }
            }
}

} // namespace tensorium_RG::bssn
