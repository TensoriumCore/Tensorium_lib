#pragma once
#include "BSSNGridDerivatives.hpp"
#include "BSSNGridSoA.hpp"

namespace tensorium_RG::bssn {

template <typename T> inline void compute_tildeGamma_contracted(BSSNGridSoA<T> &G) {
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

                T GammaContr[3] = {T(0), T(0), T(0)};

                // Γ̃^i = g̃^{jk} Γ̃^i_{jk}
                for (int ii = 0; ii < 3; ++ii) {
                    T acc = T(0);

                    for (int jj = 0; jj < 3; ++jj)
                        for (int kk = 0; kk < 3; ++kk) {
                            T sum_l = T(0);
                            for (int ll = 0; ll < 3; ++ll) {
                                const T ginv_il = sym6_inv_get(G.gamma_tilde_inv, id, ii, ll);

                                const T term = d_g(jj, kk, ll) + d_g(kk, jj, ll) - d_g(ll, jj, kk);

                                sum_l += ginv_il * term;
                            }
                            const T Gamma_i_jk = T(0.5) * sum_l;

                            const T ginv_jk = sym6_inv_get(G.gamma_tilde_inv, id, jj, kk);
                            acc += ginv_jk * Gamma_i_jk;
                        }

                    GammaContr[ii] = acc;
                }

                G.tildeGamma[0].ptr()[id] = GammaContr[0];
                G.tildeGamma[1].ptr()[id] = GammaContr[1];
                G.tildeGamma[2].ptr()[id] = GammaContr[2];
            }
}

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
