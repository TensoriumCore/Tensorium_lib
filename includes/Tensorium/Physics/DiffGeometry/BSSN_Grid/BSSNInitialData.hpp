#include "BSSNCHristoffelTilde.hpp"
#include "BSSNGridDerivatives.hpp"
#include "BSSNGridSoA.hpp"
#include <algorithm>
#include <stdio.h>

namespace tensorium_RG::init {

template <typename T>
inline void invert_gamma_tilde(BSSNGridSoA<T> &G, size_t i, size_t j, size_t k) {
    const size_t id = G.gamma_tilde[XX].idx(i, j, k);

    const T a = G.gamma_tilde[XX].ptr()[id];
    const T b = G.gamma_tilde[XY].ptr()[id];
    const T c = G.gamma_tilde[XZ].ptr()[id];
    const T d = G.gamma_tilde[YY].ptr()[id];
    const T e = G.gamma_tilde[YZ].ptr()[id];
    const T f = G.gamma_tilde[ZZ].ptr()[id];

    const T det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);

    const T inv_det = T(1) / det;

    G.gamma_tilde_inv[XX].ptr()[id] = (d * f - e * e) * inv_det;
    G.gamma_tilde_inv[XY].ptr()[id] = (c * e - b * f) * inv_det;
    G.gamma_tilde_inv[XZ].ptr()[id] = (b * e - c * d) * inv_det;
    G.gamma_tilde_inv[YY].ptr()[id] = (a * f - c * c) * inv_det;
    G.gamma_tilde_inv[YZ].ptr()[id] = (b * c - a * e) * inv_det;
    G.gamma_tilde_inv[ZZ].ptr()[id] = (a * d - b * b) * inv_det;
}

template <typename T> inline void minkowski(BSSNGridSoA<T> &G) {

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);

                G.alpha.ptr()[id] = T(1);
                G.chi.ptr()[id] = T(1);
                G.K.ptr()[id] = T(0);

                for (int c = 0; c < 3; ++c) {
                    G.beta[c].ptr()[id] = T(0);
                    G.tildeGamma[c].ptr()[id] = T(0);
                }

                const T one = T(1);
                const T zero = T(0);

                G.gamma_tilde[XX].ptr()[id] = one;
                G.gamma_tilde[XY].ptr()[id] = zero;
                G.gamma_tilde[XZ].ptr()[id] = zero;
                G.gamma_tilde[YY].ptr()[id] = one;
                G.gamma_tilde[YZ].ptr()[id] = zero;
                G.gamma_tilde[ZZ].ptr()[id] = one;

                G.gamma_tilde_inv[XX].ptr()[id] = one;
                G.gamma_tilde_inv[XY].ptr()[id] = zero;
                G.gamma_tilde_inv[XZ].ptr()[id] = zero;
                G.gamma_tilde_inv[YY].ptr()[id] = one;
                G.gamma_tilde_inv[YZ].ptr()[id] = zero;
                G.gamma_tilde_inv[ZZ].ptr()[id] = one;

                G.A_tilde[XX].ptr()[id] = zero;
                G.A_tilde[XY].ptr()[id] = zero;
                G.A_tilde[XZ].ptr()[id] = zero;
                G.A_tilde[YY].ptr()[id] = zero;
                G.A_tilde[YZ].ptr()[id] = zero;
                G.A_tilde[ZZ].ptr()[id] = zero;
            }
}

template <typename T>
inline void schwarzschild_isotropic(BSSNGridSoA<T> &G, T M, T xc = T(0), T yc = T(0), T zc = T(0),
                                    T r_floor = T(1e-6)) {

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    const T one = T(1);
    const T zero = T(0);
    const T half = T(0.5);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);

                T x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;

                T r2 = x * x + y * y + z * z;
                T r = std::sqrt(r2);
                if (r < r_floor)
                    r = r_floor;

                const T u = (M * half) / r; // M/(2r)
                const T psi = one + u;

                const T chi = one / (psi * psi * psi * psi); // psi^-4
                const T alpha = one / (psi * psi);           // psi^-2 = sqrt(chi)

                G.alpha.ptr()[id] = alpha;
                G.chi.ptr()[id] = chi;
                G.K.ptr()[id] = zero;

                for (int c = 0; c < 3; ++c) {
                    G.beta[c].ptr()[id] = zero;
                    G.tildeGamma[c].ptr()[id] = zero;
                }

                G.gamma_tilde[XX].ptr()[id] = one;
                G.gamma_tilde[XY].ptr()[id] = zero;
                G.gamma_tilde[XZ].ptr()[id] = zero;
                G.gamma_tilde[YY].ptr()[id] = one;
                G.gamma_tilde[YZ].ptr()[id] = zero;
                G.gamma_tilde[ZZ].ptr()[id] = one;

                invert_gamma_tilde(G, i, j, k);

                G.A_tilde[XX].ptr()[id] = zero;
                G.A_tilde[XY].ptr()[id] = zero;
                G.A_tilde[XZ].ptr()[id] = zero;
                G.A_tilde[YY].ptr()[id] = zero;
                G.A_tilde[YZ].ptr()[id] = zero;
                G.A_tilde[ZZ].ptr()[id] = zero;
            }

    apply_halos_grid<BoundaryClamp>(G);
    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
}

} // namespace tensorium_RG::init
