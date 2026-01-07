#include "BSSNGridSoA.hpp"
#include <stdio.h>

namespace tensorium_RG::init {

template <typename T> inline void minkowski(BSSNGridSoA<T> &G) {

#ifdef TENSORIUM_INIT_DEBUG
    printf("=== init::minkowski ===\n");
    printf("Grid dims: nx=%zu ny=%zu nz=%zu ng=%zu\n", G.dims.nx, G.dims.ny, G.dims.nz, G.dims.ng);
    printf("dx=%g dy=%g dz=%g\n", double(G.dx), double(G.dy), double(G.dz));
#endif

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#ifdef TENSORIUM_INIT_DEBUG
    printf("Interior bounds:\n");
    printf("  i: [%zu, %zu)\n", I0, I1);
    printf("  j: [%zu, %zu)\n", J0, J1);
    printf("  k: [%zu, %zu)\n", K0, K1);
#endif

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

#ifdef TENSORIUM_INIT_DEBUG
    const size_t i = I0, j = J0, k = K0;
    const size_t id = G.alpha.idx(i, j, k);

    printf("Sample cell (i=%zu,j=%zu,k=%zu):\n", i, j, k);
    printf("  alpha = %g\n", double(G.alpha.ptr()[id]));
    printf("  chi   = %g\n", double(G.chi.ptr()[id]));
    printf("  K     = %g\n", double(G.K.ptr()[id]));

    printf("  gamma_tilde diag = (%g,%g,%g)\n", double(G.gamma_tilde[XX].ptr()[id]),
           double(G.gamma_tilde[YY].ptr()[id]), double(G.gamma_tilde[ZZ].ptr()[id]));

    printf("  A_tilde diag = (%g,%g,%g)\n", double(G.A_tilde[XX].ptr()[id]),
           double(G.A_tilde[YY].ptr()[id]), double(G.A_tilde[ZZ].ptr()[id]));

    printf("=== init::minkowski DONE ===\n\n");
#endif
}

template <typename T>
inline void schwarzschild_isotropic(BSSNGridSoA<T> &G, T M, T xc = T(0), T yc = T(0), T zc = T(0),
                                    T r_floor = T(1e-6)) {

    printf("=== init::schwarzschild_isotropic_1plog ===\n");
    printf("Grid dims: nx=%zu ny=%zu nz=%zu ng=%zu\n", G.dims.nx, G.dims.ny, G.dims.nz, G.dims.ng);
    printf("dx=%g dy=%g dz=%g\n", double(G.dx), double(G.dy), double(G.dz));
    printf("M=%g center=(%g,%g,%g) r_floor=%g\n", double(M), double(xc), double(yc), double(zc),
           double(r_floor));

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const T one = T(1);
    const T zero = T(0);
    const T half = T(0.5);

#pragma omp parallel for collapse(2)
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

    const size_t si = I0, sj = J0, sk = K0;
    const size_t sid = G.alpha.idx(si, sj, sk);
    printf("Sample cell (i=%zu,j=%zu,k=%zu): alpha=%g chi=%g K=%g\n", si, sj, sk,
           double(G.alpha.ptr()[sid]), double(G.chi.ptr()[sid]), double(G.K.ptr()[sid]));
    printf("=== init::schwarzschild_isotropic_1plog DONE ===\n\n");
    double max_rel = 0.0;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    for (size_t i = I0; i < I1; i += 17)
        for (size_t j = J0; j < J1; j += 19)
            for (size_t k = K0; k < K1; k += 23) {
                const size_t id = G.alpha.idx(i, j, k);
                const double a = G.alpha.ptr()[id];
                const double c = G.chi.ptr()[id];
                const double ref = a * a;
                const double rel = std::abs(c - ref) / (std::abs(ref) + 1e-300);
                if (rel > max_rel)
                    max_rel = rel;
            }

    printf("[CHECK] chi ~= alpha^2 : max_rel = %.3e\n", max_rel);
    size_t i = I0 + G.dims.nx - 1;
    size_t j = J0 + G.dims.ny / 2;
    size_t k = K0 + G.dims.nz / 2;
    size_t id = G.alpha.idx(i, j, k);

    printf("[FAR] alpha=%g chi=%g\n", double(G.alpha.ptr()[id]), double(G.chi.ptr()[id]));
}

} // namespace tensorium_RG::init
