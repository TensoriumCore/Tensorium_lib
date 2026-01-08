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

    const size_t si = I0, sj = J0, sk = K0;
    const size_t sid = G.alpha.idx(si, sj, sk);
    printf("Sample cell (i=%zu,j=%zu,k=%zu): alpha=%g chi=%g K=%g\n", si, sj, sk,
           double(G.alpha.ptr()[sid]), double(G.chi.ptr()[sid]), double(G.K.ptr()[sid]));
    printf("=== init::schwarzschild_isotropic_1plog DONE ===\n\n");
    double max_rel = 0.0;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t ii = I0 + 2 + G.dims.nx / 2;
    const size_t jj = J0 + 2 + G.dims.ny / 2;
    const size_t kk = K0 + 2 + G.dims.nz / 2;
    const size_t idc = G.alpha.idx(ii, jj, kk);
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
    {
        const size_t id = G.chi.idx(ii, jj, kk);

        const double chi = G.chi.ptr()[id];
        const double inv_chi = 1.0 / chi;

        const double g_xx = inv_chi * G.gamma_tilde[XX].ptr()[id];
        const double g_xy = inv_chi * G.gamma_tilde[XY].ptr()[id];
        const double g_xz = inv_chi * G.gamma_tilde[XZ].ptr()[id];
        const double g_yy = inv_chi * G.gamma_tilde[YY].ptr()[id];
        const double g_yz = inv_chi * G.gamma_tilde[YZ].ptr()[id];
        const double g_zz = inv_chi * G.gamma_tilde[ZZ].ptr()[id];

        printf("[CHECK] physical gamma_ij at (i=%zu,j=%zu,k=%zu)\n"
               "  [[%.6e %.6e %.6e]\n"
               "   [%.6e %.6e %.6e]\n"
               "   [%.6e %.6e %.6e]]\n",
               ii, jj, kk, g_xx, g_xy, g_xz, g_xy, g_yy, g_yz, g_xz, g_yz, g_zz);
    }
    {
        const size_t idf = G.chi.idx(i, j, k);
        const double chi_f = G.chi.ptr()[idf];
        const double inv_chi_f = 1.0 / chi_f;

        const double gxx = inv_chi_f * G.gamma_tilde[XX].ptr()[idf];
        const double gyy = inv_chi_f * G.gamma_tilde[YY].ptr()[idf];
        const double gzz = inv_chi_f * G.gamma_tilde[ZZ].ptr()[idf];

        printf("[FAR] physical gamma diag = (%g, %g, %g)\n", gxx, gyy, gzz);
    }
    {
        const size_t id = G.chi.idx(ii, jj, kk);

        const double chi = G.chi.ptr()[id];
        const double inv_chi = 1.0 / chi;

        const double g_xx = inv_chi * G.gamma_tilde[XX].ptr()[id];
        const double g_xy = inv_chi * G.gamma_tilde[XY].ptr()[id];
        const double g_xz = inv_chi * G.gamma_tilde[XZ].ptr()[id];
        const double g_yy = inv_chi * G.gamma_tilde[YY].ptr()[id];
        const double g_yz = inv_chi * G.gamma_tilde[YZ].ptr()[id];
        const double g_zz = inv_chi * G.gamma_tilde[ZZ].ptr()[id];

        const double gi_xx = chi * G.gamma_tilde_inv[XX].ptr()[id];
        const double gi_xy = chi * G.gamma_tilde_inv[XY].ptr()[id];
        const double gi_xz = chi * G.gamma_tilde_inv[XZ].ptr()[id];
        const double gi_yy = chi * G.gamma_tilde_inv[YY].ptr()[id];
        const double gi_yz = chi * G.gamma_tilde_inv[YZ].ptr()[id];
        const double gi_zz = chi * G.gamma_tilde_inv[ZZ].ptr()[id];

        const double Cxx = g_xx * gi_xx + g_xy * gi_xy + g_xz * gi_xz;
        const double Cxy = g_xx * gi_xy + g_xy * gi_yy + g_xz * gi_yz;
        const double Cxz = g_xx * gi_xz + g_xy * gi_yz + g_xz * gi_zz;

        const double Cyx = g_xy * gi_xx + g_yy * gi_xy + g_yz * gi_xz;
        const double Cyy = g_xy * gi_xy + g_yy * gi_yy + g_yz * gi_yz;
        const double Cyz = g_xy * gi_xz + g_yy * gi_yz + g_yz * gi_zz;

        const double Czx = g_xz * gi_xx + g_yz * gi_xy + g_zz * gi_xz;
        const double Czy = g_xz * gi_xy + g_yz * gi_yy + g_zz * gi_yz;
        const double Czz = g_xz * gi_xz + g_yz * gi_yz + g_zz * gi_zz;

        const double err =
            std::max({std::abs(Cxx - 1.0), std::abs(Cyy - 1.0), std::abs(Czz - 1.0), std::abs(Cxy),
                      std::abs(Cxz), std::abs(Cyx), std::abs(Cyz), std::abs(Czx), std::abs(Czy)});

        printf("[CHECK] ||gamma_phys * gamma_phys_inv - I||_inf = %.3e\n", err);
    }
    {
        using tensorium_RG::fd::Dx;
        using tensorium_RG::fd::Dy;
        using tensorium_RG::fd::Dz;

        double max_abs_dchi = 0.0;
        double max_abs_dgtilde = 0.0;

        size_t imax = I0, jmax = J0, kmax = K0;
        int    amax = 0;

        const size_t iD0 = I0 + 2, iD1 = I1 - 2;
        const size_t jD0 = J0 + 2, jD1 = J1 - 2;
        const size_t kD0 = K0 + 2, kD1 = K1 - 2;
        for (size_t i = iD0; i < iD1; ++i)
            for (size_t j = jD0; j < jD1; ++j)
                for (size_t k = kD0; k < kD1; ++k) {

                    const double dchi_x = Dx(G.chi, i, j, k, double(G.dx));
                    const double dchi_y = Dy(G.chi, i, j, k, double(G.dy));
                    const double dchi_z = Dz(G.chi, i, j, k, double(G.dz));

                    max_abs_dchi = std::max(max_abs_dchi, std::abs(dchi_x));
                    max_abs_dchi = std::max(max_abs_dchi, std::abs(dchi_y));
                    max_abs_dchi = std::max(max_abs_dchi, std::abs(dchi_z));

                    const double dgxx_x = Dx(G.gamma_tilde[XX], i, j, k, double(G.dx));
                    const double dgxx_y = Dy(G.gamma_tilde[XX], i, j, k, double(G.dy));
                    const double dgxx_z = Dz(G.gamma_tilde[XX], i, j, k, double(G.dz));

                    const double dgxy_x = Dx(G.gamma_tilde[XY], i, j, k, double(G.dx));
                    const double dgxy_y = Dy(G.gamma_tilde[XY], i, j, k, double(G.dy));
                    const double dgxy_z = Dz(G.gamma_tilde[XY], i, j, k, double(G.dz));

                    const double dgxz_x = Dx(G.gamma_tilde[XZ], i, j, k, double(G.dx));
                    const double dgxz_y = Dy(G.gamma_tilde[XZ], i, j, k, double(G.dy));
                    const double dgxz_z = Dz(G.gamma_tilde[XZ], i, j, k, double(G.dz));

                    const double dgyy_x = Dx(G.gamma_tilde[YY], i, j, k, double(G.dx));
                    const double dgyy_y = Dy(G.gamma_tilde[YY], i, j, k, double(G.dy));
                    const double dgyy_z = Dz(G.gamma_tilde[YY], i, j, k, double(G.dz));

                    const double dgyz_x = Dx(G.gamma_tilde[YZ], i, j, k, double(G.dx));
                    const double dgyz_y = Dy(G.gamma_tilde[YZ], i, j, k, double(G.dy));
                    const double dgyz_z = Dz(G.gamma_tilde[YZ], i, j, k, double(G.dz));

                    const double dgzz_x = Dx(G.gamma_tilde[ZZ], i, j, k, double(G.dx));
                    const double dgzz_y = Dy(G.gamma_tilde[ZZ], i, j, k, double(G.dy));
                    const double dgzz_z = Dz(G.gamma_tilde[ZZ], i, j, k, double(G.dz));

                    const double local_max = std::max(
                        {std::abs(dgxx_x), std::abs(dgxx_y), std::abs(dgxx_z), std::abs(dgxy_x),
                         std::abs(dgxy_y), std::abs(dgxy_z), std::abs(dgxz_x), std::abs(dgxz_y),
                         std::abs(dgxz_z), std::abs(dgyy_x), std::abs(dgyy_y), std::abs(dgyy_z),
                         std::abs(dgyz_x), std::abs(dgyz_y), std::abs(dgyz_z), std::abs(dgzz_x),
                         std::abs(dgzz_y), std::abs(dgzz_z)});

                    if (local_max > max_abs_dgtilde) {
                        max_abs_dgtilde = local_max;
                        imax = i;
                        jmax = j;
                        kmax = k;
                    }
                }

        printf("[CHECK] max |∂chi| over grid = %.6e\n", max_abs_dchi);
        printf("[CHECK] max |∂gamma_tilde| over grid = %.6e (should be ~0 for isotropic init)\n",
               max_abs_dgtilde);

        {
            const double dchi_x = Dx(G.chi, ii, jj, kk, double(G.dx));
            const double dchi_y = Dy(G.chi, ii, jj, kk, double(G.dy));
            const double dchi_z = Dz(G.chi, ii, jj, kk, double(G.dz));

            printf("[CHECK] dchi at (i=%zu,j=%zu,k=%zu) = (% .6e, % .6e, % .6e)\n", ii, jj, kk,
                   dchi_x, dchi_y, dchi_z);

            const double dgxx_x = Dx(G.gamma_tilde[XX], ii, jj, kk, double(G.dx));
            const double dgxx_y = Dy(G.gamma_tilde[XX], ii, jj, kk, double(G.dy));
            const double dgxx_z = Dz(G.gamma_tilde[XX], ii, jj, kk, double(G.dz));

            printf("[CHECK] d(gamma_tilde_xx) at sample = (% .6e, % .6e, % .6e)\n", dgxx_x, dgxx_y,
                   dgxx_z);
        }
    }
    {
        const size_t id = G.tildeGamma[0].idx(ii, jj, kk);

        printf("[CHECK] tildeGamma^i at (%zu,%zu,%zu) = (% .6e, % .6e, % .6e)\n", ii, jj, kk,
               double(G.tildeGamma[0].ptr()[id]), double(G.tildeGamma[1].ptr()[id]),
               double(G.tildeGamma[2].ptr()[id]));
    }
    {
        const size_t id = G.gamma_tilde[0].idx(ii, jj, kk);

        printf("[CHECK] Gamma_tilde^k_{ij} at (%zu,%zu,%zu):\n", ii, jj, kk);
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double v = G.Gamma_tilde[k * 9 + i * 3 + j].ptr()[id];
                    printf("  Γ̃^%d_{%d%d} = % .6e\n", k, i, j, v);
                }
            }
        }
    }
}

} // namespace tensorium_RG::init
