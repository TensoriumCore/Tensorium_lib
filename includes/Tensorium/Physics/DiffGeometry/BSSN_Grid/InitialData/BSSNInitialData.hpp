#pragma once

#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNInvariants.hpp"
#include "../Solvers/BSSNConstrainSolver.hpp"
#include "../Constraints/BSSNConstraintsGrid.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tensorium_RG::init {

template <typename T> void print_ricci_samples(const BSSNGridSoA<T> &G) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    size_t cx = I0 + G.dims.nx / 2;
    size_t cy = J0 + G.dims.ny / 2;
    size_t cz = K0 + G.dims.nz / 2;

    struct Sample {
        const char *name;
        size_t      i, j, k;
    };
    Sample samples[] = {{"Near (r ~ 4*dx)", cx + 4, cy, cz},
                        {"Mid  (r ~ N/4) ", cx + G.dims.nx / 4, cy, cz},
                        {"Far  (r ~ N/2) ", I1 - 4, cy, cz}};

    printf("\n=== Ricci Tensor Samples ===\n");
    for (const auto &s : samples) {
        size_t id = G.alpha.idx(s.i, s.j, s.k);

        T x, y, z;
        G.coords(s.i, s.j, s.k, x, y, z);
        T r = std::sqrt(x * x + y * y + z * z);

        T Rxx = G.Ricci[XX].ptr()[id];
        T Rxy = G.Ricci[XY].ptr()[id];
        T Rxz = G.Ricci[XZ].ptr()[id];
        T Ryy = G.Ricci[YY].ptr()[id];
        T Ryz = G.Ricci[YZ].ptr()[id];
        T Rzz = G.Ricci[ZZ].ptr()[id];

        printf("[%s] Index(%zu,%zu,%zu) r=%.4f\n", s.name, s.i, s.j, s.k, double(r));
        printf("      [[ % .4e  % .4e  % .4e ]\n", double(Rxx), double(Rxy), double(Rxz));
        printf("       [ % .4e  % .4e  % .4e ]\n", double(Rxy), double(Ryy), double(Ryz));
        printf("       [ % .4e  % .4e  % .4e ]]\n", double(Rxz), double(Ryz), double(Rzz));
    }
    printf("============================\n\n");
}

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

template <typename T>
inline void minkowski(BSSNGridSoA<T> &G, T M, T xc = T(0), T yc = T(0), T zc = T(0),
                      T r_floor = T(1e-6)) {

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
    bssn::apply_halos_grid<BoundaryClamp>(G);
    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);
    double maxAbsR = 0.0;
    size_t nSamples = 0;
    double rMin = 1e300, rMax = 0.0;

    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;

                const double r = std::sqrt(x * x + y * y + z * z);

                const double r_min = 2.0;

                const double r_max = 0.45 * std::min({(I1 - I0 - 1) * G.dx, (J1 - J0 - 1) * G.dy,
                                                      (K1 - K0 - 1) * G.dz});
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.alpha.idx(i, j, k);

                const double chi = G.chi.ptr()[id];

                const double ginv_xx = chi * G.gamma_tilde_inv[XX].ptr()[id];
                const double ginv_xy = chi * G.gamma_tilde_inv[XY].ptr()[id];
                const double ginv_xz = chi * G.gamma_tilde_inv[XZ].ptr()[id];
                const double ginv_yy = chi * G.gamma_tilde_inv[YY].ptr()[id];
                const double ginv_yz = chi * G.gamma_tilde_inv[YZ].ptr()[id];
                const double ginv_zz = chi * G.gamma_tilde_inv[ZZ].ptr()[id];

                const double Rxx = G.Ricci[XX].ptr()[id];
                const double Rxy = G.Ricci[XY].ptr()[id];
                const double Rxz = G.Ricci[XZ].ptr()[id];
                const double Ryy = G.Ricci[YY].ptr()[id];
                const double Ryz = G.Ricci[YZ].ptr()[id];
                const double Rzz = G.Ricci[ZZ].ptr()[id];

                const double R = ginv_xx * Rxx + 2.0 * ginv_xy * Rxy + 2.0 * ginv_xz * Rxz +
                                 ginv_yy * Ryy + 2.0 * ginv_yz * Ryz + ginv_zz * Rzz;
                maxAbsR = std::max(maxAbsR, std::abs(R));
                nSamples++;

                if (nSamples == 1) {
                    printf("[DBG] chi=%g Rxx=%g Ryy=%g Rzz=%g\n", chi, Rxx, Ryy, Rzz);
                }
                rMin = std::min(rMin, r);
                rMax = std::max(rMax, r);
            }

    printf("[CHECK] samples=%zu, rMin=%g, rMax=%g\n", nSamples, rMin, rMax);
    printf("[CHECK] max |R| for r > %.3f = %.3e\n", rMin, maxAbsR);
    print_ricci_samples(G);
    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, G.Hc, G.Mc, G.Cc, 2.0, rMax, xc, yc,
                                                 zc);

    tensorium_RG::bssn::print_constraint_norms(G, G.Hc, G.Mc, G.Cc, 2.0, rMax, xc, yc, zc);
    tensorium_RG::bssn::assert_invariants(G, "init.minkowski");
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

    bssn::apply_halos_grid<BoundaryClamp>(G);
    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);

    tensorium_RG::bssn::compute_tildeGamma_contracted(G); // Γ̃^i depuis γ̃_ij
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);   // R_ij

    double maxAbsR = 0.0;
    size_t nSamples = 0;
    double rMin = 1e300, rMax = 0.0;

    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;

                const double r = std::sqrt(x * x + y * y + z * z);

                const double r_min = 2.0;

                const double r_max = 0.45 * std::min({(I1 - I0 - 1) * G.dx, (J1 - J0 - 1) * G.dy,
                                                      (K1 - K0 - 1) * G.dz});
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.alpha.idx(i, j, k);

                const double chi = G.chi.ptr()[id];

                const double ginv_xx = chi * G.gamma_tilde_inv[XX].ptr()[id];
                const double ginv_xy = chi * G.gamma_tilde_inv[XY].ptr()[id];
                const double ginv_xz = chi * G.gamma_tilde_inv[XZ].ptr()[id];
                const double ginv_yy = chi * G.gamma_tilde_inv[YY].ptr()[id];
                const double ginv_yz = chi * G.gamma_tilde_inv[YZ].ptr()[id];
                const double ginv_zz = chi * G.gamma_tilde_inv[ZZ].ptr()[id];

                const double Rxx = G.Ricci[XX].ptr()[id];
                const double Rxy = G.Ricci[XY].ptr()[id];
                const double Rxz = G.Ricci[XZ].ptr()[id];
                const double Ryy = G.Ricci[YY].ptr()[id];
                const double Ryz = G.Ricci[YZ].ptr()[id];
                const double Rzz = G.Ricci[ZZ].ptr()[id];

                const double R = ginv_xx * Rxx + 2.0 * ginv_xy * Rxy + 2.0 * ginv_xz * Rxz +
                                 ginv_yy * Ryy + 2.0 * ginv_yz * Ryz + ginv_zz * Rzz;
                maxAbsR = std::max(maxAbsR, std::abs(R));
                nSamples++;

                if (nSamples == 1) {
                    printf("[DBG] chi=%g Rxx=%g Ryy=%g Rzz=%g\n", chi, Rxx, Ryy, Rzz);
                }
                rMin = std::min(rMin, r);
                rMax = std::max(rMax, r);
            }

    printf("[CHECK] samples=%zu, rMin=%g, rMax=%g\n", nSamples, rMin, rMax);
    printf("[CHECK] max |R| for r > %.3f = %.3e\n", rMin, maxAbsR);
    print_ricci_samples(G);

    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, G.Hc, G.Mc, G.Cc, 2.0, rMax, xc, yc,
                                                 zc);

    tensorium_RG::bssn::print_constraint_norms(G, G.Hc, G.Mc, G.Cc, 2.0, rMax, xc, yc, zc);
    tensorium_RG::bssn::assert_invariants(G, "init.schwarzschild");
}

template <typename T>
inline void binary_schwarzschild_isotropic_2centers(BSSNGridSoA<T> &G, T m1, T x1, T y1, T z1, T m2,
                                                    T x2, T y2, T z2, T r_floor = T(1e-6)) {
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

                T r1 = std::sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1));
                T r2 = std::sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2));

                if (r1 < r_floor)
                    r1 = r_floor;
                if (r2 < r_floor)
                    r2 = r_floor;

                const T psi = one + (m1 * half) / r1 + (m2 * half) / r2;

                const T chi = one / (psi * psi * psi * psi);
                const T alpha = one / (psi * psi);

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

    bssn::apply_halos_grid<BoundaryClamp>(G);

    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);

    double maxAbsR = 0.0;
    size_t nSamples = 0;
    double rMin = 1e300, rMax = 0.0;

    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);

                const double r1 = std::sqrt((x - double(x1)) * (x - double(x1)) +
                                            (y - double(y1)) * (y - double(y1)) +
                                            (z - double(z1)) * (z - double(z1)));
                const double r2 = std::sqrt((x - double(x2)) * (x - double(x2)) +
                                            (y - double(y2)) * (y - double(y2)) +
                                            (z - double(z2)) * (z - double(z2)));
                const double r = std::min(r1, r2);

                const double r_min = 2.0;
                const double r_max = 0.45 * std::min({(I1 - I0 - 1) * G.dx, (J1 - J0 - 1) * G.dy,
                                                      (K1 - K0 - 1) * G.dz});
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.alpha.idx(i, j, k);

                const double chi = G.chi.ptr()[id];

                const double ginv_xx = chi * G.gamma_tilde_inv[XX].ptr()[id];
                const double ginv_xy = chi * G.gamma_tilde_inv[XY].ptr()[id];
                const double ginv_xz = chi * G.gamma_tilde_inv[XZ].ptr()[id];
                const double ginv_yy = chi * G.gamma_tilde_inv[YY].ptr()[id];
                const double ginv_yz = chi * G.gamma_tilde_inv[YZ].ptr()[id];
                const double ginv_zz = chi * G.gamma_tilde_inv[ZZ].ptr()[id];

                const double Rxx = G.Ricci[XX].ptr()[id];
                const double Rxy = G.Ricci[XY].ptr()[id];
                const double Rxz = G.Ricci[XZ].ptr()[id];
                const double Ryy = G.Ricci[YY].ptr()[id];
                const double Ryz = G.Ricci[YZ].ptr()[id];
                const double Rzz = G.Ricci[ZZ].ptr()[id];

                const double R = ginv_xx * Rxx + 2.0 * ginv_xy * Rxy + 2.0 * ginv_xz * Rxz +
                                 ginv_yy * Ryy + 2.0 * ginv_yz * Ryz + ginv_zz * Rzz;

                maxAbsR = std::max(maxAbsR, std::abs(R));
                nSamples++;
                rMin = std::min(rMin, r);
                rMax = std::max(rMax, r);
            }

    printf("[CHECK] samples=%zu, rMin=%g, rMax=%g\n", nSamples, rMin, rMax);
    printf("[CHECK] max |R| for r > %.3f = %.3e\n", rMin, maxAbsR);
    print_ricci_samples(G);

    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, G.Hc, G.Mc, G.Cc, 2.0, rMax, 0.0, 0.0,
                                                 0.0);
    tensorium_RG::bssn::print_constraint_norms(G, G.Hc, G.Mc, G.Cc, 2.0, rMax, 0.0, 0.0, 0.0);
    tensorium_RG::bssn::assert_invariants(G, "init.binary_schwarzschild");
}

template <typename T>
inline void binary_bowen_york_puncture_init(BSSNGridSoA<T> &G, T m1, T x1, T y1, T z1,
                                            const T P1[3], const T S1[3], T m2, T x2, T y2, T z2,
                                            const T P2[3], const T S2[3], T r_floor = T(1e-6)) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const T one = T(1);
    const T zero = T(0);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);

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

                G.chi.ptr()[id] = one;
                G.alpha.ptr()[id] = one;
            }

    bssn::apply_halos_grid<BoundaryClamp>(G);

    fill_Atilde_bowen_york_binary(G, x1, y1, z1, P1, S1, x2, y2, z2, P2, S2, r_floor);
    bssn::apply_halos_grid<BoundaryClamp>(G);

    solve_lichnerowicz_u_SOR(G, m1, x1, y1, z1, m2, x2, y2, z2, r_floor, 400, T(1e-10), T(1.8));
    bssn::apply_halos_grid<BoundaryClamp>(G);

    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);
    print_ricci_samples(G);
    tensorium_RG::bssn::assert_invariants(G, "init.bowen_york");
}

} // namespace tensorium_RG::init
