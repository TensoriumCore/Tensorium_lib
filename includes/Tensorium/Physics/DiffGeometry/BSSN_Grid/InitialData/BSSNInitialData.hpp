#pragma once

#include "../Constraints/BSSNConstraintsGrid.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNInvariants.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "../Solvers/BSSNConstrainSolver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

/**
 * @file BSSNInitialData.hpp
 * @brief Analytic and Bowen–York puncture initial data generators for the BSSN grid.
 * @details
 * Provides ready-to-use setups for:
 * - Minkowski space (all fields equal to flat-space values).
 * - Single and binary isotropic Schwarzschild punctures with conformal factor \f$\psi = 1 + \sum
 * M_a/(2 r_a)\f$.
 * - Bowen–York extrinsic curvature contributions driven by specified momenta/spins.
 * - Lichnerowicz equation solver for binary data via successive over-relaxation.
 * Each initializer fills
 * \f$\alpha,\chi,\tilde{\gamma}_{ij},\tilde{A}_{ij},K,\beta^i,B^i,\tilde{\Gamma}^i\f$ and
 * recomputes geometric caches (Christoffels, Ricci) before asserting invariants.
 */

namespace tensorium_RG::init {

template <typename T>
inline void print_bssn_state_at(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                const char *label = "BSSN state") {
    const size_t id = G.alpha.idx(i, j, k);

    auto print_vec3 = [&](const char *name, const T v[3]) {
        printf("%s = [ % .6e  % .6e  % .6e ]\n", name, double(v[0]), double(v[1]), double(v[2]));
    };

    auto print_mat3 = [&](const char *name, const T m[3][3]) {
        printf("%s =\n", name);
        for (int a = 0; a < 3; ++a) {
            printf("  [");
            for (int b = 0; b < 3; ++b)
                printf(" % .6e", double(m[a][b]));
            printf(" ]\n");
        }
    };

    const T alpha = G.alpha.ptr()[id];
    const T chi = G.chi.ptr()[id];
    const T K = G.K.ptr()[id];

    const T beta[3] = {G.beta[0].ptr()[id], G.beta[1].ptr()[id], G.beta[2].ptr()[id]};

    const T B[3] = {G.B[0].ptr()[id], G.B[1].ptr()[id], G.B[2].ptr()[id]};

    const T GammaT[3] = {G.tildeGamma[0].ptr()[id], G.tildeGamma[1].ptr()[id],
                         G.tildeGamma[2].ptr()[id]};

    T gtilde[3][3] = {
        {G.gamma_tilde[XX].ptr()[id], G.gamma_tilde[XY].ptr()[id], G.gamma_tilde[XZ].ptr()[id]},
        {G.gamma_tilde[XY].ptr()[id], G.gamma_tilde[YY].ptr()[id], G.gamma_tilde[YZ].ptr()[id]},
        {G.gamma_tilde[XZ].ptr()[id], G.gamma_tilde[YZ].ptr()[id], G.gamma_tilde[ZZ].ptr()[id]}};

    T gtilde_inv[3][3] = {{G.gamma_tilde_inv[XX].ptr()[id], G.gamma_tilde_inv[XY].ptr()[id],
                           G.gamma_tilde_inv[XZ].ptr()[id]},
                          {G.gamma_tilde_inv[XY].ptr()[id], G.gamma_tilde_inv[YY].ptr()[id],
                           G.gamma_tilde_inv[YZ].ptr()[id]},
                          {G.gamma_tilde_inv[XZ].ptr()[id], G.gamma_tilde_inv[YZ].ptr()[id],
                           G.gamma_tilde_inv[ZZ].ptr()[id]}};

    T Atilde[3][3] = {{G.A_tilde[XX].ptr()[id], G.A_tilde[XY].ptr()[id], G.A_tilde[XZ].ptr()[id]},
                      {G.A_tilde[XY].ptr()[id], G.A_tilde[YY].ptr()[id], G.A_tilde[YZ].ptr()[id]},
                      {G.A_tilde[XZ].ptr()[id], G.A_tilde[YZ].ptr()[id], G.A_tilde[ZZ].ptr()[id]}};

    T x, y, z;
    G.coords(i, j, k, x, y, z);

    printf("\n==============================\n");
    printf("%s\n", label);
    printf("Grid index: (%zu, %zu, %zu)\n", i, j, k);
    printf("Coordinates: x=% .6e  y=% .6e  z=% .6e\n", double(x), double(y), double(z));
    printf("------------------------------\n");

    printf("alpha = % .6e\n", double(alpha));
    printf("chi   = % .6e\n", double(chi));
    printf("K     = % .6e\n", double(K));

    print_vec3("beta^i", beta);
    print_vec3("B^i", B);
    print_vec3("tildeGamma^i", GammaT);

    print_mat3("tilde_gamma_ij", gtilde);
    print_mat3("tilde_gamma^ij", gtilde_inv);
    print_mat3("Atilde_ij", Atilde);

    printf("==============================\n\n");
}
/**
 * @brief Print Ricci tensor samples at representative radii for diagnostic purposes.
 * @details Evaluates \f$R_{ij}\f$ at three locations (near, mid, far) along the x-axis to verify
 * that the initial data satisfy \f$R_{ij}=0\f$ away from punctures.  Used after
 * Minkowski/Schwarzschild set-up.
 */
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

/**
 * @brief Explicit 3x3 inversion of \f$\tilde{\gamma}_{ij}\f$ at a given grid index.
 * @details Used by initial-data routines after writing the conformal metric to keep the inverse
 * cache synchronized without invoking the global projector.
 */
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

/**
 * @brief Populate the grid with Minkowski (flat) initial data.
 * @param M Unused placeholder (kept for interface compatibility with Schwarzschild builders).
 * @param xc,yc,zc Center of the coordinate system used for diagnostics.
 * @param r_floor Minimum radius when sampling Ricci/constraint monitors.
 *
 * @details Sets
 * \f$\alpha=1,\chi=1,\tilde{\gamma}_{ij}=\delta_{ij},\tilde{A}_{ij}=0,K=0,\beta^i=B^i=0,\tilde{\Gamma}^i=0\f$.
 * After filling halos it recomputes Christoffels, Ricci, constraints, and enforces invariants.
 */
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
                    G.B[c].ptr()[id] = T(0);
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
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
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
    tensorium_RG::bssn::project_bssn_state(G);
}

/**
 * @brief Construct an isotropic Schwarzschild puncture with conformal factor \f$\psi=1+M/(2r)\f$.
 * @param M Puncture mass parameter.
 * @param xc,yc,zc Puncture center.
 * @param r_floor Regularization radius to avoid division by zero at the puncture.
 *
 * @details Sets \f$\alpha=\psi^{-2}\f$, \f$\chi=\psi^{-4}\f$,
 * \f$\tilde{\gamma}_{ij}=\delta_{ij}\f$, and
 * \f$\tilde{A}_{ij}=K=0\f$.  The routine clamps \f$r\ge r_{\text{floor}}\f$, applies halo BCs,
 * recomputes
 * \f$\tilde{\Gamma}^i\f$ and \f$R_{ij}\f$, then prints Ricci and constraint diagnostics.
 */
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
                    G.B[c].ptr()[id] = zero;
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

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
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
    tensorium_RG::bssn::project_bssn_state(G);
}

/**
 * @brief Superpose two isotropic Schwarzschild conformal factors for binary puncture data.
 * @details Uses \f$\psi = 1 + m_1/(2 r_1) + m_2/(2 r_2)\f$ with centers at \f$(x_a,y_a,z_a)\f$ and
 * enforces the algebraic constraints afterwards.  Generates conformally flat, time-symmetric data
 * (\f$K=0\f$).
 */
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
                    G.B[c].ptr()[id] = zero;
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

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);

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
    tensorium_RG::bssn::project_bssn_state(G);
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
                    G.B[c].ptr()[id] = zero;
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

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);

    fill_Atilde_bowen_york_binary(G, x1, y1, z1, P1, S1, x2, y2, z2, P2, S2, r_floor);
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);

    solve_lichnerowicz_u_SOR(G, m1, x1, y1, z1, m2, x2, y2, z2, r_floor, 2000, T(1e-10), T(1.8));
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i) {
        for (size_t j = J0; j < J1; ++j) {
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);
                const T      chi_val = G.chi.ptr()[id];
                G.alpha.ptr()[id] = std::sqrt(chi_val);
            }
        }
    }
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);

    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);

    print_ricci_samples(G);
    tensorium_RG::bssn::assert_invariants(G, "init.bowen_york");
    tensorium_RG::bssn::project_bssn_state(G);
}

template <typename T>
static inline void ks_kerr_r_H_l(T x, T y, T z, T M, T a, T &r, T &H, T l[3]) {
    const T rho2 = x * x + y * y + z * z;
    const T a2 = a * a;

    const T A = rho2 - a2;
    const T B = std::sqrt(A * A + T(4) * a2 * z * z);
    const T r2 = T(0.5) * (A + B);
    r = std::sqrt(std::max(r2, T(0)));

    const T denom = r * r + a2;
    const T inv_denom = T(1) / denom;

    l[0] = (r * x + a * y) * inv_denom;
    l[1] = (r * y - a * x) * inv_denom;
    l[2] = z / std::max(r, T(1e-30));

    const T r4 = r2 * r2;
    H = (M * r * r2) / (r4 + a2 * z * z);
}

template <typename T>
static inline void invert_sym_3x3(const T g_xx, const T g_xy, const T g_xz, const T g_yy,
                                  const T g_yz, const T g_zz, T &inv_xx, T &inv_xy, T &inv_xz,
                                  T &inv_yy, T &inv_yz, T &inv_zz, T &det) {
    const T a = g_xx, b = g_xy, c = g_xz;
    const T d = g_yy, e = g_yz, f = g_zz;

    det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    const T inv_det = T(1) / det;

    inv_xx = (d * f - e * e) * inv_det;
    inv_xy = (c * e - b * f) * inv_det;
    inv_xz = (b * e - c * d) * inv_det;
    inv_yy = (a * f - c * c) * inv_det;
    inv_yz = (b * c - a * e) * inv_det;
    inv_zz = (a * d - b * b) * inv_det;
}

template <typename T>
inline void kerr_schild_single(BSSNGridSoA<T> &G, T M, T a, T xc = T(0), T yc = T(0), T zc = T(0),
                               T r_floor = T(1e-6)) {

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const T one = T(1);
    const T zero = T(0);

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

                const T rr = std::sqrt(x * x + y * y + z * z);
                if (rr < r_floor) {
                    const T s = r_floor / std::max(rr, T(1e-30));
                    x *= s;
                    y *= s;
                    z *= s;
                }

                T r, H;
                T l[3];
                ks_kerr_r_H_l(x, y, z, M, a, r, H, l);

                const T alpha = one / std::sqrt(one + T(2) * H);

                const T g_xx = one + T(2) * H * l[0] * l[0];
                const T g_xy = T(2) * H * l[0] * l[1];
                const T g_xz = T(2) * H * l[0] * l[2];
                const T g_yy = one + T(2) * H * l[1] * l[1];
                const T g_yz = T(2) * H * l[1] * l[2];
                const T g_zz = one + T(2) * H * l[2] * l[2];

                T ginv_xx, ginv_xy, ginv_xz, ginv_yy, ginv_yz, ginv_zz, detg;
                invert_sym_3x3(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, ginv_xx, ginv_xy, ginv_xz,
                               ginv_yy, ginv_yz, ginv_zz, detg);

                const T chi = std::pow(detg, T(-1) / T(3));
                const T inv_chi = one / chi;

                G.alpha.ptr()[id] = alpha;
                G.chi.ptr()[id] = chi;

                const T beta_cov_x = T(2) * H * l[0];
                const T beta_cov_y = T(2) * H * l[1];
                const T beta_cov_z = T(2) * H * l[2];

                const T beta_x = ginv_xx * beta_cov_x + ginv_xy * beta_cov_y + ginv_xz * beta_cov_z;
                const T beta_y = ginv_xy * beta_cov_x + ginv_yy * beta_cov_y + ginv_yz * beta_cov_z;
                const T beta_z = ginv_xz * beta_cov_x + ginv_yz * beta_cov_y + ginv_zz * beta_cov_z;

                G.beta[0].ptr()[id] = beta_x;
                G.beta[1].ptr()[id] = beta_y;
                G.beta[2].ptr()[id] = beta_z;

                G.B[0].ptr()[id] = zero;
                G.B[1].ptr()[id] = zero;
                G.B[2].ptr()[id] = zero;

                G.gamma_tilde[XX].ptr()[id] = chi * g_xx;
                G.gamma_tilde[XY].ptr()[id] = chi * g_xy;
                G.gamma_tilde[XZ].ptr()[id] = chi * g_xz;
                G.gamma_tilde[YY].ptr()[id] = chi * g_yy;
                G.gamma_tilde[YZ].ptr()[id] = chi * g_yz;
                G.gamma_tilde[ZZ].ptr()[id] = chi * g_zz;

                G.gamma_tilde_inv[XX].ptr()[id] = inv_chi * ginv_xx;
                G.gamma_tilde_inv[XY].ptr()[id] = inv_chi * ginv_xy;
                G.gamma_tilde_inv[XZ].ptr()[id] = inv_chi * ginv_xz;
                G.gamma_tilde_inv[YY].ptr()[id] = inv_chi * ginv_yy;
                G.gamma_tilde_inv[YZ].ptr()[id] = inv_chi * ginv_yz;
                G.gamma_tilde_inv[ZZ].ptr()[id] = inv_chi * ginv_zz;

                G.K.ptr()[id] = zero;

                for (int c = 0; c < 6; ++c)
                    G.A_tilde[c].ptr()[id] = zero;
            }

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::compute_tildeGamma_full(G, G.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);

#pragma omp parallel for collapse(3)
    for (size_t i = I0 + 2; i < I1 - 2; ++i)
        for (size_t j = J0 + 2; j < J1 - 2; ++j)
            for (size_t k = K0 + 2; k < K1 - 2; ++k) {

                const size_t id = G.alpha.idx(i, j, k);

                double Dbeta[3][3];
                tensorium_RG::fd::covariant_D_vec(G, G.beta, i, j, k, Dbeta,
                                                  tensorium_RG::fd::Variance::Up,
                                                  tensorium_RG::fd::Variance::Down);

                const T alpha = G.alpha.ptr()[id];
                const T inv2a = T(0.5) / alpha;

                const T K_xx = inv2a * (Dbeta[0][0] + Dbeta[0][0]);
                const T K_xy = inv2a * (Dbeta[0][1] + Dbeta[1][0]);
                const T K_xz = inv2a * (Dbeta[0][2] + Dbeta[2][0]);
                const T K_yy = inv2a * (Dbeta[1][1] + Dbeta[1][1]);
                const T K_yz = inv2a * (Dbeta[1][2] + Dbeta[2][1]);
                const T K_zz = inv2a * (Dbeta[2][2] + Dbeta[2][2]);

                const T chi = G.chi.ptr()[id];
                const T inv_chi = one / chi;

                const T g_xx = inv_chi * G.gamma_tilde[XX].ptr()[id];
                const T g_xy = inv_chi * G.gamma_tilde[XY].ptr()[id];
                const T g_xz = inv_chi * G.gamma_tilde[XZ].ptr()[id];
                const T g_yy = inv_chi * G.gamma_tilde[YY].ptr()[id];
                const T g_yz = inv_chi * G.gamma_tilde[YZ].ptr()[id];
                const T g_zz = inv_chi * G.gamma_tilde[ZZ].ptr()[id];

                T ginv_xx, ginv_xy, ginv_xz, ginv_yy, ginv_yz, ginv_zz, detg;
                invert_sym_3x3(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz, ginv_xx, ginv_xy, ginv_xz,
                               ginv_yy, ginv_yz, ginv_zz, detg);

                const T Ktr = ginv_xx * K_xx + T(2) * ginv_xy * K_xy + T(2) * ginv_xz * K_xz +
                              ginv_yy * K_yy + T(2) * ginv_yz * K_yz + ginv_zz * K_zz;

                G.K.ptr()[id] = Ktr;

                const T one_third = T(1) / T(3);

                const T A_xx = K_xx - one_third * g_xx * Ktr;
                const T A_xy = K_xy - one_third * g_xy * Ktr;
                const T A_xz = K_xz - one_third * g_xz * Ktr;
                const T A_yy = K_yy - one_third * g_yy * Ktr;
                const T A_yz = K_yz - one_third * g_yz * Ktr;
                const T A_zz = K_zz - one_third * g_zz * Ktr;

                G.A_tilde[XX].ptr()[id] = chi * A_xx;
                G.A_tilde[XY].ptr()[id] = chi * A_xy;
                G.A_tilde[XZ].ptr()[id] = chi * A_xz;
                G.A_tilde[YY].ptr()[id] = chi * A_yy;
                G.A_tilde[YZ].ptr()[id] = chi * A_yz;
                G.A_tilde[ZZ].ptr()[id] = chi * A_zz;
            }

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);

    double maxAbsH = 0.0;
    double maxAbsM = 0.0;
    double maxAbsC = 0.0;

    double rMin = 1e300;
    double rMax = 0.0;
    size_t nSamples = 0;

    const double r_min_cut = 2.0; 
    const double r_max_cut =
        0.45 * std::min({(double)(I1 - I0 - 1) * (double)G.dx, (double)(J1 - J0 - 1) * (double)G.dy,
                         (double)(K1 - K0 - 1) * (double)G.dz});

    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k) {

                double x, y, z;
                G.coords(i, j, k, x, y, z);
                x -= (double)xc;
                y -= (double)yc;
                z -= (double)zc;

                const double r = std::sqrt(x * x + y * y + z * z);
                if (r <= r_min_cut || r >= r_max_cut)
                    continue;

                rMin = std::min(rMin, r);
                rMax = std::max(rMax, r);
                nSamples++;
            }

    if (nSamples == 0) {
        rMin = r_min_cut;
        rMax = r_max_cut;
    }

    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, G.Hc, G.Mc, G.Cc, rMin, rMax,
                                                 (double)xc, (double)yc, (double)zc);

    tensorium_RG::bssn::print_constraint_norms(G, G.Hc, G.Mc, G.Cc, rMin, rMax, (double)xc,
                                               (double)yc, (double)zc);

    print_ricci_samples(G);
    tensorium_RG::bssn::assert_invariants(G, "init.kerr_schild_single");
    tensorium_RG::bssn::project_bssn_state(G);
}
} // namespace tensorium_RG::init
