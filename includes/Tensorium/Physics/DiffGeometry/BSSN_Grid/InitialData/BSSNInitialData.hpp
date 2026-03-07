#pragma once

#include "../Constraints/BSSNConstraintsGrid.hpp"
#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNGamma.hpp"
#include "../Geometry/BSSNInvariants.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include "../Geometry/BSSNRicci.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "../Solvers/BSSNConstrainSolver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
#    include "TwoPunctures.h"
#endif

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

template <typename T> struct ConstraintScratch {
    Field3D<T> H;
    Field3D<T> M[3];
    Field3D<T> C[3];
};

template <typename T>
inline ConstraintScratch<T> make_constraint_scratch(const BSSNGridSoA<T> &G) {
    ConstraintScratch<T> scratch;
    scratch.H = tensorium_RG::make_field(G.alpha.st);
    for (int q = 0; q < 3; ++q) {
        scratch.M[q] = tensorium_RG::make_field(G.alpha.st);
        scratch.C[q] = tensorium_RG::make_field(G.alpha.st);
    }
    return scratch;
}

template <typename T> inline void zero_z4c_fields(BSSNGridSoA<T> &G) {
    const size_t total = G.Theta.st.nx_tot * G.Theta.st.ny_tot * G.Theta.st.nz_tot;
    std::fill_n(G.Theta.ptr(), total, T(0));
    for (int c = 0; c < 3; ++c)
        std::fill_n(G.Z[c].ptr(), total, T(0));
}

namespace detail {

template <typename Grid, typename = void> struct HaloFinalizer {
    static inline void run(Grid &G) {
        tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(G);
    }
};

template <typename Grid> struct HaloFinalizer<Grid, std::void_t<decltype(std::declval<Grid &>().apply_halos())>> {
    static inline void run(Grid &G) { G.apply_halos(); }
};

} // namespace detail

template <typename T> void sanitize_bssn_initial_state(BSSNGridSoA<T> &G) {
    const size_t nx_tot = G.alpha.st.nx_tot;
    const size_t ny_tot = G.alpha.st.ny_tot;
    const size_t nz_tot = G.alpha.st.nz_tot;

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < nx_tot; ++i) {
        for (size_t j = 0; j < ny_tot; ++j) {
            for (size_t k = 0; k < nz_tot; ++k) {
                const size_t id = G.chi.idx(i, j, k);
                const T       chi = G.chi.ptr()[id];
                // Fix import error: convert physical A_ij to conformal A_tilde_ij by multiplying by chi
                for (int s = 0; s < 6; ++s)
                    G.A_tilde[s].ptr()[id] *= chi;
            }
        }
    }

    // Ensure Z4c constraint damping variables start at zero
    zero_z4c_fields(G);

    tensorium_RG::bssn::project_bssn_state(G);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i) {
        for (size_t j = J0; j < J1; ++j) {
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.gamma_tilde[XX].idx(i, j, k);
                T            gamma_conn[3];
                tensorium_RG::bssn::compute_contracted_gamma_from_metric(G, i, j, k, gamma_conn);
                G.tildeGamma[0].ptr()[id] = gamma_conn[0];
                G.tildeGamma[1].ptr()[id] = gamma_conn[1];
                G.tildeGamma[2].ptr()[id] = gamma_conn[2];
            }
        }
    }

    detail::HaloFinalizer<BSSNGridSoA<T>>::run(G);
}

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
    zero_z4c_fields(G);

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
    auto constraints = make_constraint_scratch(G);
    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, constraints.H, constraints.M,
                                                 constraints.C, 2.0, rMax, xc, yc, zc);

    tensorium_RG::bssn::print_constraint_norms(G, constraints.H, constraints.M, constraints.C, 2.0,
                                              rMax, xc, yc, zc);
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
    zero_z4c_fields(G);

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

    auto constraints = make_constraint_scratch(G);
    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, constraints.H, constraints.M,
                                                 constraints.C, 2.0, rMax, xc, yc, zc);

    tensorium_RG::bssn::print_constraint_norms(G, constraints.H, constraints.M, constraints.C, 2.0,
                                              rMax, xc, yc, zc);
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
    zero_z4c_fields(G);

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

    auto constraints = make_constraint_scratch(G);
    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, constraints.H, constraints.M,
                                                 constraints.C, 2.0, rMax, 0.0, 0.0, 0.0);
    tensorium_RG::bssn::print_constraint_norms(G, constraints.H, constraints.M, constraints.C, 2.0,
                                              rMax, 0.0, 0.0, 0.0);
    tensorium_RG::bssn::assert_invariants(G, "init.binary_schwarzschild");
    tensorium_RG::bssn::project_bssn_state(G);
}

template <typename T>
inline void binary_bowen_york_puncture_init(BSSNGridSoA<T> &G, T m1, T x1, T y1, T z1,
                                            const T P1[3], const T S1[3], T m2, T x2, T y2, T z2,
                                            const T P2[3], const T S2[3], T r_floor = T(1e-6)) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    zero_z4c_fields(G);

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

    solve_lichnerowicz_u_SOR(G, m1, x1, y1, z1, m2, x2, y2, z2, r_floor, 2500, T(1e-10), T(1.8));
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

    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);

    print_ricci_samples(G);
    tensorium_RG::bssn::assert_invariants(G, "init.bowen_york");
    tensorium_RG::bssn::project_bssn_state(G);
}

template <typename T>
inline T sample_trilinear_field(const BSSNGridSoA<T> &src, const Field3D<T> &field, T x, T y, T z) {
    size_t I0, I1, J0, J1, K0, K1;
    src.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ix_min = I0;
    const size_t iy_min = J0;
    const size_t iz_min = K0;
    const size_t ix_max = I1 - 1;
    const size_t iy_max = J1 - 1;
    const size_t iz_max = K1 - 1;

    auto axis_weights = [](T coord, T origin, T spacing, size_t a_min, size_t a_max, size_t &a0,
                           size_t &a1, T &w) {
        const T u = (coord - origin) / spacing + T(a_min);
        if (u <= T(a_min)) {
            a0 = a_min;
            a1 = std::min(a_min + size_t(1), a_max);
            w = T(0);
            return;
        }
        if (u >= T(a_max)) {
            a1 = a_max;
            a0 = (a_max > a_min) ? (a_max - size_t(1)) : a_max;
            w = T(1);
            return;
        }
        const T uf = std::floor(u);
        a0 = static_cast<size_t>(uf);
        a1 = std::min(a0 + size_t(1), a_max);
        w = std::clamp(u - T(a0), T(0), T(1));
    };

    size_t i0, i1, j0, j1, k0, k1;
    T      tx, ty, tz;
    axis_weights(x, src.x0, src.dx, ix_min, ix_max, i0, i1, tx);
    axis_weights(y, src.y0, src.dy, iy_min, iy_max, j0, j1, ty);
    axis_weights(z, src.z0, src.dz, iz_min, iz_max, k0, k1, tz);

    auto at = [&](size_t i, size_t j, size_t k) -> T { return field.ptr()[field.idx(i, j, k)]; };

    const T c000 = at(i0, j0, k0);
    const T c100 = at(i1, j0, k0);
    const T c010 = at(i0, j1, k0);
    const T c110 = at(i1, j1, k0);
    const T c001 = at(i0, j0, k1);
    const T c101 = at(i1, j0, k1);
    const T c011 = at(i0, j1, k1);
    const T c111 = at(i1, j1, k1);

    const T c00 = c000 * (T(1) - tx) + c100 * tx;
    const T c10 = c010 * (T(1) - tx) + c110 * tx;
    const T c01 = c001 * (T(1) - tx) + c101 * tx;
    const T c11 = c011 * (T(1) - tx) + c111 * tx;
    const T c0 = c00 * (T(1) - ty) + c10 * ty;
    const T c1 = c01 * (T(1) - ty) + c11 * ty;
    return c0 * (T(1) - tz) + c1 * tz;
}

template <typename T>
static inline void invert_sym_3x3(const T g_xx, const T g_xy, const T g_xz, const T g_yy,
                                  const T g_yz, const T g_zz, T &inv_xx, T &inv_xy, T &inv_xz,
                                  T &inv_yy, T &inv_yz, T &inv_zz, T &det);

template <typename T>
inline void binary_bowen_york_puncture_interpolated_init(BSSNGridSoA<T> &G, T m1, T x1, T y1, T z1,
                                                         const T P1[3], const T S1[3], T m2, T x2,
                                                         T y2, T z2, const T P2[3], const T S2[3],
                                                         size_t interp_seed_n = 64,
                                                         T r_floor = T(1e-6)) {
#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
    (void)interp_seed_n;
    (void)r_floor;
    auto now_s = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    };
    const double t_init_begin = now_s();
    int          tp_verbose = 0;
    if (const char *v = std::getenv("TENSORIUM_TWOPUNCTURES_VERBOSE"))
        tp_verbose = (std::atoi(v) != 0) ? 1 : 0;
    if (const char *v = std::getenv("TENSORIUM_MOVING_PUNCTURE_TP_VERBOSE"))
        tp_verbose = (std::atoi(v) != 0) ? 1 : 0;
    auto parse_env_int = [](const char *name, int fallback, int min_value) {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0')
            return fallback;
        if (parsed < long(min_value))
            return min_value;
        if (parsed > long(std::numeric_limits<int>::max()))
            return std::numeric_limits<int>::max();
        return int(parsed);
    };
    auto parse_env_real = [](const char *name, double fallback, double min_value) {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw || *end != '\0')
            return fallback;
        if (!std::isfinite(parsed))
            return fallback;
        return (parsed < min_value) ? min_value : parsed;
    };
    const int    tp_npoints_A =
        parse_env_int("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_A", 30, 8);
    const int tp_npoints_B =
        parse_env_int("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_B", 30, 8);
    const int tp_npoints_phi =
        parse_env_int("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_PHI", 16, 4);
    const double tp_newton_tol =
        parse_env_real("TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_TOL", 1.0e-10, 1.0e-16);
    const int tp_newton_maxit =
        parse_env_int("TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_MAXIT", 5, 1);
    const double tp_epsilon =
        parse_env_real("TENSORIUM_MOVING_PUNCTURE_TP_EPSILON", 1.0e-6, 1.0e-16);
    int omp_threads = 1;
    if (const char *threads_env = std::getenv("OMP_NUM_THREADS")) {
        const int parsed = std::atoi(threads_env);
        if (parsed > 0)
            omp_threads = parsed;
    }
    const int tp_threads = parse_env_int("TENSORIUM_MOVING_PUNCTURE_TP_THREADS", omp_threads, 1);
#if defined(TENSORIUM_TWOPUNCTURES_OMP)
    const char *tp_prev_omp_env_raw = std::getenv("OMP_NUM_THREADS");
    char        tp_prev_omp_env[64] = {};
    bool        tp_prev_omp_env_valid = false;
    if (tp_prev_omp_env_raw != nullptr && *tp_prev_omp_env_raw != '\0') {
        std::snprintf(tp_prev_omp_env, sizeof(tp_prev_omp_env), "%s", tp_prev_omp_env_raw);
        tp_prev_omp_env_valid = true;
    }
    char tp_threads_buf[32];
    std::snprintf(tp_threads_buf, sizeof(tp_threads_buf), "%d", tp_threads);
#if !defined(_WIN32)
    setenv("OMP_NUM_THREADS", tp_threads_buf, 1);
#endif
#endif
    std::printf("[init.interpolate][step 1/7] setup TwoPunctures backend "
                "(omp_threads=%d, tp_threads=%d, tp_omp=%d)\n",
                omp_threads, tp_threads,
#if defined(TENSORIUM_TWOPUNCTURES_OMP)
                1
#else
                0
#endif
    );
    std::printf("[init.interpolate][cfg] npoints_A=%d npoints_B=%d npoints_phi=%d "
                "Newton_tol=%.3e Newton_maxit=%d TP_epsilon=%.3e verbose=%d\n",
                tp_npoints_A, tp_npoints_B, tp_npoints_phi, tp_newton_tol, tp_newton_maxit,
                tp_epsilon, tp_verbose);
    std::fflush(stdout);

    // TwoPunctures assumes punctures at (+/- par_b, 0, 0) plus a common center offset.
    // For this path we require x-aligned punctures and identical y/z coordinates.
    const T center_x = T(0.5) * (x1 + x2);
    const T center_y = T(0.5) * (y1 + y2);
    const T center_z = T(0.5) * (z1 + z2);
    const T b = T(0.5) * std::abs(x2 - x1);
    const T yz_mismatch = std::abs(y1 - y2) + std::abs(z1 - z2);
    if (b <= T(0) || yz_mismatch > T(1e-12)) {
        std::printf("[init.interpolate][warn] invalid puncture layout for TwoPunctures backend; "
                    "falling back to Bowen-York init\n");
        binary_bowen_york_puncture_init(G, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2, T(1e-6));
        return;
    }

    const bool plus_is_second = (x2 > x1);
    const T m_plus = plus_is_second ? m2 : m1;
    const T m_minus = plus_is_second ? m1 : m2;
    const T *P_plus = plus_is_second ? P2 : P1;
    const T *P_minus = plus_is_second ? P1 : P2;
    const T *S_plus = plus_is_second ? S2 : S1;
    const T *S_minus = plus_is_second ? S1 : S2;

    TwoPunctures_params_set_default();
    TwoPunctures_params_set_Int(const_cast<char *>("verbose"), tp_verbose);
    TwoPunctures_params_set_Int(const_cast<char *>("give_bare_mass"), 1);
    TwoPunctures_params_set_Int(const_cast<char *>("grid_setup_method"), evaluation);
    TwoPunctures_params_set_Int(const_cast<char *>("initial_lapse"), psin);
    TwoPunctures_params_set_Real(const_cast<char *>("initial_lapse_psi_exponent"), -2.0);
    TwoPunctures_params_set_Int(const_cast<char *>("conformal_state"), 1);
    TwoPunctures_params_set_Real(const_cast<char *>("par_b"), double(b));
    TwoPunctures_params_set_Real(const_cast<char *>("par_m_plus"), double(m_plus));
    TwoPunctures_params_set_Real(const_cast<char *>("par_m_minus"), double(m_minus));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus1"), double(P_plus[0]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus2"), double(P_plus[1]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus3"), double(P_plus[2]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus1"), double(P_minus[0]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus2"), double(P_minus[1]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus3"), double(P_minus[2]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus1"), double(S_plus[0]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus2"), double(S_plus[1]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus3"), double(S_plus[2]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus1"), double(S_minus[0]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus2"), double(S_minus[1]));
    TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus3"), double(S_minus[2]));
    TwoPunctures_params_set_Real(const_cast<char *>("center_offset1"), double(center_x));
    TwoPunctures_params_set_Real(const_cast<char *>("center_offset2"), double(center_y));
    TwoPunctures_params_set_Real(const_cast<char *>("center_offset3"), double(center_z));
    // Match the GRChombo BinaryBH TwoPunctures defaults.
    TwoPunctures_params_set_Int(const_cast<char *>("npoints_A"), tp_npoints_A);
    TwoPunctures_params_set_Int(const_cast<char *>("npoints_B"), tp_npoints_B);
    TwoPunctures_params_set_Int(const_cast<char *>("npoints_phi"), tp_npoints_phi);
    TwoPunctures_params_set_Real(const_cast<char *>("Newton_tol"), tp_newton_tol);
    TwoPunctures_params_set_Int(const_cast<char *>("Newton_maxit"), tp_newton_maxit);
    TwoPunctures_params_set_Real(const_cast<char *>("TP_epsilon"), tp_epsilon);

    std::printf("[init.interpolate][step 2/7] solving TwoPunctures spectral system...\n");
    std::fflush(stdout);
    const double t_solve_begin = now_s();
    ini_data *tp_data = TwoPunctures_make_initial_data();
    const double t_solve_end = now_s();
    std::printf("[init.interpolate][step 2/7] solve done in %.3fs\n",
                t_solve_end - t_solve_begin);
    std::fflush(stdout);
    if (tp_data == nullptr) {
        std::printf("[init.interpolate][warn] TwoPunctures solver failed; "
                    "falling back to Bowen-York init\n");
        binary_bowen_york_puncture_init(G, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2, T(1e-6));
        return;
    }

    const int nx = static_cast<int>(G.dims.nx);
    const int ny = static_cast<int>(G.dims.ny);
    const int nz = static_cast<int>(G.dims.nz);
    const size_t nt = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    const double bytes = double(nt) * double(28u) * double(sizeof(double));
    std::printf("[init.interpolate][step 3/7] preparing interpolation buffers (nt=%zu, %.2f MiB)\n",
                nt, bytes / (1024.0 * 1024.0));
    std::fflush(stdout);

    std::vector<double> xs(static_cast<size_t>(nx));
    std::vector<double> ys(static_cast<size_t>(ny));
    std::vector<double> zs(static_cast<size_t>(nz));
    for (int i = 0; i < nx; ++i)
        xs[static_cast<size_t>(i)] = double(G.x0 + T(i) * G.dx);
    for (int j = 0; j < ny; ++j)
        ys[static_cast<size_t>(j)] = double(G.y0 + T(j) * G.dy);
    for (int k = 0; k < nz; ++k)
        zs[static_cast<size_t>(k)] = double(G.z0 + T(k) * G.dz);

    std::vector<double> alp(nt), psi(nt), psix(nt), psiy(nt), psiz(nt), psixx(nt), psixy(nt),
        psixz(nt), psiyy(nt), psiyz(nt), psizz(nt);
    std::vector<double> gxx(nt), gxy(nt), gxz(nt), gyy(nt), gyz(nt), gzz(nt);
    std::vector<double> kxx(nt), kxy(nt), kxz(nt), kyy(nt), kyz(nt), kzz(nt);

    int imin[3] = {0, 0, 0};
    int imax[3] = {nx, ny, nz};
    int nxyz[3] = {nx, ny, nz};
    std::printf("[init.interpolate][step 4/7] spectral->Cartesian interpolation...\n");
    std::fflush(stdout);
    const double t_interp_begin = now_s();
    TwoPunctures_Cartesian_interpolation(
        tp_data, imin, imax, nxyz, xs.data(), ys.data(), zs.data(), alp.data(), psi.data(),
        psix.data(), psiy.data(), psiz.data(), psixx.data(), psixy.data(), psixz.data(),
        psiyy.data(), psiyz.data(), psizz.data(), gxx.data(), gxy.data(), gxz.data(), gyy.data(),
        gyz.data(), gzz.data(), kxx.data(), kxy.data(), kxz.data(), kyy.data(), kyz.data(),
        kzz.data());
#if defined(TENSORIUM_TWOPUNCTURES_OMP) && !defined(_WIN32)
    if (tp_prev_omp_env_valid)
        setenv("OMP_NUM_THREADS", tp_prev_omp_env, 1);
    else
        unsetenv("OMP_NUM_THREADS");
#endif
    const double t_interp_end = now_s();
    std::printf("[init.interpolate][step 4/7] interpolation done in %.3fs\n",
                t_interp_end - t_interp_begin);
    std::fflush(stdout);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    const T one = T(1);
    const T eps = T(1e-20);

    std::printf("[init.interpolate][step 5/7] mapping TP fields -> BSSN state...\n");
    std::fflush(stdout);
    const double t_map_begin = now_s();
#pragma omp parallel for collapse(3)
    for (int kk = 0; kk < nz; ++kk) {
        for (int jj = 0; jj < ny; ++jj) {
            for (int ii = 0; ii < nx; ++ii) {
                const size_t ind = static_cast<size_t>(ii) + static_cast<size_t>(nx) *
                                                             (static_cast<size_t>(jj) +
                                                              static_cast<size_t>(ny) *
                                                                  static_cast<size_t>(kk));
                const size_t i = I0 + static_cast<size_t>(ii);
                const size_t j = J0 + static_cast<size_t>(jj);
                const size_t k = K0 + static_cast<size_t>(kk);
                const size_t id = G.alpha.idx(i, j, k);

                // TwoPunctures C API returns conformal metric components (gbar_ij) and the
                // static puncture conformal factor psi separately. Reconstruct physical gamma_ij.
                const T psi_static = std::max(T(psi[ind]), T(1e-15));
                const T psi2 = psi_static * psi_static;
                const T psi4 = psi2 * psi2;

                const T g_xx = psi4 * T(gxx[ind]);
                const T g_xy = psi4 * T(gxy[ind]);
                const T g_xz = psi4 * T(gxz[ind]);
                const T g_yy = psi4 * T(gyy[ind]);
                const T g_yz = psi4 * T(gyz[ind]);
                const T g_zz = psi4 * T(gzz[ind]);

                const T detg = g_xx * (g_yy * g_zz - g_yz * g_yz) -
                               g_xy * (g_xy * g_zz - g_xz * g_yz) +
                               g_xz * (g_xy * g_yz - g_xz * g_yy);
                const T det_pos = std::max(detg, eps);
                const T chi = std::pow(det_pos, T(-1) / T(3));

                const T K_xx = T(kxx[ind]);
                const T K_xy = T(kxy[ind]);
                const T K_xz = T(kxz[ind]);
                const T K_yy = T(kyy[ind]);
                const T K_yz = T(kyz[ind]);
                const T K_zz = T(kzz[ind]);
                const T Abar_xx = chi * K_xx;
                const T Abar_xy = chi * K_xy;
                const T Abar_xz = chi * K_xz;
                const T Abar_yy = chi * K_yy;
                const T Abar_yz = chi * K_yz;
                const T Abar_zz = chi * K_zz;
                const T one_third = T(1) / T(3);
                const T trAbar = Abar_xx + Abar_yy + Abar_zz;

                G.alpha.ptr()[id] = std::max(T(alp[ind]), T(1e-12));
                G.chi.ptr()[id] = std::max(chi, T(1e-16));
                G.K.ptr()[id] = T(0);
                G.Theta.ptr()[id] = T(0);

                for (int c = 0; c < 3; ++c) {
                    G.beta[c].ptr()[id] = T(0);
                    G.B[c].ptr()[id] = T(0);
                    G.tildeGamma[c].ptr()[id] = T(0);
                    G.Z[c].ptr()[id] = T(0);
                }

                G.gamma_tilde[XX].ptr()[id] = one;
                G.gamma_tilde[XY].ptr()[id] = T(0);
                G.gamma_tilde[XZ].ptr()[id] = T(0);
                G.gamma_tilde[YY].ptr()[id] = one;
                G.gamma_tilde[YZ].ptr()[id] = T(0);
                G.gamma_tilde[ZZ].ptr()[id] = one;

                G.gamma_tilde_inv[XX].ptr()[id] = one;
                G.gamma_tilde_inv[XY].ptr()[id] = T(0);
                G.gamma_tilde_inv[XZ].ptr()[id] = T(0);
                G.gamma_tilde_inv[YY].ptr()[id] = one;
                G.gamma_tilde_inv[YZ].ptr()[id] = T(0);
                G.gamma_tilde_inv[ZZ].ptr()[id] = one;

                G.A_tilde[XX].ptr()[id] = Abar_xx - one_third * trAbar;
                G.A_tilde[XY].ptr()[id] = Abar_xy;
                G.A_tilde[XZ].ptr()[id] = Abar_xz;
                G.A_tilde[YY].ptr()[id] = Abar_yy - one_third * trAbar;
                G.A_tilde[YZ].ptr()[id] = Abar_yz;
                G.A_tilde[ZZ].ptr()[id] = Abar_zz - one_third * trAbar;
            }
        }
    }
    const double t_map_end = now_s();
    std::printf("[init.interpolate][step 5/7] mapping done in %.3fs\n", t_map_end - t_map_begin);
    std::fflush(stdout);

    std::printf("[init.interpolate][step 6/7] cleanup TP workspace + enforce constraints...\n");
    std::fflush(stdout);
    const double t_post_begin = now_s();
    TwoPunctures_finalise(tp_data);

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;
    proj_cfg.resync_contracted_gamma = true;
    tensorium_RG::bssn::project_bssn_state(G, proj_cfg);
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);
    tensorium_RG::bssn::assert_invariants(G, "init.twopunctures_interpolated");
    const double t_post_end = now_s();
    std::printf("[init.interpolate][step 6/7] post-process done in %.3fs\n",
                t_post_end - t_post_begin);
    std::printf("[init.interpolate][step 7/7] complete total=%.3fs\n", t_post_end - t_init_begin);
    std::fflush(stdout);
#else
    const size_t seed_n = std::max<size_t>(24, interp_seed_n);

    const T Lx = G.dx * T(G.dims.nx);
    const T Ly = G.dy * T(G.dims.ny);
    const T Lz = G.dz * T(G.dims.nz);

    const T seed_dx = Lx / T(seed_n);
    const T seed_dy = Ly / T(seed_n);
    const T seed_dz = Lz / T(seed_n);

    BSSNGridSoA<T> seed(seed_n, seed_n, seed_n, G.dims.ng, seed_dx, seed_dy, seed_dz);

    const T x_lo = G.x0 - T(0.5) * G.dx;
    const T y_lo = G.y0 - T(0.5) * G.dy;
    const T z_lo = G.z0 - T(0.5) * G.dz;
    seed.x0 = x_lo + T(0.5) * seed_dx;
    seed.y0 = y_lo + T(0.5) * seed_dy;
    seed.z0 = z_lo + T(0.5) * seed_dz;

    std::printf("[init.interpolate] seed grid=%zux%zux%zu spacing=(%.6f, %.6f, %.6f)\n", seed_n,
                seed_n, seed_n, double(seed_dx), double(seed_dy), double(seed_dz));

    binary_bowen_york_puncture_init(seed, m1, x1, y1, z1, P1, S1, m2, x2, y2, z2, P2, S2, r_floor);

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i) {
        for (size_t j = J0; j < J1; ++j) {
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);
                T            x, y, z;
                G.coords(i, j, k, x, y, z);

                G.alpha.ptr()[id] = sample_trilinear_field(seed, seed.alpha, x, y, z);
                G.chi.ptr()[id] = sample_trilinear_field(seed, seed.chi, x, y, z);
                G.K.ptr()[id] = sample_trilinear_field(seed, seed.K, x, y, z);
                G.Theta.ptr()[id] = sample_trilinear_field(seed, seed.Theta, x, y, z);

                for (int c = 0; c < 3; ++c) {
                    G.beta[c].ptr()[id] = sample_trilinear_field(seed, seed.beta[c], x, y, z);
                    G.B[c].ptr()[id] = sample_trilinear_field(seed, seed.B[c], x, y, z);
                    G.tildeGamma[c].ptr()[id] =
                        sample_trilinear_field(seed, seed.tildeGamma[c], x, y, z);
                    G.Z[c].ptr()[id] = sample_trilinear_field(seed, seed.Z[c], x, y, z);
                }

                for (int s = 0; s < 6; ++s) {
                    G.gamma_tilde[s].ptr()[id] =
                        sample_trilinear_field(seed, seed.gamma_tilde[s], x, y, z);
                    G.A_tilde[s].ptr()[id] = sample_trilinear_field(seed, seed.A_tilde[s], x, y, z);
                }
            }
        }
    }

    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;
    proj_cfg.resync_contracted_gamma = true;
    tensorium_RG::bssn::project_bssn_state(G, proj_cfg);
    bssn::apply_halos_grid<bssn::BoundaryRadiative>(G);
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::compute_ricci_bssn(G, G.Ricci);
    tensorium_RG::bssn::assert_invariants(G, "init.bowen_york_interpolated");
#endif
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
    zero_z4c_fields(G);

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
    tensorium_RG::bssn::compute_tildeGamma_contracted(G);

#pragma omp parallel for collapse(3)
    for (size_t i = I0 + 3; i < I1 - 3; ++i)
        for (size_t j = J0 + 3; j < J1 - 3; ++j)
            for (size_t k = K0 + 3; k < K1 - 3; ++k) {

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

    auto constraints = make_constraint_scratch(G);
    tensorium_RG::bssn::compute_bssn_constraints(G, G.Ricci, constraints.H, constraints.M,
                                                 constraints.C, rMin, rMax, (double)xc, (double)yc,
                                                 (double)zc);

    tensorium_RG::bssn::print_constraint_norms(G, constraints.H, constraints.M, constraints.C, rMin,
                                              rMax, (double)xc, (double)yc, (double)zc);

    print_ricci_samples(G);
    tensorium_RG::bssn::assert_invariants(G, "init.kerr_schild_single");
    tensorium_RG::bssn::project_bssn_state(G);
}
} // namespace tensorium_RG::init
