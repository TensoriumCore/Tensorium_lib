#pragma once
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNCHristoffelTilde.hpp"
#include "../Geometry/BSSNInvariants.hpp"
#include "../Geometry/BSSNProjection.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

/**
 * @file BSSNConstrainSolver.hpp
 * @brief Bowen–York extrinsic curvature assembly and Lichnerowicz equation solver.
 * @details
 * Binary puncture initial data require solving the Hamiltonian constraint in the conformally flat
 * ansatz \f$\gamma_{ij}=\psi^4\delta_{ij}\f$ with analytic \f$\tilde{A}_{ij}\f$ constructed from prescribed
 * linear momenta and spins (Bowen–York).  This header provides routines to (1) add the momentum and
 * spin pieces
 * \f[
 * \tilde{A}_{ij}^P = \frac{3}{2 r^2}(n_i P_j + n_j P_i - (\delta_{ij}-n_i n_j) n^k P_k),\qquad
 * \tilde{A}_{ij}^S = \frac{3}{r^3}\epsilon_{k \ell(i} S^k n^{\ell} n_{j)}
 * \f]
 * and (2) solve \f$\nabla^2 u + \frac{1}{8}\psi^{-7}\tilde{A}_{ij}\tilde{A}^{ij} = 0\f$ for the correction \f$u\f$ to
 * the conformal factor \f$\psi = 1 + \sum_a m_a/(2 r_a) + u\f$ via red-black successive over-relaxation.
 */

namespace tensorium_RG::init {

/**
 * @brief Add the trace-free momentum component \f$\tilde{A}_{ij}^P\f$ to \f$\tilde{A}_{ij}\f$.
 */
template <typename T>
static inline void by_add_momentum_TF(T A[3][3], const T n[3], const T P[3], T r) {
    const T one = T(1);
    const T three = T(3);
    const T two = T(2);

    const T nP = n[0] * P[0] + n[1] * P[1] + n[2] * P[2];

    // Ã_ij^P = (3/(2 r^2)) [ n_i P_j + n_j P_i - (δ_ij - n_i n_j) (n·P) ]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const T dij = (i == j ? one : T(0));
            const T term = n[i] * P[j] + n[j] * P[i] - (dij - n[i] * n[j]) * nP;
            A[i][j] += (three / (two * r * r)) * term;
        }
    }
}

/**
 * @brief Add the spin component \f$\tilde{A}_{ij}^S\f$ derived from \f$S^i\f$.
 */
template <typename T>
static inline void by_add_spin_TF(T A[3][3], const T n[3], const T S[3], T r) {
    const T one = T(1);
    const T three = T(3);
    const T two = T(2);

    // (S x n)_i
    const T cx = S[1] * n[2] - S[2] * n[1];
    const T cy = S[2] * n[0] - S[0] * n[2];
    const T cz = S[0] * n[1] - S[1] * n[0];
    const T c[3] = {cx, cy, cz};

    // Ã_ij^S = (3/r^3) [ ε_{k l (i} S^k n^{l} n_{j)} ] = (3/r^3) sym( (S×n)_i n_j )
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const T term = c[i] * n[j] + c[j] * n[i];
            A[i][j] += (three / (two * r * r * r)) * term;
        }
    }
}

/**
 * @brief Populate \f$\tilde{A}_{ij}\f$ with the superposition of two Bowen–York punctures.
 * @param x1,y1,z1,x2,y2,z2 Centers of punctures 1 and 2.
 * @param P1,S1 Linear momentum and spin of puncture 1; analogous for puncture 2.
 * @param r_floor Minimum radius enforced when normalizing \f$n_i\f$.
 */
template <typename T>
inline void fill_Atilde_bowen_york_binary(BSSNGridSoA<T> &G, T x1, T y1, T z1, const T P1[3],
                                          const T S1[3], T x2, T y2, T z2, const T P2[3],
                                          const T S2[3], T r_floor = T(1e-6)) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = G.alpha.idx(i, j, k);

                T x, y, z;
                G.coords(i, j, k, x, y, z);

                T A[3][3] = {{T(0), T(0), T(0)}, {T(0), T(0), T(0)}, {T(0), T(0), T(0)}};

                // puncture 1
                {
                    const T dx = x - x1, dy = y - y1, dz = z - z1;
                    T       r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r < r_floor)
                        r = r_floor;
                    const T invr = T(1) / r;
                    const T n[3] = {dx * invr, dy * invr, dz * invr};

                    by_add_momentum_TF(A, n, P1, r);
                    by_add_spin_TF(A, n, S1, r);
                }

                // puncture 2
                {
                    const T dx = x - x2, dy = y - y2, dz = z - z2;
                    T       r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r < r_floor)
                        r = r_floor;
                    const T invr = T(1) / r;
                    const T n[3] = {dx * invr, dy * invr, dz * invr};

                    by_add_momentum_TF(A, n, P2, r);
                    by_add_spin_TF(A, n, S2, r);
                }

                G.A_tilde[XX].ptr()[id] = A[0][0];
                G.A_tilde[XY].ptr()[id] = A[0][1];
                G.A_tilde[XZ].ptr()[id] = A[0][2];
                G.A_tilde[YY].ptr()[id] = A[1][1];
                G.A_tilde[YZ].ptr()[id] = A[1][2];
                G.A_tilde[ZZ].ptr()[id] = A[2][2];
            }
}

/**
 * @brief Compute \f$\tilde{A}_{ij}\tilde{A}^{ij}\f$ assuming \f$\tilde{\gamma}_{ij}=\delta_{ij}\f$.
 */
template <typename T> static inline T Atilde_sq_flat_from_soa(const BSSNGridSoA<T> &G, size_t id) {
    const T Axx = G.A_tilde[XX].ptr()[id];
    const T Axy = G.A_tilde[XY].ptr()[id];
    const T Axz = G.A_tilde[XZ].ptr()[id];
    const T Ayy = G.A_tilde[YY].ptr()[id];
    const T Ayz = G.A_tilde[YZ].ptr()[id];
    const T Azz = G.A_tilde[ZZ].ptr()[id];
    return Axx * Axx + Ayy * Ayy + Azz * Azz + T(2) * (Axy * Axy + Axz * Axz + Ayz * Ayz);
}

/**
 * @brief Solve the Lichnerowicz equation \f$\Delta u = -\frac{1}{8}\psi^{-7}\tilde{A}_{ij}\tilde{A}^{ij}\f$ via red-black SOR.
 * @param m1,m2,x1,... Coordinates and bare masses entering the Brill–Lindquist conformal factor.
 * @param maxIter,tol,omega Solver controls (iterations, residual tolerance, relaxation factor).
 * @details
 * The discrete Laplacian uses standard 2nd-order stencils to balance efficiency with robustness.
 * After convergence the routine updates \f$\chi=\psi^{-4}\f$, \f$\alpha=\psi^{-2}\f$, converts the
 * Bowen--York conformal source \f$\bar{A}_{ij}\f$ to the evolved BSSN variable
 * \f$\tilde{A}_{ij}=\psi^{-6}\bar{A}_{ij}=\chi^{3/2}\bar{A}_{ij}\f$, and reprojects the grid.
 */
template <typename T>
inline void solve_lichnerowicz_u_SOR(BSSNGridSoA<T> &G, T m1, T x1, T y1, T z1, T m2, T x2, T y2,
                                     T z2, T r_floor = T(1e-6), int maxIter = 4000,
                                     T tol = T(1e-10), T omega = T(1.6)) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t nxTot = (I1 - I0);
    const size_t nyTot = (J1 - J0);
    const size_t nzTot = (K1 - K0);
    const size_t N = nxTot * nyTot * nzTot;

    std::vector<T> u(N, T(0));
    std::vector<T> u_new(N, T(0));

    auto lid = [&](size_t i, size_t j, size_t k) -> size_t {
        return (i - I0) * nyTot * nzTot + (j - J0) * nzTot + (k - K0);
    };

    const T idx2 = T(1) / (G.dx * G.dx);
    const T idy2 = T(1) / (G.dy * G.dy);
    const T idz2 = T(1) / (G.dz * G.dz);
    const T denom = T(2) * (idx2 + idy2 + idz2);

    for (int it = 0; it < maxIter; ++it) {
        T maxRes = T(0);

        for (int color = 0; color < 2; ++color) {
#pragma omp parallel for collapse(3) reduction(max : maxRes)
            for (size_t i = I0 + 1; i < I1 - 1; ++i)
                for (size_t j = J0 + 1; j < J1 - 1; ++j)
                    for (size_t k = K0 + 1; k < K1 - 1; ++k) {
                        if (int((i + j + k) & 1) != color)
                            continue;

                        const size_t p = lid(i, j, k);

                        T x, y, z;
                        G.coords(i, j, k, x, y, z);

                        T r1 = std::sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) +
                                         (z - z1) * (z - z1));
                        T r2 = std::sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2) +
                                         (z - z2) * (z - z2));
                        if (r1 < r_floor)
                            r1 = r_floor;
                        if (r2 < r_floor)
                            r2 = r_floor;

                        const T psi0 = T(1) + (m1 * T(0.5)) / r1 + (m2 * T(0.5)) / r2;
                        const T psi = psi0 + u[p];

                        const size_t id = G.alpha.idx(i, j, k);
                        const T      A2 = Atilde_sq_flat_from_soa(G, id);

                        const T rhs = -(T(1) / T(8)) * std::pow(psi, T(-7)) * A2;

                        const T u_xm = u[lid(i - 1, j, k)];
                        const T u_xp = u[lid(i + 1, j, k)];
                        const T u_ym = u[lid(i, j - 1, k)];
                        const T u_yp = u[lid(i, j + 1, k)];
                        const T u_zm = u[lid(i, j, k - 1)];
                        const T u_zp = u[lid(i, j, k + 1)];

                        const T num = idx2 * (u_xm + u_xp) + idy2 * (u_ym + u_yp) +
                                      idz2 * (u_zm + u_zp) - rhs;
                        const T uGS = num / denom;

                        const T du = uGS - u[p];
                        u[p] = u[p] + omega * du;

                        // résidu discret : (Δu - rhs)
                        const T lap = idx2 * (u_xm - T(2) * u[p] + u_xp) +
                                      idy2 * (u_ym - T(2) * u[p] + u_yp) +
                                      idz2 * (u_zm - T(2) * u[p] + u_zp);
                        const T res = lap - rhs;
                        maxRes = std::max(maxRes, std::abs(res));
                    }
        }

        if (it % 50 == 0) {
            printf("[Lichnerowicz] it=%d maxRes=%.3e\n", it, double(maxRes));
        }
        if (maxRes < tol) {
            printf("[Lichnerowicz] converged it=%d maxRes=%.3e\n", it, double(maxRes));
            break;
        }
        if (it == maxIter - 1) {
            printf("[Lichnerowicz] reached maxIter=%d maxRes=%.3e\n", maxIter, double(maxRes));
        }
    }

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t p = lid(i, j, k);
                const size_t id = G.alpha.idx(i, j, k);

                T x, y, z;
                G.coords(i, j, k, x, y, z);

                T r1 = std::sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1));
                T r2 = std::sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2));
                if (r1 < r_floor)
                    r1 = r_floor;
                if (r2 < r_floor)
                    r2 = r_floor;

                const T psi0 = T(1) + (m1 * T(0.5)) / r1 + (m2 * T(0.5)) / r2;
                const T psi = psi0 + u[p];

                const T chi = T(1) / (psi * psi * psi * psi);
                const T alpha = T(1) / (psi * psi);
                const T chi_sqrt = std::sqrt(std::max(chi, T(0)));
                const T A_scale = chi * chi_sqrt; // psi^{-6}

                G.chi.ptr()[id] = chi;
                G.alpha.ptr()[id] = alpha;
                for (int s = 0; s < 6; ++s)
                    G.A_tilde[s].ptr()[id] *= A_scale;
            }

    tensorium_RG::bssn::compute_tildeGamma_contracted(G);
    tensorium_RG::bssn::assert_invariants(G, "lichnerowicz");
    tensorium_RG::bssn::project_bssn_state(G);
}

} // namespace tensorium_RG::init
