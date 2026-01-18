#pragma once

#include <algorithm>
#include <cmath>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"

/**
 * @file BSSNEvolutionCommon.hpp
 * @brief Shared utilities (interior clamping helpers) for all RHS kernels.
 * @details Kernels call `clamped_lower/upper` to skip guard cells that halo/B.C. handlers may still
 * be updating.  Default padding of 4 ensures the ±2 (and ±3 KO6) stencils remain interior.
 */

namespace tensorium_RG::bssn {

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}
#endif

struct InteriorRegion {
    size_t i0 = 0, i1 = 0;
    size_t j0 = 0, j1 = 0;
    size_t k0 = 0, k1 = 0;

    [[nodiscard]] bool empty() const noexcept { return i0 >= i1 || j0 >= j1 || k0 >= k1; }
};

template <typename T>
inline InteriorRegion interior_bounds(const BSSNGridSoA<T> &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    InteriorRegion region;
    region.i0 = clamped_lower(I0, padding, I1);
    region.j0 = clamped_lower(J0, padding, J1);
    region.k0 = clamped_lower(K0, padding, K1);
    region.i1 = clamped_upper(I1, padding, I0);
    region.j1 = clamped_upper(J1, padding, J0);
    region.k1 = clamped_upper(K1, padding, K0);
    return region;
}

template <typename T, typename Fn>
inline void for_each_interior_index(const BSSNGridSoA<T> &grid, size_t padding, Fn &&fn) {
    const auto region = interior_bounds(grid, padding);
    if (region.empty())
        return;
    for (size_t i = region.i0; i < region.i1; ++i)
        for (size_t j = region.j0; j < region.j1; ++j)
            for (size_t k = region.k0; k < region.k1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                fn(i, j, k, idx);
            }
}

struct KOState {
    double scale = 0.0;
};

inline KOState &ko_state() {
    static KOState state{};
    return state;
}

template <typename T> inline double compute_cfl_scale(const BSSNGridSoA<T> &grid, T dt) {
    const double dt_val = double(dt);
    if (!(dt_val > 0.0))
        return 0.0;
    const double dx_min = std::min({double(grid.dx), double(grid.dy), double(grid.dz)});
    if (!(dx_min > 0.0))
        return 0.0;

    const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
    const T     *p_alpha = grid.alpha.ptr();
    const T     *p_bx = grid.beta[0].ptr();
    const T     *p_by = grid.beta[1].ptr();
    const T     *p_bz = grid.beta[2].ptr();

    double max_speed = 0.0;
#pragma omp parallel for reduction(max:max_speed)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(total); ++idx) {
        const double alpha = double(p_alpha[idx]);
        const double bx = std::abs(double(p_bx[idx]));
        const double by = std::abs(double(p_by[idx]));
        const double bz = std::abs(double(p_bz[idx]));
        const double local = std::max({bx, by, bz}) + alpha;
        if (std::isfinite(local))
            max_speed = std::max(max_speed, local);
    }

    const double cfl = max_speed * dt_val / dx_min;
    constexpr double cfl_ref = 0.5;
    if (!(cfl_ref > 0.0))
        return 0.0;
    const double ratio = cfl / cfl_ref;
    return std::clamp(ratio, 0.0, 1.0);
}

template <typename T> inline void update_ko_scale(const BSSNGridSoA<T> &grid, T dt) {
    ko_state().scale = compute_cfl_scale(grid, dt);
}

inline double current_ko_scale() { return ko_state().scale; }

template <typename T> inline T scaled_ko_sigma(T base_sigma) {
    return base_sigma * T(current_ko_scale());
}

inline void sym_index_to_pair(int s, int &row, int &col) {
    switch (s) {
    case 0:
        row = 0;
        col = 0;
        return;
    case 1:
        row = 0;
        col = 1;
        return;
    case 2:
        row = 0;
        col = 2;
        return;
    case 3:
        row = 1;
        col = 1;
        return;
    case 4:
        row = 1;
        col = 2;
        return;
    default:
        row = 2;
        col = 2;
        return;
    }
}

template <typename T> inline T Khat(T K, T Theta) {
    return K - T(2) * Theta;
}

template <typename T>
inline void compute_RicciZ4(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k, T Z4corr[6]) {
    using namespace tensorium_RG::fd;

    const double    inv_12dx = 1.0 / (60.0 * G.dx);
    const double    inv_12dy = 1.0 / (60.0 * G.dy);
    const double    inv_12dz = 1.0 / (60.0 * G.dz);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;

    const size_t idx = G.alpha.idx(i, j, k);

    const T chi = G.chi.ptr()[idx];
    const T inv_chi = T(1) / chi;

    T gamma_phys[3][3];
    T gamma_phys_inv[3][3];

    const T *p_gamma[6];
    const T *p_gamma_inv[6];
    for (int s = 0; s < 6; ++s) {
        p_gamma[s] = G.gamma_tilde[s].ptr() + idx;
        p_gamma_inv[s] = G.gamma_tilde_inv[s].ptr() + idx;
    }

    for (int s = 0; s < 6; ++s) {
        int row = 0;
        int col = 0;
        sym_index_to_pair(s, row, col);
        const T gt = *p_gamma[s];
        const T gt_inv = *p_gamma_inv[s];
        const T g_phys = gt * inv_chi;
        const T g_phys_inv = gt_inv * chi;
        gamma_phys[row][col] = g_phys;
        gamma_phys[col][row] = g_phys;
        gamma_phys_inv[row][col] = g_phys_inv;
        gamma_phys_inv[col][row] = g_phys_inv;
    }

    const T *p_chi = G.chi.ptr() + idx;
    const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                        Dz_ptr(p_chi, inv_12dz)};

    T d_g_phys[3][3][3];
    for (int s = 0; s < 6; ++s) {
        int row = 0;
        int col = 0;
        sym_index_to_pair(s, row, col);
        const T *p_g = p_gamma[s];
        const T  d_gt_x = Dx_ptr(p_g, sx, inv_12dx);
        const T  d_gt_y = Dy_ptr(p_g, sy, inv_12dy);
        const T  d_gt_z = Dz_ptr(p_g, inv_12dz);
        const T  g_val = gamma_phys[row][col];
        const T  val_x = (d_gt_x - g_val * d_chi[0]) * inv_chi;
        const T  val_y = (d_gt_y - g_val * d_chi[1]) * inv_chi;
        const T  val_z = (d_gt_z - g_val * d_chi[2]) * inv_chi;
        d_g_phys[0][row][col] = val_x;
        d_g_phys[0][col][row] = val_x;
        d_g_phys[1][row][col] = val_y;
        d_g_phys[1][col][row] = val_y;
        d_g_phys[2][row][col] = val_z;
        d_g_phys[2][col][row] = val_z;
    }

    T gamma_conn[3][3][3];
    for (int up = 0; up < 3; ++up)
        for (int lo1 = 0; lo1 < 3; ++lo1)
            for (int lo2 = 0; lo2 < 3; ++lo2) {
                T sum = T(0);
                for (int m = 0; m < 3; ++m)
                    sum += gamma_phys_inv[up][m] *
                           (d_g_phys[lo1][lo2][m] + d_g_phys[lo2][lo1][m] -
                            d_g_phys[m][lo1][lo2]);
                gamma_conn[up][lo1][lo2] = T(0.5) * sum;
            }

    const T *p_Z[3] = {G.Z[0].ptr() + idx, G.Z[1].ptr() + idx, G.Z[2].ptr() + idx};
    const T z_contra[3] = {*p_Z[0], *p_Z[1], *p_Z[2]};

    T Z_cov[3] = {T(0), T(0), T(0)};
    for (int row = 0; row < 3; ++row)
        for (int m = 0; m < 3; ++m)
            Z_cov[row] += gamma_phys[row][m] * z_contra[m];

    T dZ_contra[3][3];
    for (int comp = 0; comp < 3; ++comp) {
        const T *p_vec = p_Z[comp];
        dZ_contra[0][comp] = Dx_ptr(p_vec, sx, inv_12dx);
        dZ_contra[1][comp] = Dy_ptr(p_vec, sy, inv_12dy);
        dZ_contra[2][comp] = Dz_ptr(p_vec, inv_12dz);
    }

    T partial_Zcov[3][3];
    for (int dir = 0; dir < 3; ++dir)
        for (int b = 0; b < 3; ++b) {
            T sum = T(0);
            for (int m = 0; m < 3; ++m) {
                sum += d_g_phys[dir][b][m] * z_contra[m];
                sum += gamma_phys[b][m] * dZ_contra[dir][m];
            }
            partial_Zcov[dir][b] = sum;
        }

    T covZ[3][3];
    for (int dir = 0; dir < 3; ++dir)
        for (int b = 0; b < 3; ++b) {
            T val = partial_Zcov[dir][b];
            for (int m = 0; m < 3; ++m)
                val -= gamma_conn[m][dir][b] * Z_cov[m];
            covZ[dir][b] = val;
        }

    T divZ = T(0);
    for (int dir = 0; dir < 3; ++dir)
        for (int b = 0; b < 3; ++b)
            divZ += gamma_phys_inv[dir][b] * covZ[dir][b];

    const T two_thirds = T(2) / T(3);
    for (int s = 0; s < 6; ++s)
        Z4corr[s] = T(0);

    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            const int s = tensorium_RG::fd::sym6(a, b);
            const T   sym = covZ[a][b] + covZ[b][a];
            const T   val = sym - two_thirds * gamma_phys[a][b] * divZ;
            Z4corr[s] = val;
        }
}

} // namespace tensorium_RG::bssn
