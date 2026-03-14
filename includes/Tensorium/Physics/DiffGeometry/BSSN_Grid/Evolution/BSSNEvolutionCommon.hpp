#pragma once

#include <algorithm>
#include <cmath>

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "../Geometry/BSSNGamma.hpp"

/**
 * @file BSSNEvolutionCommon.hpp
 * @brief Shared utilities (interior clamping helpers) for all RHS kernels.
 * @details Kernels call `clamped_lower/upper` to skip guard cells that halo/B.C. handlers may still
 * be updating.  Default padding of 4 ensures the ±2 (and ±3 KO6) stencils remain interior.
 */

namespace tensorium_RG::bssn {

inline bool &rhs_kernel_team_mode_flag() {
    static thread_local bool enabled = false;
    return enabled;
}

inline void set_rhs_kernel_team_mode(bool enabled) { rhs_kernel_team_mode_flag() = enabled; }

inline bool rhs_kernel_team_mode_enabled() { return rhs_kernel_team_mode_flag(); }

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

struct InteriorPaddingOverride {
    bool active = false;
    size_t lower[3] = {0, 0, 0};
    size_t upper[3] = {0, 0, 0};
};

inline InteriorPaddingOverride &interior_padding_override() {
    static InteriorPaddingOverride override_state{};
    return override_state;
}

inline void set_interior_padding_override(size_t i_lower, size_t i_upper, size_t j_lower,
                                          size_t j_upper, size_t k_lower, size_t k_upper) {
    auto &state = interior_padding_override();
    state.active = true;
    state.lower[0] = i_lower;
    state.upper[0] = i_upper;
    state.lower[1] = j_lower;
    state.upper[1] = j_upper;
    state.lower[2] = k_lower;
    state.upper[2] = k_upper;
}

inline void clear_interior_padding_override() { interior_padding_override().active = false; }

struct ScopedInteriorPaddingOverride {
    InteriorPaddingOverride previous_{};

    explicit ScopedInteriorPaddingOverride(size_t i_lower, size_t i_upper, size_t j_lower,
                                           size_t j_upper, size_t k_lower, size_t k_upper) {
        previous_ = interior_padding_override();
        set_interior_padding_override(i_lower, i_upper, j_lower, j_upper, k_lower, k_upper);
    }

    ~ScopedInteriorPaddingOverride() { interior_padding_override() = previous_; }
};

template <typename T>
inline InteriorRegion interior_bounds(const BSSNGridSoA<T> &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const auto &override_state = interior_padding_override();
    const size_t i_lower = override_state.active ? override_state.lower[0] : padding;
    const size_t i_upper = override_state.active ? override_state.upper[0] : padding;
    const size_t j_lower = override_state.active ? override_state.lower[1] : padding;
    const size_t j_upper = override_state.active ? override_state.upper[1] : padding;
    const size_t k_lower = override_state.active ? override_state.lower[2] : padding;
    const size_t k_upper = override_state.active ? override_state.upper[2] : padding;

    InteriorRegion region;
    region.i0 = clamped_lower(I0, i_lower, I1);
    region.j0 = clamped_lower(J0, j_lower, J1);
    region.k0 = clamped_lower(K0, k_lower, K1);
    region.i1 = clamped_upper(I1, i_upper, I0);
    region.j1 = clamped_upper(J1, j_upper, J0);
    region.k1 = clamped_upper(K1, k_upper, K0);
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

template <typename T, typename Fn>
inline void for_each_interior_index_parallel(const BSSNGridSoA<T> &grid, size_t padding,
                                             Fn &&fn) {
    const auto region = interior_bounds(grid, padding);
    if (region.empty())
        return;
#pragma omp parallel for collapse(3)
    for (size_t i = region.i0; i < region.i1; ++i)
        for (size_t j = region.j0; j < region.j1; ++j)
            for (size_t k = region.k0; k < region.k1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                fn(i, j, k, idx);
            }
}

struct KOState {
    double scale = 1.0;
};

inline KOState &ko_state() {
    static KOState state{};
    return state;
}

// Keep KO dissipation coefficient constant (reference-parity behavior).
template <typename T> inline double compute_cfl_scale(const BSSNGridSoA<T> &, T) {
    return 1.0;
}

template <typename T> inline void update_ko_scale(const BSSNGridSoA<T> &, T) {
    ko_state().scale = 1.0;
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

template <typename T> inline T guard_chi_div(T chi, T chi_div_floor) {
    return (chi > chi_div_floor) ? chi : chi_div_floor;
}

template <typename T>
inline void recover_z_over_chi_from_gamma_ptr(const T *p_tildeGamma0, const T *p_tildeGamma1,
                                              const T *p_tildeGamma2, const T *p_ginv_xx,
                                              const T *p_ginv_xy, const T *p_ginv_xz,
                                              const T *p_ginv_yy, const T *p_ginv_yz,
                                              const T *p_ginv_zz, const ptrdiff_t sx,
                                              const ptrdiff_t sy, const double inv_12dx,
                                              const double inv_12dy, const double inv_12dz,
                                              T z_over_chi[3], T *gamma_metric_out = nullptr) {
    T gamma_metric_local[3] = {T(0), T(0), T(0)};
    T *gamma_metric = gamma_metric_out ? gamma_metric_out : gamma_metric_local;
    detail::metric_inverse_divergence_ptr(p_ginv_xx, p_ginv_xy, p_ginv_xz, p_ginv_yy, p_ginv_yz,
                                          p_ginv_zz, sx, sy, inv_12dx, inv_12dy, inv_12dz,
                                          gamma_metric);
    gamma_metric[0] = -gamma_metric[0];
    gamma_metric[1] = -gamma_metric[1];
    gamma_metric[2] = -gamma_metric[2];

    z_over_chi[0] = T(0.5) * ((*p_tildeGamma0) - gamma_metric[0]);
    z_over_chi[1] = T(0.5) * ((*p_tildeGamma1) - gamma_metric[1]);
    z_over_chi[2] = T(0.5) * ((*p_tildeGamma2) - gamma_metric[2]);
}

template <typename T>
inline void compute_RicciZ4_core(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                 T Z4corr[6], T chi_div_floor, const double inv_12dx,
                                 const double inv_12dy, const double inv_12dz,
                                 const ptrdiff_t sx, const ptrdiff_t sy) {
    using namespace tensorium_RG::fd;

    const size_t idx = G.alpha.idx(i, j, k);

    const T chi = G.chi.ptr()[idx];
    const T chi_guarded = guard_chi_div(chi, chi_div_floor);
    const T inv_chi = T(1) / chi_guarded;

    const T *p_chi = G.chi.ptr() + idx;
    const T d_chi[3] = {Dx_ptr(p_chi, sx, inv_12dx), Dy_ptr(p_chi, sy, inv_12dy),
                        Dz_ptr(p_chi, inv_12dz)};

    T g_tilde[3][3];
    g_tilde[0][0] = G.gamma_tilde[XX].ptr()[idx];
    g_tilde[0][1] = g_tilde[1][0] = G.gamma_tilde[XY].ptr()[idx];
    g_tilde[0][2] = g_tilde[2][0] = G.gamma_tilde[XZ].ptr()[idx];
    g_tilde[1][1] = G.gamma_tilde[YY].ptr()[idx];
    g_tilde[1][2] = g_tilde[2][1] = G.gamma_tilde[YZ].ptr()[idx];
    g_tilde[2][2] = G.gamma_tilde[ZZ].ptr()[idx];

    const T *p_ginv_xx = G.gamma_tilde_inv[XX].ptr() + idx;
    const T *p_ginv_xy = G.gamma_tilde_inv[XY].ptr() + idx;
    const T *p_ginv_xz = G.gamma_tilde_inv[XZ].ptr() + idx;
    const T *p_ginv_yy = G.gamma_tilde_inv[YY].ptr() + idx;
    const T *p_ginv_yz = G.gamma_tilde_inv[YZ].ptr() + idx;
    const T *p_ginv_zz = G.gamma_tilde_inv[ZZ].ptr() + idx;

    // Keep the Ricci Z correction self-consistent with the evolved contracted Gamma,
    // matching the reference CCZ4/BSSN formulations that recover Z^i/chi from
    // \hat{Gamma}^i - Gamma^i(metric).
    T z_over_chi[3] = {T(0), T(0), T(0)};
    recover_z_over_chi_from_gamma_ptr(G.tildeGamma[0].ptr() + idx, G.tildeGamma[1].ptr() + idx,
                                      G.tildeGamma[2].ptr() + idx, p_ginv_xx, p_ginv_xy,
                                      p_ginv_xz, p_ginv_yy, p_ginv_yz, p_ginv_zz, sx, sy,
                                      inv_12dx, inv_12dy, inv_12dz, z_over_chi);

    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            T z_terms = T(0);
            for (int m = 0; m < 3; ++m) {
                z_terms += z_over_chi[m] * (g_tilde[a][m] * d_chi[b] + g_tilde[b][m] * d_chi[a] -
                                            g_tilde[a][b] * d_chi[m]);
            }
            Z4corr[sym6(a, b)] = z_terms * inv_chi;
        }
}

template <typename T>
inline void compute_RicciZ4(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k, T Z4corr[6],
                            T chi_div_floor = T(-1000.0)) {
    const double    inv_12dx = 1.0 / (60.0 * G.dx);
    const double    inv_12dy = 1.0 / (60.0 * G.dy);
    const double    inv_12dz = 1.0 / (60.0 * G.dz);
    const ptrdiff_t sx = G.alpha.st.sx;
    const ptrdiff_t sy = G.alpha.st.sy;
    compute_RicciZ4_core(G, i, j, k, Z4corr, chi_div_floor, inv_12dx, inv_12dy, inv_12dz, sx, sy);
}

} // namespace tensorium_RG::bssn
