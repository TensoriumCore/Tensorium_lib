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
    size_t boundary_width = 0;
    double boundary_floor = 0.0;
    double boundary_boost = 0.0;
    double edge_corner_boost = 0.0;
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

inline void configure_ko_boundary_taper(size_t width, double floor = 0.0, double boost = 0.0,
                                        double edge_corner_boost = 0.0) {
    auto &state = ko_state();
    state.boundary_width = width;
    state.boundary_floor = std::clamp(floor, 0.0, 1.0);
    state.boundary_boost = std::max(boost, 0.0);
    state.edge_corner_boost = std::max(edge_corner_boost, 0.0);
}

inline size_t current_ko_boundary_width() { return ko_state().boundary_width; }

inline double current_ko_boundary_floor() { return ko_state().boundary_floor; }

inline double current_ko_boundary_boost() { return ko_state().boundary_boost; }

inline double current_ko_edge_corner_boost() { return ko_state().edge_corner_boost; }

template <typename T> inline T scaled_ko_sigma(T base_sigma) {
    return base_sigma * T(current_ko_scale());
}

template <typename T>
inline T ko_boundary_taper(const BSSNGridSoA<T> &grid, size_t i, size_t j, size_t k) {
    const size_t width = current_ko_boundary_width();
    if (width == 0)
        return T(1);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    if (I0 >= I1 || J0 >= J1 || K0 >= K1)
        return T(1);

    const size_t di = std::min(i - I0, (I1 - 1) - i);
    const size_t dj = std::min(j - J0, (J1 - 1) - j);
    const size_t dk = std::min(k - K0, (K1 - 1) - k);
    const size_t dist = std::min({di, dj, dk});
    if (dist >= width)
        return T(1);

    const auto axis_wall_weight = [width](size_t d) -> double {
        if (d >= width)
            return 0.0;
        const double x = static_cast<double>(d) / static_cast<double>(width);
        const double smooth = x * x * (3.0 - 2.0 * x);
        return 1.0 - smooth;
    };

    const double wx = axis_wall_weight(di);
    const double wy = axis_wall_weight(dj);
    const double wz = axis_wall_weight(dk);
    const double wall = std::max({wx, wy, wz});
    const double smooth = 1.0 - wall;
    const double floor = current_ko_boundary_floor();
    const double base = floor + (1.0 - floor) * smooth;
    std::array<double, 3> weights{wx, wy, wz};
    std::sort(weights.begin(), weights.end(), std::greater<double>());
    // Edge/corner reinforcement should vanish on a single face. Use the second strongest wall
    // proximity as the edge trigger and the third as the extra corner weight.
    const double edge_weight = weights[1];
    const double corner_weight = weights[2];
    const double boost = current_ko_boundary_boost() * edge_weight *
                         (1.0 + current_ko_edge_corner_boost() * corner_weight);
    return T(base + boost);
}

template <typename T>
inline T local_ko_scale(const BSSNGridSoA<T> &grid, T scaled_sigma, size_t i, size_t j, size_t k) {
    return (scaled_sigma / grid.dx) * ko_boundary_taper(grid, i, j, k);
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

#if defined(__clang__) || defined(__GNUC__)
#define TENSORIUM_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define TENSORIUM_ALWAYS_INLINE inline
#endif

template <int Order, typename T>
TENSORIUM_ALWAYS_INLINE void recover_z_over_chi_components_from_gamma_ptr_order(
    const T *__restrict p_tildeGamma0, const T *__restrict p_tildeGamma1,
    const T *__restrict p_tildeGamma2, const T *__restrict p_ginv_xx,
    const T *__restrict p_ginv_xy, const T *__restrict p_ginv_xz,
    const T *__restrict p_ginv_yy, const T *__restrict p_ginv_yz,
    const T *__restrict p_ginv_zz, const ptrdiff_t sx, const ptrdiff_t sy,
    const double inv_12dx, const double inv_12dy, const double inv_12dz, T &z0, T &z1, T &z2,
    T *gamma_metric_out = nullptr) {
    T gamma_metric_local[3] = {T(0), T(0), T(0)};
    T *gamma_metric = gamma_metric_out ? gamma_metric_out : gamma_metric_local;
    detail::metric_inverse_divergence_components_ptr_order<Order>(
        p_ginv_xx, p_ginv_xy, p_ginv_xz, p_ginv_yy, p_ginv_yz, p_ginv_zz, sx, sy, inv_12dx,
        inv_12dy, inv_12dz, gamma_metric[0], gamma_metric[1], gamma_metric[2]);
    gamma_metric[0] = -gamma_metric[0];
    gamma_metric[1] = -gamma_metric[1];
    gamma_metric[2] = -gamma_metric[2];

    z0 = T(0.5) * ((*p_tildeGamma0) - gamma_metric[0]);
    z1 = T(0.5) * ((*p_tildeGamma1) - gamma_metric[1]);
    z2 = T(0.5) * ((*p_tildeGamma2) - gamma_metric[2]);
}

template <int Order, typename T>
TENSORIUM_ALWAYS_INLINE void recover_z_over_chi_from_gamma_ptr_order(
    const T *__restrict p_tildeGamma0, const T *__restrict p_tildeGamma1,
    const T *__restrict p_tildeGamma2, const T *__restrict p_ginv_xx,
    const T *__restrict p_ginv_xy, const T *__restrict p_ginv_xz,
    const T *__restrict p_ginv_yy, const T *__restrict p_ginv_yz,
    const T *__restrict p_ginv_zz, const ptrdiff_t sx, const ptrdiff_t sy,
    const double inv_12dx, const double inv_12dy, const double inv_12dz, T z_over_chi[3],
    T *gamma_metric_out = nullptr) {
    recover_z_over_chi_components_from_gamma_ptr_order<Order>(
        p_tildeGamma0, p_tildeGamma1, p_tildeGamma2, p_ginv_xx, p_ginv_xy, p_ginv_xz, p_ginv_yy,
        p_ginv_yz, p_ginv_zz, sx, sy, inv_12dx, inv_12dy, inv_12dz, z_over_chi[0], z_over_chi[1],
        z_over_chi[2], gamma_metric_out);
}

#undef TENSORIUM_ALWAYS_INLINE

template <typename T>
inline void recover_z_over_chi_from_gamma_ptr(const T *__restrict p_tildeGamma0,
                                              const T *__restrict p_tildeGamma1,
                                              const T *__restrict p_tildeGamma2,
                                              const T *__restrict p_ginv_xx,
                                              const T *__restrict p_ginv_xy,
                                              const T *__restrict p_ginv_xz,
                                              const T *__restrict p_ginv_yy,
                                              const T *__restrict p_ginv_yz,
                                              const T *__restrict p_ginv_zz, const ptrdiff_t sx,
                                              const ptrdiff_t sy, const double inv_12dx,
                                              const double inv_12dy, const double inv_12dz,
                                              T z_over_chi[3], T *gamma_metric_out = nullptr) {
    if (tensorium_RG::fd::max_spatial_derivative_order() == 4) {
        recover_z_over_chi_from_gamma_ptr_order<4>(
            p_tildeGamma0, p_tildeGamma1, p_tildeGamma2, p_ginv_xx, p_ginv_xy, p_ginv_xz,
            p_ginv_yy, p_ginv_yz, p_ginv_zz, sx, sy, inv_12dx, inv_12dy, inv_12dz, z_over_chi,
            gamma_metric_out);
    } else {
        recover_z_over_chi_from_gamma_ptr_order<6>(
            p_tildeGamma0, p_tildeGamma1, p_tildeGamma2, p_ginv_xx, p_ginv_xy, p_ginv_xz,
            p_ginv_yy, p_ginv_yz, p_ginv_zz, sx, sy, inv_12dx, inv_12dy, inv_12dz, z_over_chi,
            gamma_metric_out);
    }
}

template <int Order, typename T>
inline void compute_RicciZ4_core(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                 T Z4corr[6], T chi_div_floor, const double inv_12dx,
                                 const double inv_12dy, const double inv_12dz,
                                 const ptrdiff_t sx, const ptrdiff_t sy) {
    const size_t idx = G.alpha.idx(i, j, k);

    const T chi = G.chi.ptr()[idx];
    const T chi_guarded = guard_chi_div(chi, chi_div_floor);
    const T inv_chi = T(1) / chi_guarded;

    const T *p_chi = G.chi.ptr() + idx;
    const T d_chi[3] = {tensorium_RG::fd::Dx_ptr_order<Order>(p_chi, sx, inv_12dx),
                        tensorium_RG::fd::Dy_ptr_order<Order>(p_chi, sy, inv_12dy),
                        tensorium_RG::fd::Dz_ptr_order<Order>(p_chi, inv_12dz)};

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
    recover_z_over_chi_from_gamma_ptr_order<Order>(
        G.tildeGamma[0].ptr() + idx, G.tildeGamma[1].ptr() + idx, G.tildeGamma[2].ptr() + idx,
        p_ginv_xx, p_ginv_xy, p_ginv_xz, p_ginv_yy, p_ginv_yz, p_ginv_zz, sx, sy, inv_12dx,
        inv_12dy, inv_12dz, z_over_chi);

    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            T z_terms = T(0);
            for (int m = 0; m < 3; ++m) {
                z_terms += z_over_chi[m] * (g_tilde[a][m] * d_chi[b] + g_tilde[b][m] * d_chi[a] -
                                            g_tilde[a][b] * d_chi[m]);
            }
            Z4corr[tensorium_RG::fd::sym6(a, b)] = z_terms * inv_chi;
        }
}

template <typename T>
inline void compute_RicciZ4_core(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                 T Z4corr[6], T chi_div_floor, const double inv_12dx,
                                 const double inv_12dy, const double inv_12dz,
                                 const ptrdiff_t sx, const ptrdiff_t sy) {
    if (tensorium_RG::fd::max_spatial_derivative_order() == 4) {
        compute_RicciZ4_core<4>(G, i, j, k, Z4corr, chi_div_floor, inv_12dx, inv_12dy, inv_12dz,
                                sx, sy);
    } else {
        compute_RicciZ4_core<6>(G, i, j, k, Z4corr, chi_div_floor, inv_12dx, inv_12dy, inv_12dz,
                                sx, sy);
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
