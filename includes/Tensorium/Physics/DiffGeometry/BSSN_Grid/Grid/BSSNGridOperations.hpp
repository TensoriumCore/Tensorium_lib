#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

/**
 * @file BSSNGridOperations.hpp
 * @brief Grid-wide helpers such as halo application shared by evolution and initialization code.
 * @details
 * The Runge–Kutta driver templates over a `Boundary` functor that satisfies the `apply(Field3D,
 * GridDims)` interface.  This indirection centralizes halo synchronization (or boundary clamping)
 * so the physics kernels only see fully-populated guard zones before computing derivatives.  Every
 * field contained in `BSSNGridSoA` is forwarded to the boundary functor in a consistent order to
 * keep caches coherent and to support boundary conditions that need coupled updates (e.g., for
 * shift/torsion).
 */

namespace tensorium_RG::bssn {

enum class BoundaryField {
    Alpha,
    Chi,
    K,
    Theta,
    Beta,
    B,
    TildeGamma,
    GammaTilde,
    GammaTildeInverse,
    ATilde,
    Z
};

namespace detail {

struct SpongeProfileConfig {
    bool enabled = false;
    size_t width = 0;
    double strength = 0.0;
    double exponent = 2.0;
};

template <typename T>
inline bool nearest_sponge_layer(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                 const SpongeProfileConfig &cfg,
                                 const bool (&active_face)[3][2],
                                 const bool (&reflective_face)[3][2], size_t &layer_out) {
    if (!cfg.enabled || cfg.width == 0 || cfg.strength <= 0.0)
        return false;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    if (I0 >= I1 || J0 >= J1 || K0 >= K1)
        return false;

    const size_t width_x = std::min(cfg.width, I1 - I0);
    const size_t width_y = std::min(cfg.width, J1 - J0);
    const size_t width_z = std::min(cfg.width, K1 - K0);

    size_t best = cfg.width;
    const auto consider = [&](bool enabled, size_t dist, size_t width) {
        if (!enabled || dist >= width)
            return;
        best = std::min(best, dist);
    };

    consider(active_face[0][0] && !reflective_face[0][0], i - I0, width_x);
    consider(active_face[0][1] && !reflective_face[0][1], (I1 - 1) - i, width_x);
    consider(active_face[1][0] && !reflective_face[1][0], j - J0, width_y);
    consider(active_face[1][1] && !reflective_face[1][1], (J1 - 1) - j, width_y);
    consider(active_face[2][0] && !reflective_face[2][0], k - K0, width_z);
    consider(active_face[2][1] && !reflective_face[2][1], (K1 - 1) - k, width_z);

    if (best >= cfg.width)
        return false;
    layer_out = best;
    return true;
}

inline double sponge_profile_value(size_t layer, const SpongeProfileConfig &cfg) {
    if (layer >= cfg.width || cfg.width == 0 || cfg.strength <= 0.0)
        return 0.0;
    const double remaining =
        static_cast<double>(cfg.width - layer) / static_cast<double>(cfg.width);
    const double exponent = std::max(cfg.exponent, 1.0);
    return cfg.strength * std::pow(remaining, exponent);
}

inline double sponge_blend_lambda(size_t layer, const SpongeProfileConfig &cfg, double dt_scale) {
    return std::clamp(sponge_profile_value(layer, cfg) * dt_scale, 0.0, 1.0);
}

template <typename Boundary, typename T>
inline void apply_physical(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                           int component) {
    Boundary::apply_physical(field, G, which, component);
}

template <typename Boundary, typename T>
inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                       int component) {
    Boundary::apply_halo(field, G, which, component);
}

template <typename Boundary, typename T> inline void apply_batch_physical(BSSNGridSoA<T> &G) {
    apply_physical<Boundary>(G.alpha, G, BoundaryField::Alpha, 0);
    apply_physical<Boundary>(G.chi, G, BoundaryField::Chi, 0);
    apply_physical<Boundary>(G.K, G, BoundaryField::K, 0);
    apply_physical<Boundary>(G.Theta, G, BoundaryField::Theta, 0);

    for (int c = 0; c < 3; ++c) {
        apply_physical<Boundary>(G.beta[c], G, BoundaryField::Beta, c);
        apply_physical<Boundary>(G.B[c], G, BoundaryField::B, c);
        apply_physical<Boundary>(G.tildeGamma[c], G, BoundaryField::TildeGamma, c);
        apply_physical<Boundary>(G.Z[c], G, BoundaryField::Z, c);
    }

    for (int s = 0; s < 6; ++s) {
        apply_physical<Boundary>(G.gamma_tilde[s], G, BoundaryField::GammaTilde, s);
        apply_physical<Boundary>(G.gamma_tilde_inv[s], G, BoundaryField::GammaTildeInverse, s);
        apply_physical<Boundary>(G.A_tilde[s], G, BoundaryField::ATilde, s);
    }

}

template <typename Boundary, typename T> inline void apply_batch_halo(BSSNGridSoA<T> &G) {
    apply_halo<Boundary>(G.alpha, G, BoundaryField::Alpha, 0);
    apply_halo<Boundary>(G.chi, G, BoundaryField::Chi, 0);
    // Keep Theta ahead of K so radiative closures can reconstruct the fast Khat mode
    // with already-refreshed Theta ghosts.
    apply_halo<Boundary>(G.Theta, G, BoundaryField::Theta, 0);
    apply_halo<Boundary>(G.K, G, BoundaryField::K, 0);

    for (int c = 0; c < 3; ++c) {
        apply_halo<Boundary>(G.beta[c], G, BoundaryField::Beta, c);
        apply_halo<Boundary>(G.B[c], G, BoundaryField::B, c);
        apply_halo<Boundary>(G.tildeGamma[c], G, BoundaryField::TildeGamma, c);
        apply_halo<Boundary>(G.Z[c], G, BoundaryField::Z, c);
    }

    for (int s = 0; s < 6; ++s) {
        apply_halo<Boundary>(G.gamma_tilde[s], G, BoundaryField::GammaTilde, s);
        apply_halo<Boundary>(G.gamma_tilde_inv[s], G, BoundaryField::GammaTildeInverse, s);
        apply_halo<Boundary>(G.A_tilde[s], G, BoundaryField::ATilde, s);
    }

}

inline double minkowski_target(BoundaryField which, int component) {
    switch (which) {
    case BoundaryField::Alpha:
    case BoundaryField::Chi:
        return 1.0;
    case BoundaryField::K:
    case BoundaryField::Beta:
    case BoundaryField::B:
    case BoundaryField::TildeGamma:
    case BoundaryField::ATilde:
    case BoundaryField::Theta:
    case BoundaryField::Z:
        return 0.0;
    case BoundaryField::GammaTilde:
        if (component == tensorium_RG::XX || component == tensorium_RG::YY ||
            component == tensorium_RG::ZZ)
            return 1.0;
        return 0.0;
    default:
        return 0.0;
    }
}

template <typename T>
inline void apply_sponge_layer(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                               int component, const SpongeProfileConfig &cfg, double dt_scale,
                               const bool (&active_face)[3][2], const bool (&reflective_face)[3][2]) {
    if (!cfg.enabled || cfg.width == 0 || cfg.strength <= 0.0 || dt_scale <= 0.0)
        return;
    if (which == BoundaryField::GammaTildeInverse)
        return;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    if (I0 >= I1 || J0 >= J1 || K0 >= K1)
        return;

    const double asymptotic = minkowski_target(which, component);
    T           *ptr = field.ptr();

    auto blend = [&](size_t idx, size_t layer) {
        const double lambda = sponge_blend_lambda(layer, cfg, dt_scale);
        if (lambda <= 0.0)
            return;
        const double current = static_cast<double>(ptr[idx]);
        ptr[idx] = static_cast<T>(current + lambda * (asymptotic - current));
    };
#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                size_t layer = 0;
                if (!nearest_sponge_layer(G, i, j, k, cfg, active_face, reflective_face, layer))
                    continue;
                blend(field.idx(i, j, k), layer);
            }
}

} // namespace detail

struct BoundaryClamp {
    template <typename T>
    static inline void apply_physical(Field3D<T> &, const BSSNGridSoA<T> &, BoundaryField, int) {}

    template <typename T>
    static inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField, int) {
        halo_linear_extrapolate_full(field, G.dims);
    }
};

struct BoundarySponge {
    inline static detail::SpongeProfileConfig sponge_config{true, 8, 1.0, 2.0};

    static inline void set_sponge(bool enabled, size_t width, double strength, double exponent = 2.0) {
        sponge_config.enabled = enabled;
        sponge_config.width = width;
        sponge_config.strength = std::max(strength, 0.0);
        sponge_config.exponent = std::max(exponent, 1.0);
    }

    template <typename T>
    static inline void apply_physical(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                                      int component) {
        constexpr bool active[3][2] = {{true, true}, {true, true}, {true, true}};
        constexpr bool reflective[3][2] = {{false, false}, {false, false}, {false, false}};
        detail::apply_sponge_layer(field, G, which, component, sponge_config, 1.0, active,
                                   reflective);
    }

    template <typename T>
    static inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField, int) {
        halo_linear_extrapolate_full(field, G.dims);
    }
};

struct BoundaryRadiative {
    inline static double characteristic_speed = 1.0;
    inline static double characteristic_dt = 0.0;
    inline static double gauge_characteristic_speed = 1.0;
    inline static double z4c_characteristic_speed = 1.0;
    inline static double khat_characteristic_speed = 1.4142135623730951;
    inline static size_t rhs_collar_width = 4;
    // Per-face mask for RHS Sommerfeld-style boundary updates:
    // [axis][side], side=0 -> inner, side=1 -> outer.
    inline static bool rhs_sommerfeld_face[3][2] = {{true, true}, {true, true}, {true, true}};
    // Per-face reflective mask (parity BC), same indexing convention as rhs_sommerfeld_face.
    inline static bool reflective_face[3][2] = {{false, false}, {false, false}, {false, false}};
    // Per-face activity mask. In MPI, internal interfaces must be skipped entirely after halo
    // exchange so exchanged ghosts are not overwritten by fallback outflow copies.
    inline static bool active_face[3][2] = {{true, true}, {true, true}, {true, true}};
    inline static detail::SpongeProfileConfig sponge_config{};

    static inline void set_characteristic(double speed, double dt) {
        characteristic_speed = std::max(speed, 1.0e-6);
        characteristic_dt = std::max(dt, 0.0);
    }

    static inline void set_field_characteristic_speeds(double gauge_speed, double z4c_speed = 1.0,
                                                       double khat_speed = 1.4142135623730951) {
        gauge_characteristic_speed = std::max(gauge_speed, 1.0e-6);
        z4c_characteristic_speed = std::max(z4c_speed, 1.0e-6);
        khat_characteristic_speed = std::max(khat_speed, 1.0e-6);
    }

    static inline void set_rhs_collar_width(size_t width) {
        rhs_collar_width = std::max<size_t>(size_t(1), width);
    }

    [[nodiscard]] static inline size_t rhs_collar() { return rhs_collar_width; }

    [[nodiscard]] static inline double characteristic_speed_for(BoundaryField which) {
        switch (which) {
        case BoundaryField::Alpha:
        case BoundaryField::Chi:
        case BoundaryField::Beta:
        case BoundaryField::B:
        case BoundaryField::GammaTilde:
            return gauge_characteristic_speed;
        case BoundaryField::K:
            return khat_characteristic_speed;
        case BoundaryField::Theta:
        case BoundaryField::TildeGamma:
        case BoundaryField::ATilde:
        case BoundaryField::Z:
            return z4c_characteristic_speed;
        case BoundaryField::GammaTildeInverse:
        default:
            return characteristic_speed;
        }
    }

    static inline void set_sponge(bool enabled, size_t width, double strength,
                                  double exponent = 2.0) {
        sponge_config.enabled = enabled;
        sponge_config.width = width;
        sponge_config.strength = std::max(strength, 0.0);
        sponge_config.exponent = std::max(exponent, 1.0);
    }

    static inline void set_rhs_sommerfeld_faces(bool ix1, bool ox1, bool ix2, bool ox2, bool ix3,
                                                bool ox3) {
        rhs_sommerfeld_face[0][0] = ix1;
        rhs_sommerfeld_face[0][1] = ox1;
        rhs_sommerfeld_face[1][0] = ix2;
        rhs_sommerfeld_face[1][1] = ox2;
        rhs_sommerfeld_face[2][0] = ix3;
        rhs_sommerfeld_face[2][1] = ox3;
    }

    static inline void set_reflective_faces(bool ix1, bool ox1, bool ix2, bool ox2, bool ix3,
                                            bool ox3) {
        reflective_face[0][0] = ix1;
        reflective_face[0][1] = ox1;
        reflective_face[1][0] = ix2;
        reflective_face[1][1] = ox2;
        reflective_face[2][0] = ix3;
        reflective_face[2][1] = ox3;
    }

    static inline void set_active_faces(bool ix1, bool ox1, bool ix2, bool ox2, bool ix3,
                                        bool ox3) {
        active_face[0][0] = ix1;
        active_face[0][1] = ox1;
        active_face[1][0] = ix2;
        active_face[1][1] = ox2;
        active_face[2][0] = ix3;
        active_face[2][1] = ox3;
    }

    static inline bool rhs_sommerfeld_enabled(int axis, bool outer) {
        const int ax = std::clamp(axis, 0, 2);
        return rhs_sommerfeld_face[ax][outer ? 1 : 0];
    }

    static inline bool reflective_enabled(int axis, bool outer) {
        const int ax = std::clamp(axis, 0, 2);
        return reflective_face[ax][outer ? 1 : 0];
    }

    static inline bool active_enabled(int axis, bool outer) {
        const int ax = std::clamp(axis, 0, 2);
        return active_face[ax][outer ? 1 : 0];
    }

    static inline int tensor_component_index_a(int component) {
        switch (component) {
        case XX:
            return 0;
        case XY:
            return 0;
        case XZ:
            return 0;
        case YY:
            return 1;
        case YZ:
            return 1;
        case ZZ:
            return 2;
        default:
            return 0;
        }
    }

    static inline int tensor_component_index_b(int component) {
        switch (component) {
        case XX:
            return 0;
        case XY:
            return 1;
        case XZ:
            return 2;
        case YY:
            return 1;
        case YZ:
            return 2;
        case ZZ:
            return 2;
        default:
            return 0;
        }
    }

    static inline int parity_sign(BoundaryField which, int component, int axis) {
        switch (which) {
        case BoundaryField::Alpha:
        case BoundaryField::Chi:
        case BoundaryField::K:
        case BoundaryField::Theta:
            return +1;
        case BoundaryField::Beta:
        case BoundaryField::B:
        case BoundaryField::TildeGamma:
        case BoundaryField::Z:
            return (component == axis) ? -1 : +1;
        case BoundaryField::GammaTilde:
        case BoundaryField::GammaTildeInverse:
        case BoundaryField::ATilde: {
            const int a = tensor_component_index_a(component);
            const int b = tensor_component_index_b(component);
            const int flips = (a == axis ? 1 : 0) + (b == axis ? 1 : 0);
            return (flips % 2 == 0) ? +1 : -1;
        }
        default:
            return +1;
        }
    }

    static inline bool use_sommerfeld_halo(BoundaryField which) {
        switch (which) {
        case BoundaryField::GammaTildeInverse:
            return false;
        default:
            return true;
        }
    }

    template <typename T>
    static inline void apply_physical(Field3D<T> &field, const BSSNGridSoA<T> &G,
                                      BoundaryField which, int component) {
        // Radiative runs now apply the sponge as an RHS damping term. Physical cells are left
        // untouched here so halo refreshes do not inject a Cartesian shell into the evolved state.
        (void)field;
        (void)G;
        (void)which;
        (void)component;
    }

    template <typename T>
    static inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                                  int component) {
        enum class HaloMode { Sommerfeld, Outflow, LinearExtrapolate, Skip };
        HaloMode mode = HaloMode::Outflow;
        if (which == BoundaryField::GammaTildeInverse)
            mode = HaloMode::Skip;
        else if (use_sommerfeld_halo(which))
            mode = HaloMode::Sommerfeld;
        if (mode == HaloMode::Skip)
            return;

        const auto  &D = G.dims;
        const double u_inf = detail::minkowski_target(which, component);
        const bool   coupled_khat = (which == BoundaryField::K);

        const size_t I0 = D.ng;
        const size_t I1 = D.ng + D.nx;
        const size_t J0 = D.ng;
        const size_t J1 = D.ng + D.ny;
        const size_t K0 = D.ng;
        const size_t K1 = D.ng + D.nz;

        auto *ptr = field.ptr();
        const T *theta_ptr = coupled_khat ? G.Theta.ptr() : nullptr;

        auto coord = [&](size_t i, size_t j, size_t k, double &x, double &y, double &z) {
            x = G.x0 + (double(i) - double(D.ng)) * G.dx;
            y = G.y0 + (double(j) - double(D.ng)) * G.dy;
            z = G.z0 + (double(k) - double(D.ng)) * G.dz;
        };

        auto r_of = [&](size_t i, size_t j, size_t k) {
            double x, y, z;
            coord(i, j, k, x, y, z);
            double r = std::sqrt(x * x + y * y + z * z);
            return (r > 1e-14) ? r : 1e-14;
        };

        auto set_sommerfeld = [&](size_t ob, size_t ib, double r_ob, double r_ib) {
            const double dt_wave = characteristic_dt;
            const double c_wave = characteristic_speed_for(which);
            double denom = r_ob;
            if (dt_wave > 0.0)
                denom += c_wave * dt_wave;
            denom = std::max(denom, 1.0e-12);
            if (coupled_khat) {
                const double khat_ib = double(ptr[ib]) - 2.0 * double(theta_ptr[ib]);
                const double khat_ob = khat_ib * (r_ib / denom);
                ptr[ob] = T(khat_ob + 2.0 * double(theta_ptr[ob]));
                return;
            }
            const double u_ib = double(ptr[ib]);
            const double du = u_ib - u_inf;
            ptr[ob] = T(u_inf + du * (r_ib / denom));
        };
        auto set_outflow = [&](size_t ob, size_t ib) {
            if (coupled_khat) {
                const double khat_ib = double(ptr[ib]) - 2.0 * double(theta_ptr[ib]);
                ptr[ob] = T(khat_ib + 2.0 * double(theta_ptr[ob]));
                return;
            }
            ptr[ob] = ptr[ib];
        };
        auto set_linear_extrapolated = [&](size_t ob, size_t ib0, size_t ib1, size_t layer) {
            if (coupled_khat) {
                const double khat0 = double(ptr[ib0]) - 2.0 * double(theta_ptr[ib0]);
                const double khat1 = double(ptr[ib1]) - 2.0 * double(theta_ptr[ib1]);
                const double khat_ob = (double(layer) + 1.0) * khat0 - double(layer) * khat1;
                ptr[ob] = T(khat_ob + 2.0 * double(theta_ptr[ob]));
                return;
            }
            const double u0 = double(ptr[ib0]);
            const double u1 = double(ptr[ib1]);
            ptr[ob] = T((double(layer) + 1.0) * u0 - double(layer) * u1);
        };
        auto set_reflective = [&](size_t ob, size_t ib_reflect, int axis) {
            const int sign = parity_sign(which, component, axis);
            if (coupled_khat) {
                const double khat_reflect =
                    double(ptr[ib_reflect]) - 2.0 * double(theta_ptr[ib_reflect]);
                ptr[ob] = T(double(sign) * khat_reflect + 2.0 * double(theta_ptr[ob]));
                return;
            }
            ptr[ob] = T(sign) * ptr[ib_reflect];
        };
        auto set_halo = [&](size_t ob, size_t ib, size_t ib0, size_t ib1, size_t layer,
                            double r_ob, double r_ib,
                            bool use_sommerfeld_face, bool use_reflective_face, int axis) {
            if (use_reflective_face) {
                set_reflective(ob, ib, axis);
                return;
            }
            if (mode == HaloMode::LinearExtrapolate) {
                set_linear_extrapolated(ob, ib0, ib1, layer);
            } else if (mode == HaloMode::Sommerfeld && use_sommerfeld_face)
                set_sommerfeld(ob, ib, r_ob, r_ib);
            else
                set_outflow(ob, ib);
        };

        const bool sf_ix1 = rhs_sommerfeld_enabled(0, false);
        const bool sf_ox1 = rhs_sommerfeld_enabled(0, true);
        const bool sf_ix2 = rhs_sommerfeld_enabled(1, false);
        const bool sf_ox2 = rhs_sommerfeld_enabled(1, true);
        const bool sf_ix3 = rhs_sommerfeld_enabled(2, false);
        const bool sf_ox3 = rhs_sommerfeld_enabled(2, true);
        const bool rf_ix1 = reflective_enabled(0, false);
        const bool rf_ox1 = reflective_enabled(0, true);
        const bool rf_ix2 = reflective_enabled(1, false);
        const bool rf_ox2 = reflective_enabled(1, true);
        const bool rf_ix3 = reflective_enabled(2, false);
        const bool rf_ox3 = reflective_enabled(2, true);
        const bool ac_ix1 = active_enabled(0, false);
        const bool ac_ox1 = active_enabled(0, true);
        const bool ac_ix2 = active_enabled(1, false);
        const bool ac_ox2 = active_enabled(1, true);
        const bool ac_ix3 = active_enabled(2, false);
        const bool ac_ox3 = active_enabled(2, true);

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k)
                    if (ac_ix1)
                        set_halo(field.idx(I0 - g, j, k), field.idx(I0 + (g - 1), j, k),
                                 field.idx(I0, j, k), field.idx(I0 + 1, j, k), g,
                                 r_of(I0 - g, j, k), r_of(I0 + (g - 1), j, k), sf_ix1, rf_ix1, 0);

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k)
                    if (ac_ox1)
                        set_halo(field.idx(I1 + g, j, k), field.idx(I1 - 1 - g, j, k),
                                 field.idx(I1 - 1, j, k), field.idx(I1 - 2, j, k), g + 1,
                                 r_of(I1 + g, j, k), r_of(I1 - 1 - g, j, k), sf_ox1, rf_ox1, 0);

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t k = K0; k < K1; ++k)
                    if (ac_ix2)
                        set_halo(field.idx(i, J0 - g, k), field.idx(i, J0 + (g - 1), k),
                                 field.idx(i, J0, k), field.idx(i, J0 + 1, k), g,
                                 r_of(i, J0 - g, k), r_of(i, J0 + (g - 1), k), sf_ix2, rf_ix2, 1);

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t k = K0; k < K1; ++k)
                    if (ac_ox2)
                        set_halo(field.idx(i, J1 + g, k), field.idx(i, J1 - 1 - g, k),
                                 field.idx(i, J1 - 1, k), field.idx(i, J1 - 2, k), g + 1,
                                 r_of(i, J1 + g, k), r_of(i, J1 - 1 - g, k), sf_ox2, rf_ox2, 1);

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t j = J0 - D.ng; j < J1 + D.ng; ++j)
                    if (ac_ix3)
                        set_halo(field.idx(i, j, K0 - g), field.idx(i, j, K0 + (g - 1)),
                                 field.idx(i, j, K0), field.idx(i, j, K0 + 1), g,
                                 r_of(i, j, K0 - g), r_of(i, j, K0 + (g - 1)), sf_ix3, rf_ix3, 2);

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t j = J0 - D.ng; j < J1 + D.ng; ++j)
                    if (ac_ox3)
                        set_halo(field.idx(i, j, K1 + g), field.idx(i, j, K1 - 1 - g),
                                 field.idx(i, j, K1 - 1), field.idx(i, j, K1 - 2), g + 1,
                                 r_of(i, j, K1 + g), r_of(i, j, K1 - 1 - g), sf_ox3, rf_ox3, 2);
    }
};

template <typename Boundary, typename T> inline void apply_halos_grid(BSSNGridSoA<T> &G) {
    detail::apply_batch_physical<Boundary>(G);
    detail::apply_batch_halo<Boundary>(G);
}

} // namespace tensorium_RG::bssn
