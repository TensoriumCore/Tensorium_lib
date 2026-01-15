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
    Beta,
    B,
    TildeGamma,
    GammaTilde,
    GammaTildeInverse,
    ATilde,
    GammaTildeCache
};

namespace detail {

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

    for (int c = 0; c < 3; ++c) {
        apply_physical<Boundary>(G.beta[c], G, BoundaryField::Beta, c);
        apply_physical<Boundary>(G.B[c], G, BoundaryField::B, c);
        apply_physical<Boundary>(G.tildeGamma[c], G, BoundaryField::TildeGamma, c);
    }

    for (int s = 0; s < 6; ++s) {
        apply_physical<Boundary>(G.gamma_tilde[s], G, BoundaryField::GammaTilde, s);
        apply_physical<Boundary>(G.gamma_tilde_inv[s], G, BoundaryField::GammaTildeInverse, s);
        apply_physical<Boundary>(G.A_tilde[s], G, BoundaryField::ATilde, s);
    }

    for (int q = 0; q < 27; ++q)
        apply_physical<Boundary>(G.Gamma_tilde[q], G, BoundaryField::GammaTildeCache, q);
}

template <typename Boundary, typename T> inline void apply_batch_halo(BSSNGridSoA<T> &G) {
    apply_halo<Boundary>(G.alpha, G, BoundaryField::Alpha, 0);
    apply_halo<Boundary>(G.chi, G, BoundaryField::Chi, 0);
    apply_halo<Boundary>(G.K, G, BoundaryField::K, 0);

    for (int c = 0; c < 3; ++c) {
        apply_halo<Boundary>(G.beta[c], G, BoundaryField::Beta, c);
        apply_halo<Boundary>(G.B[c], G, BoundaryField::B, c);
        apply_halo<Boundary>(G.tildeGamma[c], G, BoundaryField::TildeGamma, c);
    }

    for (int s = 0; s < 6; ++s) {
        apply_halo<Boundary>(G.gamma_tilde[s], G, BoundaryField::GammaTilde, s);
        apply_halo<Boundary>(G.gamma_tilde_inv[s], G, BoundaryField::GammaTildeInverse, s);
        apply_halo<Boundary>(G.A_tilde[s], G, BoundaryField::ATilde, s);
    }

    for (int q = 0; q < 27; ++q)
        apply_halo<Boundary>(G.Gamma_tilde[q], G, BoundaryField::GammaTildeCache, q);
}

inline double minkowski_target(BoundaryField which, int component) {
    switch (which) {
    case BoundaryField::Alpha:
    case BoundaryField::Chi:
        return 1.0;
    case BoundaryField::Beta:
    case BoundaryField::B:
    case BoundaryField::TildeGamma:
    case BoundaryField::ATilde:
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
    template <typename T>
    static inline void apply_physical(Field3D<T> &, const BSSNGridSoA<T> &, BoundaryField, int) {}

    template <typename T>
    static inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField, int) {
        halo_linear_extrapolate_full(field, G.dims);
    }
};

struct BoundaryRadiative {
    template <typename T>
    static inline void apply_physical(Field3D<T> &, const BSSNGridSoA<T> &, BoundaryField, int) {}

    template <typename T>
    static inline void apply_halo(Field3D<T> &field, const BSSNGridSoA<T> &G, BoundaryField which,
                                  int component) {
        const auto  &D = G.dims;
        const double u_inf = detail::minkowski_target(which, component);

        const size_t I0 = D.ng;
        const size_t I1 = D.ng + D.nx;
        const size_t J0 = D.ng;
        const size_t J1 = D.ng + D.ny;
        const size_t K0 = D.ng;
        const size_t K1 = D.ng + D.nz;

        auto *ptr = field.ptr();

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
            double u_ib = double(ptr[ib]);
            double du = u_ib - u_inf;
            ptr[ob] = T(u_inf + du * (r_ib / r_ob));
        };

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k)
                    set_sommerfeld(field.idx(I0 - g, j, k), field.idx(I0, j, k), r_of(I0 - g, j, k),
                                   r_of(I0, j, k));

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K0; k < K1; ++k)
                    set_sommerfeld(field.idx(I1 + g, j, k), field.idx(I1 - 1, j, k),
                                   r_of(I1 + g, j, k), r_of(I1 - 1, j, k));

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t k = K0; k < K1; ++k)
                    set_sommerfeld(field.idx(i, J0 - g, k), field.idx(i, J0, k), r_of(i, J0 - g, k),
                                   r_of(i, J0, k));

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t k = K0; k < K1; ++k)
                    set_sommerfeld(field.idx(i, J1 + g, k), field.idx(i, J1 - 1, k),
                                   r_of(i, J1 + g, k), r_of(i, J1 - 1, k));

        for (size_t g = 1; g <= D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t j = J0 - D.ng; j < J1 + D.ng; ++j)
                    set_sommerfeld(field.idx(i, j, K0 - g), field.idx(i, j, K0), r_of(i, j, K0 - g),
                                   r_of(i, j, K0));

        for (size_t g = 0; g < D.ng; ++g)
            for (size_t i = I0 - D.ng; i < I1 + D.ng; ++i)
                for (size_t j = J0 - D.ng; j < J1 + D.ng; ++j)
                    set_sommerfeld(field.idx(i, j, K1 + g), field.idx(i, j, K1 - 1),
                                   r_of(i, j, K1 + g), r_of(i, j, K1 - 1));
    }
};

template <typename Boundary, typename T> inline void apply_halos_grid(BSSNGridSoA<T> &G) {
    detail::apply_batch_physical<Boundary>(G);
    detail::apply_batch_halo<Boundary>(G);
}

} // namespace tensorium_RG::bssn
