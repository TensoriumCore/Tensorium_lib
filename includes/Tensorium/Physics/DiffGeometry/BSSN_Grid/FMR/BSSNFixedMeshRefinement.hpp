#pragma once

#include "../Geometry/BSSNAlgebraic.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "../TimeIntegration/BSSNRK4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensorium_RG::bssn::fmr {

struct PatchBox {
    size_t i0 = 0, i1 = 0;
    size_t j0 = 0, j1 = 0;
    size_t k0 = 0, k1 = 0;

    [[nodiscard]] bool empty() const noexcept {
        return i0 >= i1 || j0 >= j1 || k0 >= k1;
    }

    [[nodiscard]] size_t nx() const noexcept { return i1 - i0; }
    [[nodiscard]] size_t ny() const noexcept { return j1 - j0; }
    [[nodiscard]] size_t nz() const noexcept { return k1 - k0; }
};

struct LevelConfig {
    PatchBox parent_cells{};
    size_t   refinement_ratio = 2;
    size_t   halo_cells = 0;
};

namespace detail {

class ScopedFDSpacing {
  public:
    explicit ScopedFDSpacing(double dx) : previous_(tensorium_RG::fd::fd_dx()) {
        tensorium_RG::fd::set_fd_dx(dx);
    }

    ~ScopedFDSpacing() { tensorium_RG::fd::set_fd_dx(previous_); }

  private:
    double previous_ = 1.0;
};

template <typename T> inline size_t total_entries(const Field3D<T> &field) {
    return field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
}

template <typename T> inline void copy_field(const Field3D<T> &src, Field3D<T> &dst) {
    std::copy_n(src.ptr(), total_entries(src), dst.ptr());
}

template <typename T> inline void copy_evolved_state(const BSSNGridSoA<T> &src, BSSNGridSoA<T> &dst) {
    dst.x0 = src.x0;
    dst.y0 = src.y0;
    dst.z0 = src.z0;

    copy_field(src.alpha, dst.alpha);
    copy_field(src.chi, dst.chi);
    copy_field(src.K, dst.K);
    copy_field(src.Theta, dst.Theta);

    for (int a = 0; a < 3; ++a) {
        copy_field(src.beta[a], dst.beta[a]);
        copy_field(src.B[a], dst.B[a]);
        copy_field(src.tildeGamma[a], dst.tildeGamma[a]);
        copy_field(src.Z[a], dst.Z[a]);
    }

    for (int s = 0; s < 6; ++s) {
        copy_field(src.gamma_tilde[s], dst.gamma_tilde[s]);
        copy_field(src.A_tilde[s], dst.A_tilde[s]);
    }
}

template <typename T>
inline void coords_any_index(const BSSNGridSoA<T> &grid, size_t i, size_t j, size_t k, T &x, T &y,
                             T &z) {
    const ptrdiff_t di = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(grid.dims.ng);
    const ptrdiff_t dj = static_cast<ptrdiff_t>(j) - static_cast<ptrdiff_t>(grid.dims.ng);
    const ptrdiff_t dk = static_cast<ptrdiff_t>(k) - static_cast<ptrdiff_t>(grid.dims.ng);
    x = grid.x0 + T(di) * grid.dx;
    y = grid.y0 + T(dj) * grid.dy;
    z = grid.z0 + T(dk) * grid.dz;
}

template <typename T>
inline const Field3D<T> &select_field(const BSSNGridSoA<T> &grid, BoundaryField which,
                                      int component) {
    switch (which) {
    case BoundaryField::Alpha:
        return grid.alpha;
    case BoundaryField::Chi:
        return grid.chi;
    case BoundaryField::K:
        return grid.K;
    case BoundaryField::Theta:
        return grid.Theta;
    case BoundaryField::Beta:
        return grid.beta[component];
    case BoundaryField::B:
        return grid.B[component];
    case BoundaryField::TildeGamma:
        return grid.tildeGamma[component];
    case BoundaryField::GammaTilde:
        return grid.gamma_tilde[component];
    case BoundaryField::GammaTildeInverse:
        return grid.gamma_tilde_inv[component];
    case BoundaryField::ATilde:
        return grid.A_tilde[component];
    case BoundaryField::Z:
        return grid.Z[component];
    }
    return grid.alpha;
}

template <typename T>
inline Field3D<T> &select_field(BSSNGridSoA<T> &grid, BoundaryField which, int component) {
    switch (which) {
    case BoundaryField::Alpha:
        return grid.alpha;
    case BoundaryField::Chi:
        return grid.chi;
    case BoundaryField::K:
        return grid.K;
    case BoundaryField::Theta:
        return grid.Theta;
    case BoundaryField::Beta:
        return grid.beta[component];
    case BoundaryField::B:
        return grid.B[component];
    case BoundaryField::TildeGamma:
        return grid.tildeGamma[component];
    case BoundaryField::GammaTilde:
        return grid.gamma_tilde[component];
    case BoundaryField::GammaTildeInverse:
        return grid.gamma_tilde_inv[component];
    case BoundaryField::ATilde:
        return grid.A_tilde[component];
    case BoundaryField::Z:
        return grid.Z[component];
    }
    return grid.alpha;
}

template <typename T>
inline T sample_trilinear_field(const BSSNGridSoA<T> &src, const Field3D<T> &field, T x, T y,
                                T z) {
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
inline T sample_temporal_field(const BSSNGridSoA<T> &coarse_old, const BSSNGridSoA<T> &coarse_new,
                               T lambda, BoundaryField which, int component, T x, T y, T z) {
    const T clamped = std::clamp(lambda, T(0), T(1));
    const T old_value =
        sample_trilinear_field(coarse_old, select_field(coarse_old, which, component), x, y, z);
    if (&coarse_old == &coarse_new || clamped == T(0))
        return old_value;
    const T new_value =
        sample_trilinear_field(coarse_new, select_field(coarse_new, which, component), x, y, z);
    return (T(1) - clamped) * old_value + clamped * new_value;
}

struct ChildGeometry {
    size_t nx = 0, ny = 0, nz = 0, ng = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;
};

template <typename T>
inline ChildGeometry make_child_geometry(const BSSNGridSoA<T> &parent, const LevelConfig &cfg) {
    if (cfg.parent_cells.empty())
        throw std::invalid_argument("FMR level patch must contain at least one parent cell");
    if (cfg.refinement_ratio < 2)
        throw std::invalid_argument("FMR refinement ratio must be >= 2");

    const PatchBox &box = cfg.parent_cells;
    if (box.i1 > parent.dims.nx || box.j1 > parent.dims.ny || box.k1 > parent.dims.nz)
        throw std::invalid_argument("FMR patch extends outside the parent physical domain");

    constexpr size_t min_clearance = 2;
    if (box.i0 < min_clearance || box.j0 < min_clearance || box.k0 < min_clearance ||
        box.i1 + min_clearance > parent.dims.nx || box.j1 + min_clearance > parent.dims.ny ||
        box.k1 + min_clearance > parent.dims.nz) {
        throw std::invalid_argument(
            "FMR patch must stay at least two coarse cells away from the parent outer boundary");
    }

    const size_t ratio = cfg.refinement_ratio;
    const double inv_ratio = 1.0 / static_cast<double>(ratio);
    const size_t ng = (cfg.halo_cells == 0) ? parent.dims.ng : cfg.halo_cells;

    ChildGeometry child;
    child.nx = box.nx() * ratio;
    child.ny = box.ny() * ratio;
    child.nz = box.nz() * ratio;
    child.ng = ng;
    child.dx = static_cast<double>(parent.dx) * inv_ratio;
    child.dy = static_cast<double>(parent.dy) * inv_ratio;
    child.dz = static_cast<double>(parent.dz) * inv_ratio;
    child.x0 = static_cast<double>(parent.x0) + static_cast<double>(box.i0) * parent.dx -
               0.5 * static_cast<double>(parent.dx) + 0.5 * child.dx;
    child.y0 = static_cast<double>(parent.y0) + static_cast<double>(box.j0) * parent.dy -
               0.5 * static_cast<double>(parent.dy) + 0.5 * child.dy;
    child.z0 = static_cast<double>(parent.z0) + static_cast<double>(box.k0) * parent.dz -
               0.5 * static_cast<double>(parent.dz) + 0.5 * child.dz;
    return child;
}

template <typename T>
inline void prolongate_field_from_parent(const BSSNGridSoA<T> &parent, BSSNGridSoA<T> &child,
                                         BoundaryField which, int component) {
    Field3D<T> &dst = select_field(child, which, component);
    T          *out = dst.ptr();

#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < dst.st.nx_tot; ++i)
        for (size_t j = 0; j < dst.st.ny_tot; ++j)
            for (size_t k = 0; k < dst.st.nz_tot; ++k) {
                T x, y, z;
                coords_any_index(child, i, j, k, x, y, z);
                out[dst.idx(i, j, k)] =
                    sample_trilinear_field(parent, select_field(parent, which, component), x, y, z);
            }
}

template <typename T>
inline void prolongate_from_parent(const BSSNGridSoA<T> &parent, BSSNGridSoA<T> &child) {
    prolongate_field_from_parent(parent, child, BoundaryField::Alpha, 0);
    prolongate_field_from_parent(parent, child, BoundaryField::Chi, 0);
    prolongate_field_from_parent(parent, child, BoundaryField::K, 0);
    prolongate_field_from_parent(parent, child, BoundaryField::Theta, 0);
    for (int a = 0; a < 3; ++a) {
        prolongate_field_from_parent(parent, child, BoundaryField::Beta, a);
        prolongate_field_from_parent(parent, child, BoundaryField::B, a);
        prolongate_field_from_parent(parent, child, BoundaryField::TildeGamma, a);
        prolongate_field_from_parent(parent, child, BoundaryField::Z, a);
    }
    for (int s = 0; s < 6; ++s) {
        prolongate_field_from_parent(parent, child, BoundaryField::GammaTilde, s);
        prolongate_field_from_parent(parent, child, BoundaryField::ATilde, s);
    }

    tensorium_RG::bssn::enforce_algebraic_constraints(child);
}

template <typename T>
inline void blend_field_shell_from_parent(const BSSNGridSoA<T> &parent, BSSNGridSoA<T> &child,
                                          size_t shell_cells, BoundaryField which, int component) {
    if (shell_cells == 0)
        return;

    Field3D<T> &dst = select_field(child, which, component);

    size_t I0, I1, J0, J1, K0, K1;
    child.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t dist_i = std::min(i - I0, (I1 - 1) - i);
                const size_t dist_j = std::min(j - J0, (J1 - 1) - j);
                const size_t dist_k = std::min(k - K0, (K1 - 1) - k);
                const size_t dist = std::min({dist_i, dist_j, dist_k});
                if (dist >= shell_cells)
                    continue;

                T x, y, z;
                coords_any_index(child, i, j, k, x, y, z);
                const T parent_value =
                    sample_trilinear_field(parent, select_field(parent, which, component), x, y, z);
                const size_t idx = dst.idx(i, j, k);
                const T child_value = dst.ptr()[idx];

                const T s = std::clamp(T(dist) / T(shell_cells), T(0), T(1));
                const T fine_weight = s * s * (T(3) - T(2) * s);
                dst.ptr()[idx] = fine_weight * child_value + (T(1) - fine_weight) * parent_value;
            }
}

template <typename T>
inline void blend_shell_from_parent(const BSSNGridSoA<T> &parent, BSSNGridSoA<T> &child,
                                    size_t shell_cells) {
    blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::Alpha, 0);
    blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::Chi, 0);
    blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::K, 0);
    blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::Theta, 0);
    for (int a = 0; a < 3; ++a) {
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::Beta, a);
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::B, a);
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::TildeGamma, a);
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::Z, a);
    }
    for (int s = 0; s < 6; ++s) {
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::GammaTilde, s);
        blend_field_shell_from_parent(parent, child, shell_cells, BoundaryField::ATilde, s);
    }

    tensorium_RG::bssn::enforce_algebraic_constraints(child);
}

template <typename T>
inline bool same_grid_geometry(const BSSNGridSoA<T> &lhs, const BSSNGridSoA<T> &rhs) {
    const auto near_equal = [](T a, T b) {
        const T scale = std::max({T(1), std::abs(a), std::abs(b)});
        return std::abs(a - b) <= T(32) * std::numeric_limits<T>::epsilon() * scale;
    };

    return lhs.dims.nx == rhs.dims.nx && lhs.dims.ny == rhs.dims.ny && lhs.dims.nz == rhs.dims.nz &&
           lhs.dims.ng == rhs.dims.ng && near_equal(lhs.dx, rhs.dx) && near_equal(lhs.dy, rhs.dy) &&
           near_equal(lhs.dz, rhs.dz) && near_equal(lhs.x0, rhs.x0) && near_equal(lhs.y0, rhs.y0) &&
           near_equal(lhs.z0, rhs.z0);
}

template <typename T>
inline void blend_field_centered_core_from_reference(const BSSNGridSoA<T> &reference,
                                                     BSSNGridSoA<T> &target, T core_half_width,
                                                     T transition_width, BoundaryField which,
                                                     int component) {
    if (core_half_width < T(0))
        core_half_width = T(0);
    if (transition_width < T(0))
        transition_width = T(0);

    const T outer_half_width = core_half_width + transition_width;
    if (!(outer_half_width > T(0)))
        return;

    const Field3D<T> &src = select_field(reference, which, component);
    Field3D<T>       &dst = select_field(target, which, component);

    size_t I0, I1, J0, J1, K0, K1;
    target.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                T x, y, z;
                coords_any_index(target, i, j, k, x, y, z);
                const T centered_radius = std::max({std::abs(x), std::abs(y), std::abs(z)});
                if (centered_radius >= outer_half_width)
                    continue;

                T reference_weight = T(1);
                if (centered_radius > core_half_width) {
                    if (!(transition_width > T(0)))
                        reference_weight = T(0);
                    else {
                        const T s = std::clamp((outer_half_width - centered_radius) / transition_width,
                                               T(0), T(1));
                        reference_weight = s * s * (T(3) - T(2) * s);
                    }
                }

                const size_t idx = dst.idx(i, j, k);
                const T dst_value = dst.ptr()[idx];
                const T src_value = src.ptr()[idx];
                dst.ptr()[idx] = (T(1) - reference_weight) * dst_value + reference_weight * src_value;
            }
}

template <typename T>
inline void blend_centered_core_from_reference(const BSSNGridSoA<T> &reference,
                                               BSSNGridSoA<T> &target, T core_half_width,
                                               T transition_width) {
    blend_field_centered_core_from_reference(reference, target, core_half_width, transition_width,
                                             BoundaryField::Alpha, 0);
    blend_field_centered_core_from_reference(reference, target, core_half_width, transition_width,
                                             BoundaryField::Chi, 0);
    blend_field_centered_core_from_reference(reference, target, core_half_width, transition_width,
                                             BoundaryField::K, 0);
    blend_field_centered_core_from_reference(reference, target, core_half_width, transition_width,
                                             BoundaryField::Theta, 0);
    for (int a = 0; a < 3; ++a) {
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::Beta, a);
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::B, a);
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::TildeGamma, a);
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::Z, a);
    }
    for (int s = 0; s < 6; ++s) {
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::GammaTilde, s);
        blend_field_centered_core_from_reference(reference, target, core_half_width,
                                                 transition_width, BoundaryField::ATilde, s);
    }

    tensorium_RG::bssn::enforce_algebraic_constraints(target);
}

template <typename T>
inline void restrict_field_average(const BSSNGridSoA<T> &child, BSSNGridSoA<T> &parent,
                                   BoundaryField which, int component, const PatchBox &parent_box,
                                   size_t ratio) {
    const Field3D<T> &src = select_field(child, which, component);
    Field3D<T>       &dst = select_field(parent, which, component);
    const size_t      base_i = child.dims.ng;
    const size_t      base_j = child.dims.ng;
    const size_t      base_k = child.dims.ng;
    const T           inv_volume = T(1) / T(ratio * ratio * ratio);

#pragma omp parallel for collapse(3)
    for (size_t ip = parent_box.i0; ip < parent_box.i1; ++ip)
        for (size_t jp = parent_box.j0; jp < parent_box.j1; ++jp)
            for (size_t kp = parent_box.k0; kp < parent_box.k1; ++kp) {
                const size_t child_i0 = base_i + (ip - parent_box.i0) * ratio;
                const size_t child_j0 = base_j + (jp - parent_box.j0) * ratio;
                const size_t child_k0 = base_k + (kp - parent_box.k0) * ratio;

                T accum = T(0);
                for (size_t fi = 0; fi < ratio; ++fi)
                    for (size_t fj = 0; fj < ratio; ++fj)
                        for (size_t fk = 0; fk < ratio; ++fk)
                            accum += src.ptr()[src.idx(child_i0 + fi, child_j0 + fj, child_k0 + fk)];

                const size_t parent_i = parent.dims.ng + ip;
                const size_t parent_j = parent.dims.ng + jp;
                const size_t parent_k = parent.dims.ng + kp;
                dst.ptr()[dst.idx(parent_i, parent_j, parent_k)] = accum * inv_volume;
            }
}

template <typename T>
inline void restrict_from_child(const BSSNGridSoA<T> &child, BSSNGridSoA<T> &parent,
                                const PatchBox &parent_box, size_t ratio) {
    restrict_field_average(child, parent, BoundaryField::Alpha, 0, parent_box, ratio);
    restrict_field_average(child, parent, BoundaryField::Chi, 0, parent_box, ratio);
    restrict_field_average(child, parent, BoundaryField::K, 0, parent_box, ratio);
    restrict_field_average(child, parent, BoundaryField::Theta, 0, parent_box, ratio);
    for (int a = 0; a < 3; ++a) {
        restrict_field_average(child, parent, BoundaryField::Beta, a, parent_box, ratio);
        restrict_field_average(child, parent, BoundaryField::B, a, parent_box, ratio);
        restrict_field_average(child, parent, BoundaryField::TildeGamma, a, parent_box, ratio);
        restrict_field_average(child, parent, BoundaryField::Z, a, parent_box, ratio);
    }
    for (int s = 0; s < 6; ++s) {
        restrict_field_average(child, parent, BoundaryField::GammaTilde, s, parent_box, ratio);
        restrict_field_average(child, parent, BoundaryField::ATilde, s, parent_box, ratio);
    }

    tensorium_RG::bssn::enforce_algebraic_constraints(parent);
}

} // namespace detail

template <typename T, typename OuterBoundary = BoundaryRadiative> struct ParentInterpolationBoundary {
    struct Context {
        const BSSNGridSoA<T> *coarse_old = nullptr;
        const BSSNGridSoA<T> *coarse_new = nullptr;
        T                     lambda_begin = T(0);
        T                     lambda_end = T(0);

        Context() = default;

        Context(const BSSNGridSoA<T> *old_state, const BSSNGridSoA<T> *new_state, T lambda)
            : coarse_old(old_state), coarse_new(new_state), lambda_begin(lambda), lambda_end(lambda) {}

        Context(const BSSNGridSoA<T> *old_state, const BSSNGridSoA<T> *new_state, T begin_lambda,
                T end_lambda)
            : coarse_old(old_state), coarse_new(new_state), lambda_begin(begin_lambda),
              lambda_end(end_lambda) {}
    };

    inline static constexpr bool evolve_physical_cells = true;
    inline static constexpr bool radiative_rhs_collar_enabled = false;
    inline static const Context *current_context = nullptr;
    inline static T              current_stage_fraction = T(1);

    static inline void set_context(const Context *ctx) { current_context = ctx; }
    static inline void set_stage_fraction(T fraction) {
        current_stage_fraction = std::clamp(fraction, T(0), T(1));
    }

    template <typename U>
    static inline void apply_physical(Field3D<U> &field, const BSSNGridSoA<U> &grid,
                                      BoundaryField which, int component) {
        if (current_context == nullptr)
            OuterBoundary::apply_physical(field, grid, which, component);
    }

    template <typename U>
    static inline void apply_halo(Field3D<U> &field, const BSSNGridSoA<U> &grid, BoundaryField which,
                                  int component) {
        if (current_context == nullptr) {
            OuterBoundary::apply_halo(field, grid, which, component);
            return;
        }
        if (current_context->coarse_old == nullptr || current_context->coarse_new == nullptr)
            throw std::runtime_error("FMR parent interpolation boundary is missing coarse data");
        if (which == BoundaryField::GammaTildeInverse)
            return;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        U *ptr = field.ptr();

#pragma omp parallel for collapse(3)
        for (size_t i = 0; i < field.st.nx_tot; ++i)
            for (size_t j = 0; j < field.st.ny_tot; ++j)
                for (size_t k = 0; k < field.st.nz_tot; ++k) {
                    const bool in_physical = (i >= I0 && i < I1 && j >= J0 && j < J1 && k >= K0 &&
                                              k < K1);
                    if (in_physical)
                        continue;

                    U x, y, z;
                    detail::coords_any_index(grid, i, j, k, x, y, z);
                    const T lambda = std::clamp(
                        current_context->lambda_begin +
                            current_stage_fraction *
                                (current_context->lambda_end - current_context->lambda_begin),
                        T(0), T(1));
                    ptr[field.idx(i, j, k)] = detail::sample_temporal_field(
                        *current_context->coarse_old, *current_context->coarse_new,
                        lambda, which, component, x, y, z);
                }
    }
};

template <typename T, typename OuterBoundary = BoundaryRadiative>
class ScopedParentInterpolationContext {
  public:
    using BoundaryType = ParentInterpolationBoundary<T, OuterBoundary>;
    using Context = typename BoundaryType::Context;

    explicit ScopedParentInterpolationContext(const Context *ctx)
        : previous_(BoundaryType::current_context) {
        BoundaryType::set_context(ctx);
    }

    ~ScopedParentInterpolationContext() { BoundaryType::set_context(previous_); }

  private:
    const Context *previous_ = nullptr;
};

template <typename T, typename OuterBoundary = BoundaryRadiative>
class FixedMeshRefinementHierarchy {
  public:
    using RootStepper = BSSNRKStepper<T, OuterBoundary>;
    using FineBoundary = ParentInterpolationBoundary<T, OuterBoundary>;
    using FineStepper = BSSNRKStepper<T, FineBoundary>;

    struct LevelState {
        BSSNGridSoA<T> grid;
        BSSNGridSoA<T> snapshot;
        PatchBox       parent_cells{};
        size_t         refinement_ratio = 1;
        size_t         step_count = 0;

        LevelState(size_t nx, size_t ny, size_t nz, size_t ng, T dx, T dy, T dz, T x0, T y0,
                   T z0, const PatchBox &box, size_t ratio)
            : grid(nx, ny, nz, ng, dx, dy, dz),
              snapshot(nx, ny, nz, ng, dx, dy, dz),
              parent_cells(box),
              refinement_ratio(ratio) {
            grid.x0 = x0;
            grid.y0 = y0;
            grid.z0 = z0;
            snapshot.x0 = x0;
            snapshot.y0 = y0;
            snapshot.z0 = z0;
        }
    };

    explicit FixedMeshRefinementHierarchy(const BSSNGridSoA<T> &root_state,
                                          const std::vector<LevelConfig> &level_configs,
                                          size_t padding = 4)
        : FixedMeshRefinementHierarchy(root_state, level_configs, tensorium::backend::Options{},
                                       padding) {}

    FixedMeshRefinementHierarchy(const BSSNGridSoA<T> &root_state,
                                 const std::vector<LevelConfig> &level_configs,
                                 const tensorium::backend::Options &backend, size_t padding = 4)
        : padding_(padding), backend_options_(backend) {
        levels_.reserve(level_configs.size() + 1);
        levels_.push_back(std::make_unique<LevelState>(
            root_state.dims.nx, root_state.dims.ny, root_state.dims.nz, root_state.dims.ng,
            root_state.dx, root_state.dy, root_state.dz, root_state.x0, root_state.y0,
            root_state.z0, PatchBox{}, size_t(1)));
        detail::copy_evolved_state(root_state, levels_.front()->grid);
        tensorium_RG::bssn::enforce_algebraic_constraints(levels_.front()->grid);
        detail::copy_evolved_state(levels_.front()->grid, levels_.front()->snapshot);

        root_stepper_ = std::make_unique<RootStepper>(levels_.front()->grid, backend_options_,
                                                      padding_);

        const BSSNGridSoA<T> *parent = &levels_.front()->grid;
        for (const LevelConfig &cfg : level_configs) {
            const auto child = detail::make_child_geometry(*parent, cfg);
            levels_.push_back(std::make_unique<LevelState>(
                child.nx, child.ny, child.nz, child.ng, T(child.dx), T(child.dy), T(child.dz),
                T(child.x0), T(child.y0), T(child.z0), cfg.parent_cells, cfg.refinement_ratio));
            fine_steppers_.push_back(
                std::make_unique<FineStepper>(levels_.back()->grid, backend_options_, padding_));
            parent = &levels_.back()->grid;
        }

        initialize_nested_levels();
    }

    [[nodiscard]] size_t num_levels() const noexcept { return levels_.size(); }

    BSSNGridSoA<T> &root_grid() { return levels_.front()->grid; }
    const BSSNGridSoA<T> &root_grid() const { return levels_.front()->grid; }

    BSSNGridSoA<T> &level_grid(size_t level) { return levels_.at(level)->grid; }
    const BSSNGridSoA<T> &level_grid(size_t level) const { return levels_.at(level)->grid; }

    void initialize_nested_levels() {
        for (size_t level = 1; level < levels_.size(); ++level)
            prolongate_level_from_parent(level);
    }

    void prolongate_level_from_parent(size_t level) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR prolongation requires a valid child level");
        detail::prolongate_from_parent(levels_[level - 1]->grid, levels_[level]->grid);
        detail::copy_evolved_state(levels_[level]->grid, levels_[level]->snapshot);
    }

    void restrict_level_to_parent(size_t level) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR restriction requires a valid child level");
        detail::restrict_from_child(levels_[level]->grid, levels_[level - 1]->grid,
                                    levels_[level]->parent_cells, levels_[level]->refinement_ratio);
        detail::copy_evolved_state(levels_[level - 1]->grid, levels_[level - 1]->snapshot);
    }

    void blend_level_shell_from_parent(size_t level, size_t shell_cells) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR shell blending requires a valid child level");
        detail::blend_shell_from_parent(levels_[level - 1]->grid, levels_[level]->grid, shell_cells);
        detail::copy_evolved_state(levels_[level]->grid, levels_[level]->snapshot);
    }

    void blend_level_centered_core_from_reference(size_t level, const BSSNGridSoA<T> &reference,
                                                  T core_half_width, T transition_width) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR centered-core blending requires a valid child level");
        if (!detail::same_grid_geometry(levels_[level]->grid, reference)) {
            throw std::invalid_argument(
                "FMR centered-core blending requires a reference grid with identical geometry");
        }
        detail::blend_centered_core_from_reference(reference, levels_[level]->grid, core_half_width,
                                                   transition_width);
        detail::copy_evolved_state(levels_[level]->grid, levels_[level]->snapshot);
    }

    void restrict_all_levels_to_root() {
        for (size_t level = levels_.size(); level-- > 1;)
            restrict_level_to_parent(level);
    }

    void apply_level_boundaries() {
        {
            const detail::ScopedFDSpacing spacing(levels_.front()->grid.dx);
            tensorium_RG::bssn::apply_halos_grid<OuterBoundary>(levels_.front()->grid);
        }

        for (size_t level = 1; level < levels_.size(); ++level) {
            const typename FineBoundary::Context ctx{&levels_[level - 1]->grid,
                                                     &levels_[level - 1]->grid, T(0)};
            const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
            const detail::ScopedFDSpacing spacing(levels_[level]->grid.dx);
            tensorium_RG::bssn::apply_halos_grid<FineBoundary>(levels_[level]->grid);
        }
    }

    void set_gauge_parameters(const GaugeParameters<T> &params) {
        gauge_params_ = params;
        root_stepper_->set_gauge_parameters(params);
        for (auto &stepper : fine_steppers_)
            stepper->set_gauge_parameters(params);
    }

    void set_state_log_stride(size_t stride) {
        root_stepper_->set_state_log_stride(stride);
        for (auto &stepper : fine_steppers_)
            stepper->set_state_log_stride(stride);
    }

    void set_backend_options(const tensorium::backend::Options &backend) {
        backend_options_ = backend;
        root_stepper_->set_backend_options(backend);
        for (auto &stepper : fine_steppers_)
            stepper->set_backend_options(backend);
    }

    [[nodiscard]] const tensorium::backend::Options &backend_options() const noexcept {
        return backend_options_;
    }

    [[nodiscard]] T compute_dt(const CFLControl<T> &control) const {
        T      dt_root = std::numeric_limits<T>::infinity();
        size_t cumulative_ratio = 1;
        for (size_t level = 0; level < levels_.size(); ++level) {
            const T local_dt =
                tensorium_RG::bssn::compute_dt_cfl(levels_[level]->grid, control, padding_);
            dt_root = std::min(dt_root, local_dt * T(cumulative_ratio));
            if (level + 1 < levels_.size())
                cumulative_ratio *= levels_[level + 1]->refinement_ratio;
        }
        return dt_root;
    }

    void step(T dt_root) { advance_level(0, dt_root); }

  private:
    size_t                                        padding_ = 4;
    GaugeParameters<T>                            gauge_params_{};
    tensorium::backend::Options                   backend_options_{};
    std::vector<std::unique_ptr<LevelState>>      levels_;
    std::unique_ptr<RootStepper>                  root_stepper_;
    std::vector<std::unique_ptr<FineStepper>>     fine_steppers_;

    void advance_level(size_t level_idx, T dt) {
        LevelState &level = *levels_[level_idx];
        detail::copy_evolved_state(level.grid, level.snapshot);

        {
            const detail::ScopedFDSpacing spacing(level.grid.dx);
            if (level_idx == 0)
                root_stepper_->step(level.grid, dt, level.step_count++);
            else
                fine_steppers_[level_idx - 1]->step(level.grid, dt, level.step_count++);
        }

        if (level_idx + 1 >= levels_.size())
            return;

        LevelState &child = *levels_[level_idx + 1];
        const size_t ratio = child.refinement_ratio;
        const T      dt_child = dt / T(ratio);

        for (size_t substep = 0; substep < ratio; ++substep) {
            const typename FineBoundary::Context ctx{
                &level.snapshot, &level.grid, T(substep) / T(ratio), T(substep + 1) / T(ratio)};
            const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
            advance_level(level_idx + 1, dt_child);
        }

        detail::restrict_from_child(child.grid, level.grid, child.parent_cells, child.refinement_ratio);
    }
};

} // namespace tensorium_RG::bssn::fmr
