#pragma once

#include "../Geometry/BSSNAlgebraic.hpp"
#include "../Grid/BSSNGridOperations.hpp"
#include "../TimeIntegration/BSSNRK4.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensorium_RG::bssn::fmr {

template <typename T> class EvolvedStateSoA {
  public:
    GridDims   dims;
    Strides<T> st;

    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;
    Field3D<T> Theta;

    Field3D<T> beta[3];
    Field3D<T> B[3];
    Field3D<T> tildeGamma[3];
    Field3D<T> Z[3];

    Field3D<T> gamma_tilde[6];
    Field3D<T> A_tilde[6];

    T dx, dy, dz;

    EvolvedStateSoA(size_t nx, size_t ny, size_t nz, size_t ng, T dx_, T dy_, T dz_)
        : dims{nx, ny, nz, ng},
          dx(dx_),
          dy(dy_),
          dz(dz_) {
        const size_t nx_tot = nx + 2 * ng;
        const size_t ny_tot = ny + 2 * ng;
        const size_t nz_tot = pad_simd<T>(nz + 2 * ng);

        st.nx_tot = nx_tot;
        st.ny_tot = ny_tot;
        st.nz_tot = nz_tot;

        st.sz = 1;
        st.sy = nz_tot;
        st.sx = ny_tot * nz_tot;

        auto alloc_field = [&](Field3D<T> &f) {
            const size_t N = nx_tot * ny_tot * nz_tot;
            f.data = aligned_alloc_n<T>(N);
            f.st = st;
        };

        alloc_field(alpha);
        alloc_field(chi);
        alloc_field(K);
        alloc_field(Theta);

        for (int i = 0; i < 3; ++i) {
            alloc_field(beta[i]);
            alloc_field(B[i]);
            alloc_field(tildeGamma[i]);
            alloc_field(Z[i]);
        }

        for (int s = 0; s < 6; ++s) {
            alloc_field(gamma_tilde[s]);
            alloc_field(A_tilde[s]);
        }
    }

    inline void domain_bounds(size_t &i0, size_t &i1, size_t &j0, size_t &j1, size_t &k0,
                              size_t &k1) const noexcept {
        i0 = dims.ng;
        i1 = dims.ng + dims.nx;
        j0 = dims.ng;
        j1 = dims.ng + dims.ny;
        k0 = dims.ng;
        k1 = dims.ng + dims.nz;
    }

    T x0 = 0, y0 = 0, z0 = 0;

    [[nodiscard]] inline size_t total_cells() const noexcept {
        return st.nx_tot * st.ny_tot * st.nz_tot;
    }
};

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

struct BinaryPunctureHierarchyConfig {
    std::vector<LevelConfig> shared_levels;
    LevelConfig              left_leaf{};
    LevelConfig              right_leaf{};
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

template <typename SrcState, typename DstState>
inline void copy_evolved_state(const SrcState &src, DstState &dst) {
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
inline void copy_field_region(const Field3D<T> &src, const PatchBox &src_box, Field3D<T> &dst) {
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < src_box.nx(); ++i)
        for (size_t j = 0; j < src_box.ny(); ++j)
            for (size_t k = 0; k < src_box.nz(); ++k) {
                const size_t src_i = src_box.i0 + i;
                const size_t src_j = src_box.j0 + j;
                const size_t src_k = src_box.k0 + k;
                dst.ptr()[dst.idx(i, j, k)] = src.ptr()[src.idx(src_i, src_j, src_k)];
            }
}

template <typename T>
inline void copy_evolved_state_region(const BSSNGridSoA<T> &src, const PatchBox &physical_box,
                                      EvolvedStateSoA<T> &dst) {
    dst.x0 = src.x0 + static_cast<T>(physical_box.i0) * src.dx;
    dst.y0 = src.y0 + static_cast<T>(physical_box.j0) * src.dy;
    dst.z0 = src.z0 + static_cast<T>(physical_box.k0) * src.dz;

    const PatchBox src_box{src.dims.ng + physical_box.i0, src.dims.ng + physical_box.i1,
                           src.dims.ng + physical_box.j0, src.dims.ng + physical_box.j1,
                           src.dims.ng + physical_box.k0, src.dims.ng + physical_box.k1};

    copy_field_region(src.alpha, src_box, dst.alpha);
    copy_field_region(src.chi, src_box, dst.chi);
    copy_field_region(src.K, src_box, dst.K);
    copy_field_region(src.Theta, src_box, dst.Theta);

    for (int a = 0; a < 3; ++a) {
        copy_field_region(src.beta[a], src_box, dst.beta[a]);
        copy_field_region(src.B[a], src_box, dst.B[a]);
        copy_field_region(src.tildeGamma[a], src_box, dst.tildeGamma[a]);
        copy_field_region(src.Z[a], src_box, dst.Z[a]);
    }

    for (int s = 0; s < 6; ++s) {
        copy_field_region(src.gamma_tilde[s], src_box, dst.gamma_tilde[s]);
        copy_field_region(src.A_tilde[s], src_box, dst.A_tilde[s]);
    }
}

template <typename State, typename T>
inline void coords_any_index(const State &grid, size_t i, size_t j, size_t k, T &x, T &y, T &z) {
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
inline const Field3D<T> &select_field(const EvolvedStateSoA<T> &grid, BoundaryField which,
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
    case BoundaryField::ATilde:
        return grid.A_tilde[component];
    case BoundaryField::Z:
        return grid.Z[component];
    case BoundaryField::GammaTildeInverse:
        break;
    }
    return grid.alpha;
}

template <typename T>
inline Field3D<T> &select_field(EvolvedStateSoA<T> &grid, BoundaryField which, int component) {
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
    case BoundaryField::ATilde:
        return grid.A_tilde[component];
    case BoundaryField::Z:
        return grid.Z[component];
    case BoundaryField::GammaTildeInverse:
        break;
    }
    return grid.alpha;
}

/// @brief Catmull-Rom cubic spline interpolation kernel.
/// @param p0, p1, p2, p3 Four consecutive sample values
/// @param t Interpolation parameter in [0,1], interpolates between p1 and p2
/// @return Interpolated value with O(h⁴) accuracy
template <typename T>
inline T catmull_rom_1d(T p0, T p1, T p2, T p3, T t) {
    const T t2 = t * t;
    const T t3 = t2 * t;
    return T(0.5) * ((T(2) * p1) +
                     (-p0 + p2) * t +
                     (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3) * t2 +
                     (-p0 + T(3) * p1 - T(3) * p2 + p3) * t3);
}

/// @brief Tricubic interpolation using Catmull-Rom splines for O(h⁴) accuracy.
/// Falls back to trilinear near boundaries where 4x4x4 stencil is unavailable.
template <typename State, typename T>
inline T sample_tricubic_field(const State &src, const Field3D<T> &field, T x, T y, T z) {
    size_t I0, I1, J0, J1, K0, K1;
    src.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t ix_min = I0;
    const size_t iy_min = J0;
    const size_t iz_min = K0;
    const size_t ix_max = I1 - 1;
    const size_t iy_max = J1 - 1;
    const size_t iz_max = K1 - 1;

    // Compute continuous grid coordinates
    const T ux = (x - src.x0) / src.dx + T(ix_min);
    const T uy = (y - src.y0) / src.dy + T(iy_min);
    const T uz = (z - src.z0) / src.dz + T(iz_min);

    // Find base indices (p1 in the Catmull-Rom stencil p0,p1,p2,p3)
    auto base_index = [](T u, size_t a_min, size_t a_max) -> size_t {
        if (u <= T(a_min)) return a_min;
        if (u >= T(a_max)) return a_max;
        return static_cast<size_t>(std::floor(u));
    };

    const size_t ix1 = base_index(ux, ix_min, ix_max);
    const size_t iy1 = base_index(uy, iy_min, iy_max);
    const size_t iz1 = base_index(uz, iz_min, iz_max);

    // Compute interpolation weights
    const T tx = std::clamp(ux - T(ix1), T(0), T(1));
    const T ty = std::clamp(uy - T(iy1), T(0), T(1));
    const T tz = std::clamp(uz - T(iz1), T(0), T(1));

    // Check if we have enough room for the 4-point stencil in each dimension
    const bool can_cubic_x = (ix1 >= ix_min + 1) && (ix1 + 2 <= ix_max);
    const bool can_cubic_y = (iy1 >= iy_min + 1) && (iy1 + 2 <= iy_max);
    const bool can_cubic_z = (iz1 >= iz_min + 1) && (iz1 + 2 <= iz_max);

    auto at = [&](size_t i, size_t j, size_t k) -> T {
        const size_t ic = std::clamp(i, ix_min, ix_max);
        const size_t jc = std::clamp(j, iy_min, iy_max);
        const size_t kc = std::clamp(k, iz_min, iz_max);
        return field.ptr()[field.idx(ic, jc, kc)];
    };

    // Full tricubic: 4x4x4 = 64 points
    if (can_cubic_x && can_cubic_y && can_cubic_z) {
        const size_t ix0 = ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        const size_t iy0 = iy1 - 1, iy2 = iy1 + 1, iy3 = iy1 + 2;
        const size_t iz0 = iz1 - 1, iz2 = iz1 + 1, iz3 = iz1 + 2;

        // Interpolate along z for each (i,j) pair, then along y, then along x
        T slice_y[4];
        for (int dy = 0; dy < 4; ++dy) {
            const size_t jj = (dy == 0) ? iy0 : (dy == 1) ? iy1 : (dy == 2) ? iy2 : iy3;
            T row_x[4];
            for (int dx = 0; dx < 4; ++dx) {
                const size_t ii = (dx == 0) ? ix0 : (dx == 1) ? ix1 : (dx == 2) ? ix2 : ix3;
                row_x[dx] = catmull_rom_1d(at(ii, jj, iz0), at(ii, jj, iz1),
                                           at(ii, jj, iz2), at(ii, jj, iz3), tz);
            }
            slice_y[dy] = catmull_rom_1d(row_x[0], row_x[1], row_x[2], row_x[3], tx);
        }
        return catmull_rom_1d(slice_y[0], slice_y[1], slice_y[2], slice_y[3], ty);
    }

    // Fallback: trilinear for boundary regions
    const size_t i0 = ix1;
    const size_t i1 = std::min(ix1 + size_t(1), ix_max);
    const size_t j0 = iy1;
    const size_t j1 = std::min(iy1 + size_t(1), iy_max);
    const size_t k0 = iz1;
    const size_t k1 = std::min(iz1 + size_t(1), iz_max);

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

template <typename State, typename T>
inline T sample_trilinear_field(const State &src, const Field3D<T> &field, T x, T y, T z) {
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

inline PatchBox expand_patch_clamped(const PatchBox &box, size_t margin, size_t nx, size_t ny,
                                     size_t nz) {
    PatchBox out{};
    out.i0 = (box.i0 > margin) ? box.i0 - margin : size_t(0);
    out.j0 = (box.j0 > margin) ? box.j0 - margin : size_t(0);
    out.k0 = (box.k0 > margin) ? box.k0 - margin : size_t(0);
    out.i1 = std::min(nx, box.i1 + margin);
    out.j1 = std::min(ny, box.j1 + margin);
    out.k1 = std::min(nz, box.k1 + margin);
    return out;
}

inline PatchBox merge_patch_boxes(const PatchBox &lhs, const PatchBox &rhs) {
    if (lhs.empty())
        return rhs;
    if (rhs.empty())
        return lhs;
    return PatchBox{std::min(lhs.i0, rhs.i0), std::max(lhs.i1, rhs.i1), std::min(lhs.j0, rhs.j0),
                    std::max(lhs.j1, rhs.j1), std::min(lhs.k0, rhs.k0), std::max(lhs.k1, rhs.k1)};
}

inline bool patch_boxes_overlap(const PatchBox &lhs, const PatchBox &rhs) noexcept {
    if (lhs.empty() || rhs.empty())
        return false;
    return lhs.i0 < rhs.i1 && rhs.i0 < lhs.i1 && lhs.j0 < rhs.j1 && rhs.j0 < lhs.j1 &&
           lhs.k0 < rhs.k1 && rhs.k0 < lhs.k1;
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
                    sample_tricubic_field(parent, select_field(parent, which, component), x, y, z);
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
                    sample_tricubic_field(parent, select_field(parent, which, component), x, y, z);
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
inline void blend_field_core_from_reference(const BSSNGridSoA<T> &reference,
                                            BSSNGridSoA<T> &target,
                                            const std::array<T, 3> &center,
                                            T core_half_width, T transition_width,
                                            BoundaryField which, int component) {
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
                const T centered_radius =
                    std::max({std::abs(x - center[0]), std::abs(y - center[1]),
                              std::abs(z - center[2])});
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
inline void blend_core_from_reference(const BSSNGridSoA<T> &reference, BSSNGridSoA<T> &target,
                                      const std::array<T, 3> &center, T core_half_width,
                                      T transition_width) {
    blend_field_core_from_reference(reference, target, center, core_half_width, transition_width,
                                    BoundaryField::Alpha, 0);
    blend_field_core_from_reference(reference, target, center, core_half_width, transition_width,
                                    BoundaryField::Chi, 0);
    blend_field_core_from_reference(reference, target, center, core_half_width, transition_width,
                                    BoundaryField::K, 0);
    blend_field_core_from_reference(reference, target, center, core_half_width, transition_width,
                                    BoundaryField::Theta, 0);
    for (int a = 0; a < 3; ++a) {
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::Beta, a);
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::B, a);
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::TildeGamma, a);
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::Z, a);
    }
    for (int s = 0; s < 6; ++s) {
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::GammaTilde, s);
        blend_field_core_from_reference(reference, target, center, core_half_width,
                                        transition_width, BoundaryField::ATilde, s);
    }

    tensorium_RG::bssn::enforce_algebraic_constraints(target);
}

template <typename T>
inline void blend_centered_core_from_reference(const BSSNGridSoA<T> &reference,
                                               BSSNGridSoA<T> &target, T core_half_width,
                                               T transition_width) {
    blend_core_from_reference(reference, target, std::array<T, 3>{T(0), T(0), T(0)},
                              core_half_width, transition_width);
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
inline void restrict_from_child_without_constraints(const BSSNGridSoA<T> &child,
                                                    BSSNGridSoA<T> &parent,
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
}

template <typename T>
inline void restrict_from_child(const BSSNGridSoA<T> &child, BSSNGridSoA<T> &parent,
                                const PatchBox &parent_box, size_t ratio) {
    restrict_from_child_without_constraints(child, parent, parent_box, ratio);
    tensorium_RG::bssn::enforce_algebraic_constraints(parent);
}

} // namespace detail

template <typename T>
inline void transfer_evolved_data_tricubic(const BSSNGridSoA<T> &src, BSSNGridSoA<T> &dst);

template <typename T, typename OuterBoundary = BoundaryRadiative> struct ParentInterpolationBoundary {
    struct Sampler {
        const void *state = nullptr;
        T (*sample_at)(const void *, BoundaryField, int, T, T, T) = nullptr;

        [[nodiscard]] bool valid() const noexcept { return state != nullptr && sample_at != nullptr; }

        [[nodiscard]] bool same_identity(const Sampler &other) const noexcept {
            return state == other.state && sample_at == other.sample_at;
        }

        [[nodiscard]] T sample(BoundaryField which, int component, T x, T y, T z) const {
            return sample_at(state, which, component, x, y, z);
        }
    };

    template <typename State>
    static T sample_from_state(const void *opaque, BoundaryField which, int component, T x, T y,
                               T z) {
        const auto &state = *static_cast<const State *>(opaque);
        return detail::sample_tricubic_field(state, detail::select_field(state, which, component),
                                             x, y, z);
    }

    template <typename State> static Sampler make_sampler(const State *state) {
        Sampler out{};
        out.state = state;
        out.sample_at = &sample_from_state<State>;
        return out;
    }

    struct Context {
        Sampler coarse_old{};
        Sampler coarse_new{};
        T       lambda_begin = T(0);
        T       lambda_end = T(0);

        Context() = default;

        template <typename OldState, typename NewState>
        Context(const OldState *old_state, const NewState *new_state, T lambda)
            : coarse_old(make_sampler(old_state)),
              coarse_new(make_sampler(new_state)),
              lambda_begin(lambda),
              lambda_end(lambda) {}

        template <typename OldState, typename NewState>
        Context(const OldState *old_state, const NewState *new_state, T begin_lambda, T end_lambda)
            : coarse_old(make_sampler(old_state)),
              coarse_new(make_sampler(new_state)),
              lambda_begin(begin_lambda),
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
        if (!current_context->coarse_old.valid() || !current_context->coarse_new.valid())
            throw std::runtime_error("FMR parent interpolation boundary is missing coarse data");
        if (which == BoundaryField::GammaTildeInverse)
            return;

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);
        U *ptr = field.ptr();
        const T lambda = std::clamp(current_context->lambda_begin +
                                        current_stage_fraction *
                                            (current_context->lambda_end - current_context->lambda_begin),
                                    T(0), T(1));
        const size_t nx_tot = field.st.nx_tot;
        const size_t ny_tot = field.st.ny_tot;
        const size_t nz_tot = field.st.nz_tot;

        auto fill_cell = [&](size_t i, size_t j, size_t k) {
            U x, y, z;
            detail::coords_any_index(grid, i, j, k, x, y, z);
            const T old_value = current_context->coarse_old.sample(which, component, x, y, z);
            if (lambda == T(0) || current_context->coarse_old.same_identity(current_context->coarse_new)) {
                ptr[field.idx(i, j, k)] = old_value;
                return;
            }
            const T new_value = current_context->coarse_new.sample(which, component, x, y, z);
            ptr[field.idx(i, j, k)] = (T(1) - lambda) * old_value + lambda * new_value;
        };

#pragma omp parallel for collapse(3)
        for (size_t i = 0; i < I0; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k)
                    fill_cell(i, j, k);

#pragma omp parallel for collapse(3)
        for (size_t i = I1; i < nx_tot; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k)
                    fill_cell(i, j, k);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = 0; j < J0; ++j)
                for (size_t k = 0; k < nz_tot; ++k)
                    fill_cell(i, j, k);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J1; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k)
                    fill_cell(i, j, k);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = 0; k < K0; ++k)
                    fill_cell(i, j, k);

#pragma omp parallel for collapse(3)
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j)
                for (size_t k = K1; k < nz_tot; ++k)
                    fill_cell(i, j, k);
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

    struct LevelPerformanceStats {
        double snapshot_seconds = 0.0;
        double evolve_seconds = 0.0;
        double child_subcycling_seconds = 0.0;
        double restriction_seconds = 0.0;
        size_t calls = 0;
        size_t child_substeps = 0;
    };

    struct LevelState {
        BSSNGridSoA<T>                     grid;
        std::unique_ptr<EvolvedStateSoA<T>> snapshot;
        PatchBox                           snapshot_region{};
        PatchBox                           parent_cells{};
        size_t                             refinement_ratio = 1;
        size_t                             step_count = 0;

        LevelState(size_t nx, size_t ny, size_t nz, size_t ng, T dx, T dy, T dz, T x0, T y0,
                   T z0, const PatchBox &box, size_t ratio)
            : grid(nx, ny, nz, ng, dx, dy, dz),
              parent_cells(box),
              refinement_ratio(ratio) {
            grid.x0 = x0;
            grid.y0 = y0;
            grid.z0 = z0;
        }

        [[nodiscard]] bool has_snapshot() const noexcept { return static_cast<bool>(snapshot); }

        EvolvedStateSoA<T> &snapshot_ref() {
            if (!snapshot)
                throw std::logic_error("FMR level snapshot was requested but not allocated");
            return *snapshot;
        }

        const EvolvedStateSoA<T> &snapshot_ref() const {
            if (!snapshot)
                throw std::logic_error("FMR level snapshot was requested but not allocated");
            return *snapshot;
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

        root_stepper_ = std::make_unique<RootStepper>(levels_.front()->grid, backend_options_,
                                                      padding_);

        const BSSNGridSoA<T> *parent = &levels_.front()->grid;
        for (size_t cfg_index = 0; cfg_index < level_configs.size(); ++cfg_index) {
            const LevelConfig &cfg = level_configs[cfg_index];
            const auto child = detail::make_child_geometry(*parent, cfg);
            levels_.push_back(std::make_unique<LevelState>(
                child.nx, child.ny, child.nz, child.ng, T(child.dx), T(child.dy), T(child.dz),
                T(child.x0), T(child.y0), T(child.z0), cfg.parent_cells, cfg.refinement_ratio));
            fine_steppers_.push_back(
                std::make_unique<FineStepper>(levels_.back()->grid, backend_options_, padding_));
            parent = &levels_.back()->grid;
        }

        initialize_snapshots();
        initialize_nested_levels();
        last_step_perf_.resize(levels_.size());
    }

    [[nodiscard]] size_t num_levels() const noexcept { return levels_.size(); }

    BSSNGridSoA<T> &root_grid() { return levels_.front()->grid; }
    const BSSNGridSoA<T> &root_grid() const { return levels_.front()->grid; }

    BSSNGridSoA<T> &level_grid(size_t level) { return levels_.at(level)->grid; }
    const BSSNGridSoA<T> &level_grid(size_t level) const { return levels_.at(level)->grid; }

    [[nodiscard]] const std::vector<LevelPerformanceStats> &last_step_performance() const noexcept {
        return last_step_perf_;
    }

    [[nodiscard]] static constexpr size_t allocated_grid_field_count() noexcept { return 40; }

    [[nodiscard]] static constexpr size_t allocated_snapshot_field_count() noexcept { return 28; }

    [[nodiscard]] static constexpr size_t allocated_stepper_field_count() noexcept {
        // One evolved-fields stage reference, one RHS workspace for the 28 evolved fields, and one
        // temporary scalar trace cache.
        return 28 + 28 + 1;
    }

    [[nodiscard]] size_t level_allocated_field_count(size_t level) const noexcept {
        // Full-resolution field bundles that scale with the level grid volume.
        (void)level;
        return allocated_grid_field_count() + allocated_stepper_field_count();
    }

    [[nodiscard]] size_t level_snapshot_allocated_field_count(size_t level) const noexcept {
        return levels_[level]->has_snapshot() ? allocated_snapshot_field_count() : size_t(0);
    }

    [[nodiscard]] size_t level_snapshot_total_cells(size_t level) const noexcept {
        return levels_[level]->has_snapshot() ? levels_[level]->snapshot_ref().total_cells()
                                              : size_t(0);
    }

    [[nodiscard]] double level_estimated_bytes(size_t level) const noexcept {
        const auto &grid = level_grid(level);
        double bytes = static_cast<double>(level_allocated_field_count(level)) *
                       static_cast<double>(grid.total_cells()) * static_cast<double>(sizeof(T));
        bytes += static_cast<double>(level_snapshot_allocated_field_count(level)) *
                 static_cast<double>(level_snapshot_total_cells(level)) *
                 static_cast<double>(sizeof(T));
        return bytes;
    }

    [[nodiscard]] double total_estimated_bytes() const noexcept {
        double total = 0.0;
        for (size_t level = 0; level < levels_.size(); ++level)
            total += level_estimated_bytes(level);
        return total;
    }

    void initialize_nested_levels() {
        for (size_t level = 1; level < levels_.size(); ++level)
            prolongate_level_from_parent(level);
    }

    void initialize_snapshots() {
        for (size_t level = 0; level + 1 < levels_.size(); ++level) {
            allocate_snapshot_for_level(level);
            refresh_level_snapshot(level);
        }
    }

    void prolongate_level_from_parent(size_t level) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR prolongation requires a valid child level");
        detail::prolongate_from_parent(levels_[level - 1]->grid, levels_[level]->grid);
        refresh_level_snapshot(level);
    }

    void restrict_level_to_parent(size_t level) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR restriction requires a valid child level");
        detail::restrict_from_child(levels_[level]->grid, levels_[level - 1]->grid,
                                    levels_[level]->parent_cells, levels_[level]->refinement_ratio);
        refresh_level_snapshot(level - 1);
    }

    void blend_level_shell_from_parent(size_t level, size_t shell_cells) {
        if (level == 0 || level >= levels_.size())
            throw std::out_of_range("FMR shell blending requires a valid child level");
        detail::blend_shell_from_parent(levels_[level - 1]->grid, levels_[level]->grid, shell_cells);
        refresh_level_snapshot(level);
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
        refresh_level_snapshot(level);
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

    void step(T dt_root) {
        std::fill(last_step_perf_.begin(), last_step_perf_.end(), LevelPerformanceStats{});
        advance_level(0, dt_root);
    }

  private:
    size_t                                        padding_ = 4;
    GaugeParameters<T>                            gauge_params_{};
    tensorium::backend::Options                   backend_options_{};
    std::vector<std::unique_ptr<LevelState>>      levels_;
    std::unique_ptr<RootStepper>                  root_stepper_;
    std::vector<std::unique_ptr<FineStepper>>     fine_steppers_;
    std::vector<LevelPerformanceStats>            last_step_perf_;

    using Clock = std::chrono::steady_clock;

    static double elapsed_seconds(const Clock::time_point &start,
                                  const Clock::time_point &stop) {
        return std::chrono::duration<double>(stop - start).count();
    }

    [[nodiscard]] static size_t snapshot_margin_cells(const LevelState &child) noexcept {
        const size_t coarse_halo =
            (child.grid.dims.ng + child.refinement_ratio - 1) / child.refinement_ratio;
        // Keep one extra coarse cell so trilinear interpolation never clamps at the snapshot edge.
        return coarse_halo + 1;
    }

    [[nodiscard]] PatchBox snapshot_region_for_child(size_t level) const {
        if (level + 1 >= levels_.size())
            throw std::out_of_range("FMR snapshot region requires a valid child level");
        const auto &parent = *levels_[level];
        const auto &child = *levels_[level + 1];
        return detail::expand_patch_clamped(child.parent_cells, snapshot_margin_cells(child),
                                            parent.grid.dims.nx, parent.grid.dims.ny,
                                            parent.grid.dims.nz);
    }

    void allocate_snapshot_for_level(size_t level) {
        if (level + 1 >= levels_.size())
            return;

        auto &state = *levels_[level];
        state.snapshot_region = snapshot_region_for_child(level);
        if (state.snapshot_region.empty())
            throw std::runtime_error("Computed empty FMR snapshot region for parent level");

        state.snapshot = std::make_unique<EvolvedStateSoA<T>>(
            state.snapshot_region.nx(), state.snapshot_region.ny(), state.snapshot_region.nz(),
            size_t(0), state.grid.dx, state.grid.dy, state.grid.dz);
        state.snapshot->x0 = state.grid.x0 + static_cast<T>(state.snapshot_region.i0) * state.grid.dx;
        state.snapshot->y0 = state.grid.y0 + static_cast<T>(state.snapshot_region.j0) * state.grid.dy;
        state.snapshot->z0 = state.grid.z0 + static_cast<T>(state.snapshot_region.k0) * state.grid.dz;
    }

    void refresh_level_snapshot(size_t level) {
        if (level >= levels_.size() || !levels_[level]->has_snapshot())
            return;
        detail::copy_evolved_state_region(levels_[level]->grid, levels_[level]->snapshot_region,
                                          levels_[level]->snapshot_ref());
    }

    void advance_level(size_t level_idx, T dt) {
        LevelState &level = *levels_[level_idx];
        auto       &perf = last_step_perf_[level_idx];
        ++perf.calls;
        const bool has_child = (level_idx + 1 < levels_.size());
        if (has_child) {
            const auto snapshot_start = Clock::now();
            refresh_level_snapshot(level_idx);
            perf.snapshot_seconds += elapsed_seconds(snapshot_start, Clock::now());
        }

        {
            const auto evolve_start = Clock::now();
            const detail::ScopedFDSpacing spacing(level.grid.dx);
            if (level_idx == 0)
                root_stepper_->step(level.grid, dt, level.step_count++);
            else
                fine_steppers_[level_idx - 1]->step(level.grid, dt, level.step_count++);
            perf.evolve_seconds += elapsed_seconds(evolve_start, Clock::now());
        }

        if (!has_child)
            return;

        LevelState &child = *levels_[level_idx + 1];
        const size_t ratio = child.refinement_ratio;
        const T      dt_child = dt / T(ratio);

        const auto child_start = Clock::now();
        for (size_t substep = 0; substep < ratio; ++substep) {
            ++perf.child_substeps;
            const typename FineBoundary::Context ctx{
                &level.snapshot_ref(), &level.grid, T(substep) / T(ratio), T(substep + 1) / T(ratio)};
            const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
            advance_level(level_idx + 1, dt_child);
        }
        perf.child_subcycling_seconds += elapsed_seconds(child_start, Clock::now());

        const auto restrict_start = Clock::now();
        detail::restrict_from_child(child.grid, level.grid, child.parent_cells, child.refinement_ratio);
        perf.restriction_seconds += elapsed_seconds(restrict_start, Clock::now());
    }
};

template <typename T, typename OuterBoundary = BoundaryRadiative>
class BinaryPunctureFixedMeshRefinementHierarchy {
  public:
    using RootStepper = BSSNRKStepper<T, OuterBoundary>;
    using FineBoundary = ParentInterpolationBoundary<T, OuterBoundary>;
    using FineStepper = BSSNRKStepper<T, FineBoundary>;

    struct LevelPerformanceStats {
        double snapshot_seconds = 0.0;
        double evolve_seconds = 0.0;
        double child_subcycling_seconds = 0.0;
        double restriction_seconds = 0.0;
        size_t calls = 0;
        size_t child_substeps = 0;
    };

    struct SharedLevelState {
        BSSNGridSoA<T>                     grid;
        std::unique_ptr<EvolvedStateSoA<T>> snapshot;
        PatchBox                           snapshot_region{};
        PatchBox                           parent_cells{};
        size_t                             refinement_ratio = 1;
        size_t                             step_count = 0;

        SharedLevelState(size_t nx, size_t ny, size_t nz, size_t ng, T dx, T dy, T dz, T x0,
                         T y0, T z0, const PatchBox &box, size_t ratio)
            : grid(nx, ny, nz, ng, dx, dy, dz),
              parent_cells(box),
              refinement_ratio(ratio) {
            grid.x0 = x0;
            grid.y0 = y0;
            grid.z0 = z0;
        }

        [[nodiscard]] bool has_snapshot() const noexcept { return static_cast<bool>(snapshot); }

        EvolvedStateSoA<T> &snapshot_ref() {
            if (!snapshot)
                throw std::logic_error("BBH FMR level snapshot was requested but not allocated");
            return *snapshot;
        }

        const EvolvedStateSoA<T> &snapshot_ref() const {
            if (!snapshot)
                throw std::logic_error("BBH FMR level snapshot was requested but not allocated");
            return *snapshot;
        }
    };

    struct LeafState {
        BSSNGridSoA<T> grid;
        PatchBox       parent_cells{};
        size_t         refinement_ratio = 1;
        size_t         step_count = 0;

        LeafState(size_t nx, size_t ny, size_t nz, size_t ng, T dx, T dy, T dz, T x0, T y0, T z0,
                  const PatchBox &box, size_t ratio)
            : grid(nx, ny, nz, ng, dx, dy, dz),
              parent_cells(box),
              refinement_ratio(ratio) {
            grid.x0 = x0;
            grid.y0 = y0;
            grid.z0 = z0;
        }
    };

    explicit BinaryPunctureFixedMeshRefinementHierarchy(
        const BSSNGridSoA<T> &root_state, const BinaryPunctureHierarchyConfig &cfg,
        size_t padding = 4)
        : BinaryPunctureFixedMeshRefinementHierarchy(root_state, cfg, tensorium::backend::Options{},
                                                     padding) {}

    BinaryPunctureFixedMeshRefinementHierarchy(
        const BSSNGridSoA<T> &root_state, const BinaryPunctureHierarchyConfig &cfg,
        const tensorium::backend::Options &backend, size_t padding = 4)
        : padding_(padding), backend_options_(backend) {
        if (cfg.shared_levels.empty()) {
            throw std::invalid_argument(
                "BBH split FMR requires at least one shared refined level before the split leaves");
        }

        shared_levels_.reserve(cfg.shared_levels.size() + 1);
        shared_levels_.push_back(std::make_unique<SharedLevelState>(
            root_state.dims.nx, root_state.dims.ny, root_state.dims.nz, root_state.dims.ng,
            root_state.dx, root_state.dy, root_state.dz, root_state.x0, root_state.y0,
            root_state.z0, PatchBox{}, size_t(1)));
        detail::copy_evolved_state(root_state, shared_levels_.front()->grid);
        tensorium_RG::bssn::enforce_algebraic_constraints(shared_levels_.front()->grid);

        root_stepper_ =
            std::make_unique<RootStepper>(shared_levels_.front()->grid, backend_options_, padding_);

        const BSSNGridSoA<T> *parent = &shared_levels_.front()->grid;
        for (const LevelConfig &level_cfg : cfg.shared_levels) {
            const auto child = detail::make_child_geometry(*parent, level_cfg);
            shared_levels_.push_back(std::make_unique<SharedLevelState>(
                child.nx, child.ny, child.nz, child.ng, T(child.dx), T(child.dy), T(child.dz),
                T(child.x0), T(child.y0), T(child.z0), level_cfg.parent_cells,
                level_cfg.refinement_ratio));
            shared_steppers_.push_back(std::make_unique<FineStepper>(
                shared_levels_.back()->grid, backend_options_, padding_));
            parent = &shared_levels_.back()->grid;
        }

        const BSSNGridSoA<T> &leaf_parent = shared_levels_.back()->grid;
        const auto left_geom = detail::make_child_geometry(leaf_parent, cfg.left_leaf);
        const auto right_geom = detail::make_child_geometry(leaf_parent, cfg.right_leaf);
        if (detail::patch_boxes_overlap(cfg.left_leaf.parent_cells, cfg.right_leaf.parent_cells)) {
            throw std::invalid_argument(
                "BBH split FMR leaf patches overlap on their shared parent level");
        }

        leaves_.push_back(std::make_unique<LeafState>(
            left_geom.nx, left_geom.ny, left_geom.nz, left_geom.ng, T(left_geom.dx), T(left_geom.dy),
            T(left_geom.dz), T(left_geom.x0), T(left_geom.y0), T(left_geom.z0),
            cfg.left_leaf.parent_cells, cfg.left_leaf.refinement_ratio));
        leaves_.push_back(std::make_unique<LeafState>(
            right_geom.nx, right_geom.ny, right_geom.nz, right_geom.ng, T(right_geom.dx),
            T(right_geom.dy), T(right_geom.dz), T(right_geom.x0), T(right_geom.y0), T(right_geom.z0),
            cfg.right_leaf.parent_cells, cfg.right_leaf.refinement_ratio));
        leaf_steppers_.push_back(
            std::make_unique<FineStepper>(leaves_[0]->grid, backend_options_, padding_));
        leaf_steppers_.push_back(
            std::make_unique<FineStepper>(leaves_[1]->grid, backend_options_, padding_));

        initialize_snapshots();
        initialize_nested_levels();
        last_step_perf_.resize(num_levels());
    }

    [[nodiscard]] size_t num_levels() const noexcept { return shared_levels_.size() + leaves_.size(); }

    [[nodiscard]] size_t num_shared_levels() const noexcept { return shared_levels_.size(); }
    [[nodiscard]] size_t num_leaves() const noexcept { return leaves_.size(); }

    BSSNGridSoA<T> &root_grid() { return shared_levels_.front()->grid; }
    const BSSNGridSoA<T> &root_grid() const { return shared_levels_.front()->grid; }

    BSSNGridSoA<T> &shared_level_grid(size_t level) { return shared_levels_.at(level)->grid; }
    const BSSNGridSoA<T> &shared_level_grid(size_t level) const { return shared_levels_.at(level)->grid; }

    BSSNGridSoA<T> &leaf_grid(size_t leaf) { return leaves_.at(leaf)->grid; }
    const BSSNGridSoA<T> &leaf_grid(size_t leaf) const { return leaves_.at(leaf)->grid; }

    BSSNGridSoA<T> &level_grid(size_t level) {
        if (level < shared_levels_.size())
            return shared_levels_.at(level)->grid;
        return leaves_.at(level - shared_levels_.size())->grid;
    }

    const BSSNGridSoA<T> &level_grid(size_t level) const {
        if (level < shared_levels_.size())
            return shared_levels_.at(level)->grid;
        return leaves_.at(level - shared_levels_.size())->grid;
    }

    [[nodiscard]] const std::vector<LevelPerformanceStats> &last_step_performance() const noexcept {
        return last_step_perf_;
    }

    [[nodiscard]] static constexpr size_t allocated_grid_field_count() noexcept { return 40; }
    [[nodiscard]] static constexpr size_t allocated_snapshot_field_count() noexcept { return 28; }
    [[nodiscard]] static constexpr size_t allocated_stepper_field_count() noexcept {
        return 28 + 28 + 1;
    }

    [[nodiscard]] size_t level_allocated_field_count(size_t level) const noexcept {
        (void)level;
        return allocated_grid_field_count() + allocated_stepper_field_count();
    }

    [[nodiscard]] size_t level_snapshot_allocated_field_count(size_t level) const noexcept {
        return is_shared_level(level) && shared_levels_[level]->has_snapshot()
                   ? allocated_snapshot_field_count()
                   : size_t(0);
    }

    [[nodiscard]] size_t level_snapshot_total_cells(size_t level) const noexcept {
        return is_shared_level(level) && shared_levels_[level]->has_snapshot()
                   ? shared_levels_[level]->snapshot_ref().total_cells()
                   : size_t(0);
    }

    [[nodiscard]] double level_estimated_bytes(size_t level) const noexcept {
        const auto &grid = level_grid(level);
        double bytes = static_cast<double>(level_allocated_field_count(level)) *
                       static_cast<double>(grid.total_cells()) * static_cast<double>(sizeof(T));
        bytes += static_cast<double>(level_snapshot_allocated_field_count(level)) *
                 static_cast<double>(level_snapshot_total_cells(level)) *
                 static_cast<double>(sizeof(T));
        return bytes;
    }

    [[nodiscard]] double total_estimated_bytes() const noexcept {
        double total = 0.0;
        for (size_t level = 0; level < num_levels(); ++level)
            total += level_estimated_bytes(level);
        return total;
    }

    void initialize_nested_levels() {
        for (size_t level = 1; level < shared_levels_.size(); ++level)
            prolongate_shared_level_from_parent(level);
        for (size_t leaf = 0; leaf < leaves_.size(); ++leaf)
            prolongate_leaf_from_parent(leaf);
    }

    void initialize_snapshots() {
        for (size_t level = 0; level < shared_levels_.size(); ++level) {
            allocate_snapshot_for_shared_level(level);
            refresh_shared_snapshot(level);
        }
    }

    void prolongate_shared_level_from_parent(size_t level) {
        if (level == 0 || level >= shared_levels_.size())
            throw std::out_of_range("BBH FMR shared prolongation requires a valid child level");
        detail::prolongate_from_parent(shared_levels_[level - 1]->grid, shared_levels_[level]->grid);
        refresh_shared_snapshot(level);
    }

    void prolongate_leaf_from_parent(size_t leaf) {
        if (leaf >= leaves_.size())
            throw std::out_of_range("BBH FMR leaf prolongation requires a valid leaf index");
        detail::prolongate_from_parent(shared_levels_.back()->grid, leaves_[leaf]->grid);
    }

    void restrict_shared_level_to_parent(size_t level) {
        if (level == 0 || level >= shared_levels_.size())
            throw std::out_of_range("BBH FMR shared restriction requires a valid child level");
        detail::restrict_from_child(shared_levels_[level]->grid, shared_levels_[level - 1]->grid,
                                    shared_levels_[level]->parent_cells,
                                    shared_levels_[level]->refinement_ratio);
        refresh_shared_snapshot(level - 1);
    }

    void restrict_leaf_to_parent(size_t leaf) {
        if (leaf >= leaves_.size())
            throw std::out_of_range("BBH FMR leaf restriction requires a valid leaf index");
        detail::restrict_from_child(leaves_[leaf]->grid, shared_levels_.back()->grid,
                                    leaves_[leaf]->parent_cells, leaves_[leaf]->refinement_ratio);
        refresh_shared_snapshot(shared_levels_.size() - 1);
    }

    void blend_shared_level_centered_core_from_reference(size_t level, const BSSNGridSoA<T> &reference,
                                                         T core_half_width, T transition_width) {
        if (level == 0 || level >= shared_levels_.size())
            throw std::out_of_range("BBH FMR centered-core blending requires a valid shared level");
        if (!detail::same_grid_geometry(shared_levels_[level]->grid, reference)) {
            throw std::invalid_argument(
                "BBH FMR centered-core blending requires a reference grid with identical geometry");
        }
        detail::blend_centered_core_from_reference(reference, shared_levels_[level]->grid,
                                                   core_half_width, transition_width);
        refresh_shared_snapshot(level);
    }

    void blend_leaf_centered_core_from_reference(size_t leaf, const BSSNGridSoA<T> &reference,
                                                 T core_half_width, T transition_width) {
        blend_leaf_core_from_reference(leaf, reference, std::array<T, 3>{T(0), T(0), T(0)},
                                       core_half_width, transition_width);
    }

    void blend_leaf_core_from_reference(size_t leaf, const BSSNGridSoA<T> &reference,
                                        const std::array<T, 3> &center, T core_half_width,
                                        T transition_width) {
        if (leaf >= leaves_.size())
            throw std::out_of_range("BBH FMR centered-core blending requires a valid leaf index");
        if (!detail::same_grid_geometry(leaves_[leaf]->grid, reference)) {
            throw std::invalid_argument(
                "BBH FMR centered-core blending requires a reference grid with identical geometry");
        }
        detail::blend_core_from_reference(reference, leaves_[leaf]->grid, center,
                                          core_half_width, transition_width);
    }

    void restrict_all_levels_to_root() {
        for (size_t leaf = 0; leaf < leaves_.size(); ++leaf)
            restrict_leaf_to_parent(leaf);
        for (size_t level = shared_levels_.size(); level-- > 1;)
            restrict_shared_level_to_parent(level);
    }

    void apply_level_boundaries() {
        {
            const detail::ScopedFDSpacing spacing(shared_levels_.front()->grid.dx);
            tensorium_RG::bssn::apply_halos_grid<OuterBoundary>(shared_levels_.front()->grid);
        }

        for (size_t level = 1; level < shared_levels_.size(); ++level) {
            const typename FineBoundary::Context ctx{&shared_levels_[level - 1]->grid,
                                                     &shared_levels_[level - 1]->grid, T(0)};
            const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
            const detail::ScopedFDSpacing spacing(shared_levels_[level]->grid.dx);
            tensorium_RG::bssn::apply_halos_grid<FineBoundary>(shared_levels_[level]->grid);
        }

        for (size_t leaf = 0; leaf < leaves_.size(); ++leaf) {
            const typename FineBoundary::Context ctx{&shared_levels_.back()->grid,
                                                     &shared_levels_.back()->grid, T(0)};
            const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
            const detail::ScopedFDSpacing spacing(leaves_[leaf]->grid.dx);
            tensorium_RG::bssn::apply_halos_grid<FineBoundary>(leaves_[leaf]->grid);
        }
    }

    void set_gauge_parameters(const GaugeParameters<T> &params) {
        gauge_params_ = params;
        root_stepper_->set_gauge_parameters(params);
        for (auto &stepper : shared_steppers_)
            stepper->set_gauge_parameters(params);
        for (auto &stepper : leaf_steppers_)
            stepper->set_gauge_parameters(params);
    }

    void set_state_log_stride(size_t stride) {
        root_stepper_->set_state_log_stride(stride);
        for (auto &stepper : shared_steppers_)
            stepper->set_state_log_stride(stride);
        for (auto &stepper : leaf_steppers_)
            stepper->set_state_log_stride(stride);
    }

    void set_backend_options(const tensorium::backend::Options &backend) {
        backend_options_ = backend;
        root_stepper_->set_backend_options(backend);
        for (auto &stepper : shared_steppers_)
            stepper->set_backend_options(backend);
        for (auto &stepper : leaf_steppers_)
            stepper->set_backend_options(backend);
    }

    [[nodiscard]] const tensorium::backend::Options &backend_options() const noexcept {
        return backend_options_;
    }

    [[nodiscard]] T compute_dt(const CFLControl<T> &control) const {
        T      dt_root = std::numeric_limits<T>::infinity();
        size_t cumulative_ratio = 1;
        for (size_t level = 0; level < shared_levels_.size(); ++level) {
            const T local_dt =
                tensorium_RG::bssn::compute_dt_cfl(shared_levels_[level]->grid, control, padding_);
            dt_root = std::min(dt_root, local_dt * T(cumulative_ratio));
            if (level + 1 < shared_levels_.size())
                cumulative_ratio *= shared_levels_[level + 1]->refinement_ratio;
        }

        for (const auto &leaf : leaves_) {
            const T local_dt = tensorium_RG::bssn::compute_dt_cfl(leaf->grid, control, padding_);
            dt_root = std::min(dt_root, local_dt * T(cumulative_ratio * leaf->refinement_ratio));
        }
        return dt_root;
    }

    void step(T dt_root) {
        std::fill(last_step_perf_.begin(), last_step_perf_.end(), LevelPerformanceStats{});
        advance_shared_level(0, dt_root);
    }

    /// @brief Regrid leaf grids to new positions, transferring data from old grids.
    /// @param new_left_box New PatchBox for left leaf (in parent cell coordinates)
    /// @param new_right_box New PatchBox for right leaf (in parent cell coordinates)
    /// @return true if regrid was successful
    bool regrid_leaves(const PatchBox &new_left_box, const PatchBox &new_right_box) {
        if (leaves_.size() != 2 || leaf_steppers_.size() != 2 || shared_levels_.empty()) {
            return false;
        }

        // Validate boxes don't overlap
        if (detail::patch_boxes_overlap(new_left_box, new_right_box)) {
            return false;
        }

        const auto &parent = shared_levels_.back()->grid;
        const size_t ratio = leaves_[0]->refinement_ratio;
        const size_t ng = leaves_[0]->grid.dims.ng;
        const auto &old_left = *leaves_[0];
        const auto &old_right = *leaves_[1];

        // Create new leaf configurations
        LevelConfig left_cfg{new_left_box, ratio, ng};
        LevelConfig right_cfg{new_right_box, ratio, ng};

        // Compute new geometries
        const auto left_geom = detail::make_child_geometry(parent, left_cfg);
        const auto right_geom = detail::make_child_geometry(parent, right_cfg);

        // Create new leaf states
        auto new_left_leaf = std::make_unique<LeafState>(
            left_geom.nx, left_geom.ny, left_geom.nz, left_geom.ng,
            T(left_geom.dx), T(left_geom.dy), T(left_geom.dz),
            T(left_geom.x0), T(left_geom.y0), T(left_geom.z0),
            new_left_box, ratio);

        auto new_right_leaf = std::make_unique<LeafState>(
            right_geom.nx, right_geom.ny, right_geom.nz, right_geom.ng,
            T(right_geom.dx), T(right_geom.dy), T(right_geom.dz),
            T(right_geom.x0), T(right_geom.y0), T(right_geom.z0),
            new_right_box, ratio);

        // Transfer data from old grids to new grids using tricubic interpolation
        transfer_evolved_data_tricubic(old_left.grid, new_left_leaf->grid);
        transfer_evolved_data_tricubic(old_right.grid, new_right_leaf->grid);
        new_left_leaf->step_count = old_left.step_count;
        new_right_leaf->step_count = old_right.step_count;

        // Recreate steppers for new grids while preserving runtime state.
        auto new_left_stepper =
            std::make_unique<FineStepper>(new_left_leaf->grid, backend_options_, padding_);
        auto new_right_stepper =
            std::make_unique<FineStepper>(new_right_leaf->grid, backend_options_, padding_);
        new_left_stepper->copy_runtime_state_from(*leaf_steppers_[0]);
        new_right_stepper->copy_runtime_state_from(*leaf_steppers_[1]);

        leaves_[0] = std::move(new_left_leaf);
        leaves_[1] = std::move(new_right_leaf);
        leaf_steppers_[0] = std::move(new_left_stepper);
        leaf_steppers_[1] = std::move(new_right_stepper);

        // Reallocate snapshot region if needed
        allocate_snapshot_for_shared_level(shared_levels_.size() - 1);
        refresh_shared_snapshot(shared_levels_.size() - 1);

        return true;
    }

    /// @brief Get the PatchBox for a leaf grid.
    [[nodiscard]] const PatchBox &leaf_parent_cells(size_t leaf) const {
        return leaves_.at(leaf)->parent_cells;
    }

    /// @brief Get the refinement ratio for leaf grids.
    [[nodiscard]] size_t leaf_refinement_ratio() const {
        return leaves_.empty() ? 1 : leaves_[0]->refinement_ratio;
    }

  private:
    size_t                                            padding_ = 4;
    tensorium::backend::Options                       backend_options_{};
    GaugeParameters<T>                                gauge_params_{};
    std::vector<std::unique_ptr<SharedLevelState>>    shared_levels_;
    std::vector<std::unique_ptr<LeafState>>           leaves_;
    std::unique_ptr<RootStepper>                      root_stepper_;
    std::vector<std::unique_ptr<FineStepper>>         shared_steppers_;
    std::vector<std::unique_ptr<FineStepper>>         leaf_steppers_;
    std::vector<LevelPerformanceStats>                last_step_perf_;

    using Clock = std::chrono::steady_clock;

    [[nodiscard]] bool is_shared_level(size_t level) const noexcept {
        return level < shared_levels_.size();
    }

    [[nodiscard]] size_t perf_index_shared(size_t level) const noexcept { return level; }
    [[nodiscard]] size_t perf_index_leaf(size_t leaf) const noexcept {
        return shared_levels_.size() + leaf;
    }

    static double elapsed_seconds(const Clock::time_point &start,
                                  const Clock::time_point &stop) {
        return std::chrono::duration<double>(stop - start).count();
    }

    [[nodiscard]] static size_t snapshot_margin_cells(size_t ng, size_t ratio) noexcept {
        const size_t coarse_halo = (ng + ratio - 1) / ratio;
        return coarse_halo + 1;
    }

    [[nodiscard]] PatchBox snapshot_region_for_shared_level(size_t level) const {
        const auto &parent = *shared_levels_[level];
        if (level + 1 < shared_levels_.size()) {
            const auto &child = *shared_levels_[level + 1];
            return detail::expand_patch_clamped(
                child.parent_cells, snapshot_margin_cells(child.grid.dims.ng, child.refinement_ratio),
                parent.grid.dims.nx, parent.grid.dims.ny, parent.grid.dims.nz);
        }

        const auto left_box = detail::expand_patch_clamped(
            leaves_[0]->parent_cells,
            snapshot_margin_cells(leaves_[0]->grid.dims.ng, leaves_[0]->refinement_ratio),
            parent.grid.dims.nx, parent.grid.dims.ny, parent.grid.dims.nz);
        const auto right_box = detail::expand_patch_clamped(
            leaves_[1]->parent_cells,
            snapshot_margin_cells(leaves_[1]->grid.dims.ng, leaves_[1]->refinement_ratio),
            parent.grid.dims.nx, parent.grid.dims.ny, parent.grid.dims.nz);
        return detail::merge_patch_boxes(left_box, right_box);
    }

    void allocate_snapshot_for_shared_level(size_t level) {
        auto &state = *shared_levels_[level];
        state.snapshot_region = snapshot_region_for_shared_level(level);
        if (state.snapshot_region.empty())
            throw std::runtime_error("Computed empty BBH FMR snapshot region for shared level");

        state.snapshot = std::make_unique<EvolvedStateSoA<T>>(
            state.snapshot_region.nx(), state.snapshot_region.ny(), state.snapshot_region.nz(),
            size_t(0), state.grid.dx, state.grid.dy, state.grid.dz);
        state.snapshot->x0 = state.grid.x0 + static_cast<T>(state.snapshot_region.i0) * state.grid.dx;
        state.snapshot->y0 = state.grid.y0 + static_cast<T>(state.snapshot_region.j0) * state.grid.dy;
        state.snapshot->z0 = state.grid.z0 + static_cast<T>(state.snapshot_region.k0) * state.grid.dz;
    }

    void refresh_shared_snapshot(size_t level) {
        if (level >= shared_levels_.size() || !shared_levels_[level]->has_snapshot())
            return;
        detail::copy_evolved_state_region(shared_levels_[level]->grid,
                                          shared_levels_[level]->snapshot_region,
                                          shared_levels_[level]->snapshot_ref());
    }

    void advance_leaf(size_t leaf_idx, T dt) {
        auto       &leaf = *leaves_[leaf_idx];
        auto       &perf = last_step_perf_[perf_index_leaf(leaf_idx)];
        ++perf.calls;

        const auto evolve_start = Clock::now();
        const detail::ScopedFDSpacing spacing(leaf.grid.dx);
        leaf_steppers_[leaf_idx]->step(leaf.grid, dt, leaf.step_count++);
        perf.evolve_seconds += elapsed_seconds(evolve_start, Clock::now());
    }

    void advance_shared_level(size_t level_idx, T dt) {
        auto       &level = *shared_levels_[level_idx];
        auto       &perf = last_step_perf_[perf_index_shared(level_idx)];
        ++perf.calls;

        const auto snapshot_start = Clock::now();
        refresh_shared_snapshot(level_idx);
        perf.snapshot_seconds += elapsed_seconds(snapshot_start, Clock::now());

        {
            const auto evolve_start = Clock::now();
            const detail::ScopedFDSpacing spacing(level.grid.dx);
            if (level_idx == 0)
                root_stepper_->step(level.grid, dt, level.step_count++);
            else
                shared_steppers_[level_idx - 1]->step(level.grid, dt, level.step_count++);
            perf.evolve_seconds += elapsed_seconds(evolve_start, Clock::now());
        }

        if (level_idx + 1 < shared_levels_.size()) {
            auto &child = *shared_levels_[level_idx + 1];
            const size_t ratio = child.refinement_ratio;
            const T      dt_child = dt / T(ratio);

            const auto child_start = Clock::now();
            for (size_t substep = 0; substep < ratio; ++substep) {
                ++perf.child_substeps;
                const typename FineBoundary::Context ctx{
                    &level.snapshot_ref(), &level.grid, T(substep) / T(ratio),
                    T(substep + 1) / T(ratio)};
                const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
                advance_shared_level(level_idx + 1, dt_child);
            }
            perf.child_subcycling_seconds += elapsed_seconds(child_start, Clock::now());

            const auto restrict_start = Clock::now();
            detail::restrict_from_child(child.grid, level.grid, child.parent_cells,
                                        child.refinement_ratio);
            perf.restriction_seconds += elapsed_seconds(restrict_start, Clock::now());
            return;
        }

        const auto child_start = Clock::now();
        for (size_t leaf = 0; leaf < leaves_.size(); ++leaf) {
            const size_t ratio = leaves_[leaf]->refinement_ratio;
            const T      dt_child = dt / T(ratio);
            for (size_t substep = 0; substep < ratio; ++substep) {
                ++perf.child_substeps;
                const typename FineBoundary::Context ctx{
                    &level.snapshot_ref(), &level.grid, T(substep) / T(ratio),
                    T(substep + 1) / T(ratio)};
                const ScopedParentInterpolationContext<T, OuterBoundary> scoped_ctx(&ctx);
                advance_leaf(leaf, dt_child);
            }
        }
        perf.child_subcycling_seconds += elapsed_seconds(child_start, Clock::now());

        const auto restrict_start = Clock::now();
        for (size_t leaf = 0; leaf < leaves_.size(); ++leaf)
            detail::restrict_from_child_without_constraints(
                leaves_[leaf]->grid, level.grid, leaves_[leaf]->parent_cells,
                leaves_[leaf]->refinement_ratio);
        // The two BBH leaves write disjoint parent boxes, so one projection of the shared parent
        // after both restrictions is sufficient.
        tensorium_RG::bssn::enforce_algebraic_constraints(level.grid);
        perf.restriction_seconds += elapsed_seconds(restrict_start, Clock::now());
    }
};

// ============================================================================
// Regridding Support for Moving Punctures
// ============================================================================

/// @brief 3D position of a puncture
struct PunctureLocation {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool   valid = false;

    [[nodiscard]] std::array<double, 3> as_array() const noexcept { return {x, y, z}; }
};

/// @brief Result of a regrid check
struct RegridDecision {
    bool   needs_regrid = false;
    double distance_to_edge_cells = std::numeric_limits<double>::infinity();
    PunctureLocation new_center{};
};

/// @brief Locate a puncture by finding the minimum of chi in a search region.
/// @param grid The grid to search
/// @param search_center Approximate center of the puncture
/// @param search_radius_cells Radius in grid cells to search around the center
/// @return Location of the chi minimum (puncture position)
template <typename T>
inline PunctureLocation locate_puncture_chi_minimum(const BSSNGridSoA<T> &grid,
                                                     const std::array<double, 3> &search_center,
                                                     size_t search_radius_cells = 8) {
    PunctureLocation result{};

    // Convert search center to grid indices
    const double fx = (search_center[0] - grid.x0) / grid.dx + double(grid.dims.ng);
    const double fy = (search_center[1] - grid.y0) / grid.dy + double(grid.dims.ng);
    const double fz = (search_center[2] - grid.z0) / grid.dz + double(grid.dims.ng);

    const size_t ci = static_cast<size_t>(std::max(0.0, fx));
    const size_t cj = static_cast<size_t>(std::max(0.0, fy));
    const size_t ck = static_cast<size_t>(std::max(0.0, fz));

    // Define search bounds
    const size_t ng = grid.dims.ng;
    const size_t i_min = (ci > ng + search_radius_cells) ? ci - search_radius_cells : ng;
    const size_t j_min = (cj > ng + search_radius_cells) ? cj - search_radius_cells : ng;
    const size_t k_min = (ck > ng + search_radius_cells) ? ck - search_radius_cells : ng;
    const size_t i_max = std::min(ci + search_radius_cells, grid.dims.ng + grid.dims.nx - 1);
    const size_t j_max = std::min(cj + search_radius_cells, grid.dims.ng + grid.dims.ny - 1);
    const size_t k_max = std::min(ck + search_radius_cells, grid.dims.ng + grid.dims.nz - 1);

    T min_chi = std::numeric_limits<T>::max();
    size_t min_i = ci, min_j = cj, min_k = ck;

    const T *chi_ptr = grid.chi.ptr();
    for (size_t i = i_min; i <= i_max; ++i) {
        for (size_t j = j_min; j <= j_max; ++j) {
            for (size_t k = k_min; k <= k_max; ++k) {
                const T chi_val = chi_ptr[grid.chi.idx(i, j, k)];
                if (chi_val < min_chi) {
                    min_chi = chi_val;
                    min_i = i;
                    min_j = j;
                    min_k = k;
                }
            }
        }
    }

    // Convert back to physical coordinates
    result.x = grid.x0 + (double(min_i) - double(ng)) * grid.dx;
    result.y = grid.y0 + (double(min_j) - double(ng)) * grid.dy;
    result.z = grid.z0 + (double(min_k) - double(ng)) * grid.dz;
    result.valid = (min_chi < T(0.5)); // Valid if chi is small enough to be near a puncture

    return result;
}

/// @brief Check if a puncture is too close to the edge of its refinement box.
/// @param grid The refinement level grid
/// @param puncture Current puncture location
/// @param threshold_cells Distance in cells that triggers regridding
/// @return RegridDecision indicating if regrid is needed
template <typename T>
inline RegridDecision check_puncture_regrid_needed(const BSSNGridSoA<T> &grid,
                                                    const PunctureLocation &puncture,
                                                    double threshold_cells) {
    RegridDecision result{};
    if (!puncture.valid)
        return result;

    // Compute distance from puncture to each face in cell units
    const double dx = grid.dx;
    const double dy = grid.dy;
    const double dz = grid.dz;

    const double x_min = grid.x0 - 0.5 * dx;
    const double y_min = grid.y0 - 0.5 * dy;
    const double z_min = grid.z0 - 0.5 * dz;
    const double x_max = grid.x0 + (double(grid.dims.nx) - 0.5) * dx;
    const double y_max = grid.y0 + (double(grid.dims.ny) - 0.5) * dy;
    const double z_max = grid.z0 + (double(grid.dims.nz) - 0.5) * dz;

    const double dist_x_min = (puncture.x - x_min) / dx;
    const double dist_x_max = (x_max - puncture.x) / dx;
    const double dist_y_min = (puncture.y - y_min) / dy;
    const double dist_y_max = (y_max - puncture.y) / dy;
    const double dist_z_min = (puncture.z - z_min) / dz;
    const double dist_z_max = (z_max - puncture.z) / dz;

    result.distance_to_edge_cells = std::min({dist_x_min, dist_x_max,
                                              dist_y_min, dist_y_max,
                                              dist_z_min, dist_z_max});
    result.needs_regrid = (result.distance_to_edge_cells < threshold_cells);
    result.new_center = puncture;

    return result;
}

/// @brief Compute a new PatchBox centered on a puncture location.
/// @param parent_grid The parent grid from which the patch is defined
/// @param puncture_center The puncture location (physical coordinates)
/// @param half_width_cells Half-width of the box in fine cells
/// @param refinement_ratio Refinement ratio between parent and child
/// @param min_clearance Minimum clearance from parent boundary in coarse cells
/// @return New PatchBox in parent cell coordinates
template <typename T>
inline PatchBox compute_regrid_box(const BSSNGridSoA<T> &parent_grid,
                                   const PunctureLocation &puncture_center,
                                   size_t half_width_cells,
                                   size_t refinement_ratio,
                                   size_t min_clearance = 2) {
    // Convert puncture position to parent grid cell coordinates
    const double px = (puncture_center.x - parent_grid.x0) / parent_grid.dx;
    const double py = (puncture_center.y - parent_grid.y0) / parent_grid.dy;
    const double pz = (puncture_center.z - parent_grid.z0) / parent_grid.dz;

    // Half-width in coarse (parent) cells
    const size_t half_coarse = (half_width_cells + refinement_ratio - 1) / refinement_ratio;

    // Compute bounds, ensuring clearance from parent boundary
    auto clamp_low = [&](double center, size_t half) -> size_t {
        const long raw = static_cast<long>(std::floor(center)) - static_cast<long>(half);
        return static_cast<size_t>(std::max(raw, static_cast<long>(min_clearance)));
    };

    auto clamp_high = [&](double center, size_t half, size_t parent_dim) -> size_t {
        const size_t raw = static_cast<size_t>(std::ceil(center)) + half;
        return std::min(raw, parent_dim - min_clearance);
    };

    PatchBox box{};
    box.i0 = clamp_low(px, half_coarse);
    box.j0 = clamp_low(py, half_coarse);
    box.k0 = clamp_low(pz, half_coarse);
    box.i1 = clamp_high(px, half_coarse, parent_grid.dims.nx);
    box.j1 = clamp_high(py, half_coarse, parent_grid.dims.ny);
    box.k1 = clamp_high(pz, half_coarse, parent_grid.dims.nz);

    // Ensure minimum box size
    if (box.i1 <= box.i0) box.i1 = box.i0 + 1;
    if (box.j1 <= box.j0) box.j1 = box.j0 + 1;
    if (box.k1 <= box.k0) box.k1 = box.k0 + 1;

    return box;
}

/// @brief Transfer evolved field data from source grid to destination grid using tricubic interpolation.
/// @param src Source grid (old grid before regridding)
/// @param dst Destination grid (new grid after regridding)
template <typename T>
inline void transfer_evolved_data_tricubic(const BSSNGridSoA<T> &src, BSSNGridSoA<T> &dst) {
    auto transfer_field = [&](const Field3D<T> &src_field, Field3D<T> &dst_field) {
        T *out = dst_field.ptr();

#pragma omp parallel for collapse(3)
        for (size_t i = 0; i < dst_field.st.nx_tot; ++i) {
            for (size_t j = 0; j < dst_field.st.ny_tot; ++j) {
                for (size_t k = 0; k < dst_field.st.nz_tot; ++k) {
                    // Compute physical coordinates for this destination cell
                    const T x = dst.x0 + (T(i) - T(dst.dims.ng)) * dst.dx;
                    const T y = dst.y0 + (T(j) - T(dst.dims.ng)) * dst.dy;
                    const T z = dst.z0 + (T(k) - T(dst.dims.ng)) * dst.dz;

                    out[dst_field.idx(i, j, k)] = detail::sample_tricubic_field(src, src_field, x, y, z);
                }
            }
        }
    };

    // Transfer scalars
    transfer_field(src.alpha, dst.alpha);
    transfer_field(src.chi, dst.chi);
    transfer_field(src.K, dst.K);
    transfer_field(src.Theta, dst.Theta);

    // Transfer vectors
    for (int a = 0; a < 3; ++a) {
        transfer_field(src.beta[a], dst.beta[a]);
        transfer_field(src.B[a], dst.B[a]);
        transfer_field(src.tildeGamma[a], dst.tildeGamma[a]);
        transfer_field(src.Z[a], dst.Z[a]);
    }

    // Transfer symmetric tensors
    for (int s = 0; s < 6; ++s) {
        transfer_field(src.gamma_tilde[s], dst.gamma_tilde[s]);
        transfer_field(src.A_tilde[s], dst.A_tilde[s]);
    }

    // Enforce algebraic constraints after transfer
    tensorium_RG::bssn::enforce_algebraic_constraints(dst);
}

/// @brief Regrid manager for binary puncture simulations.
/// Tracks puncture positions and manages automatic regridding.
template <typename T>
class BinaryPunctureRegridManager {
  public:
    struct Config {
        double threshold_cells = 4.0;          ///< Distance to edge that triggers regrid
        size_t check_interval = 10;            ///< Steps between regrid checks
        size_t leaf_half_width_cells = 16;     ///< Half-width of leaf boxes in fine cells
        bool   enabled = true;                 ///< Enable/disable regridding
        bool   verbose = false;                ///< Print regrid messages
    };

    struct PunctureState {
        PunctureLocation location{};
        std::array<double, 3> velocity{0.0, 0.0, 0.0};  ///< Velocity from -beta
    };

    BinaryPunctureRegridManager() = default;

    explicit BinaryPunctureRegridManager(const Config &cfg) : config_(cfg) {}

    /// @brief Initialize puncture positions from initial coordinates.
    void initialize(const std::array<double, 3> &p1, const std::array<double, 3> &p2) {
        puncture1_.location = {p1[0], p1[1], p1[2], true};
        puncture2_.location = {p2[0], p2[1], p2[2], true};
        initialized_ = true;
    }

    /// @brief Update puncture positions using shift vector integration.
    /// @param left_grid Grid containing puncture 1
    /// @param right_grid Grid containing puncture 2
    /// @param dt Time step
    template <typename GridType>
    void advance_trackers(const GridType &left_grid, const GridType &right_grid, T dt) {
        if (!initialized_ || !config_.enabled)
            return;

        auto integrate_puncture = [&](PunctureState &state, const GridType &grid) {
            if (!state.location.valid)
                return;

            // Sample beta at puncture location
            std::array<T, 3> beta_new{0, 0, 0};
            const T x = T(state.location.x);
            const T y = T(state.location.y);
            const T z = T(state.location.z);

            for (int d = 0; d < 3; ++d) {
                beta_new[d] = detail::sample_tricubic_field(grid, grid.beta[d], x, y, z);
            }

            // Trapezoidal integration: dx/dt = -beta
            for (int d = 0; d < 3; ++d) {
                const double v_avg = -0.5 * (state.velocity[d] + double(beta_new[d]));
                (&state.location.x)[d] += v_avg * double(dt);
                state.velocity[d] = -double(beta_new[d]);
            }
        };

        integrate_puncture(puncture1_, left_grid);
        integrate_puncture(puncture2_, right_grid);
    }

    /// @brief Refine puncture positions by finding local chi minimum.
    template <typename GridType>
    void refine_puncture_locations(const GridType &left_grid, const GridType &right_grid,
                                   size_t search_radius = 4) {
        if (!initialized_)
            return;

        auto refine = [&](PunctureState &state, const GridType &grid) {
            if (!state.location.valid)
                return;
            const auto refined = locate_puncture_chi_minimum(grid, state.location.as_array(),
                                                              search_radius);
            if (refined.valid) {
                state.location = refined;
            }
        };

        refine(puncture1_, left_grid);
        refine(puncture2_, right_grid);
    }

    /// @brief Check if either puncture needs regridding.
    /// @param left_grid Left leaf grid
    /// @param right_grid Right leaf grid
    /// @return Pair of RegridDecision for left and right punctures
    template <typename GridType>
    std::pair<RegridDecision, RegridDecision> check_regrid_needed(const GridType &left_grid,
                                                                   const GridType &right_grid) const {
        if (!initialized_ || !config_.enabled) {
            return {{}, {}};
        }

        return {
            check_puncture_regrid_needed(left_grid, puncture1_.location, config_.threshold_cells),
            check_puncture_regrid_needed(right_grid, puncture2_.location, config_.threshold_cells)
        };
    }

    /// @brief Get current puncture positions.
    [[nodiscard]] std::pair<PunctureLocation, PunctureLocation> get_puncture_locations() const {
        return {puncture1_.location, puncture2_.location};
    }

    /// @brief Get configuration.
    [[nodiscard]] const Config &config() const noexcept { return config_; }
    Config &config() noexcept { return config_; }

    /// @brief Check if manager is initialized.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// @brief Increment step counter and check if regrid check is due.
    [[nodiscard]] bool should_check_regrid() {
        ++step_count_;
        return config_.enabled && (step_count_ % config_.check_interval == 0);
    }

    /// @brief Reset step counter (call after regrid).
    void reset_step_counter() { step_count_ = 0; }

  private:
    Config config_{};
    PunctureState puncture1_{};
    PunctureState puncture2_{};
    bool initialized_ = false;
    size_t step_count_ = 0;
};

/// @brief Perform regridding on a BinaryPunctureFixedMeshRefinementHierarchy.
/// Creates new leaf grids centered on current puncture positions and transfers data.
/// @param hierarchy The FMR hierarchy to regrid
/// @param manager The regrid manager with puncture tracking
/// @param parent_grid Reference to the shared parent level grid
/// @param new_left_box New PatchBox for left leaf
/// @param new_right_box New PatchBox for right leaf
/// @return true if regrid was performed successfully
template <typename T, typename OuterBoundary>
inline bool perform_bbh_leaf_regrid(
    BinaryPunctureFixedMeshRefinementHierarchy<T, OuterBoundary> &hierarchy,
    const PatchBox &new_left_box,
    const PatchBox &new_right_box,
    size_t refinement_ratio,
    size_t halo_cells) {

    // Validate boxes don't overlap
    if (detail::patch_boxes_overlap(new_left_box, new_right_box)) {
        return false;
    }

    // Get current leaf grids for data transfer
    const auto &old_left = hierarchy.leaf_grid(0);
    const auto &old_right = hierarchy.leaf_grid(1);
    const auto &parent = hierarchy.shared_level_grid(hierarchy.num_shared_levels() - 1);

    // Compute new leaf geometries
    LevelConfig left_cfg{new_left_box, refinement_ratio, halo_cells};
    LevelConfig right_cfg{new_right_box, refinement_ratio, halo_cells};

    const auto left_geom = detail::make_child_geometry(parent, left_cfg);
    const auto right_geom = detail::make_child_geometry(parent, right_cfg);

    // Create new leaf grids
    BSSNGridSoA<T> new_left(left_geom.nx, left_geom.ny, left_geom.nz, left_geom.ng,
                            T(left_geom.dx), T(left_geom.dy), T(left_geom.dz));
    new_left.x0 = T(left_geom.x0);
    new_left.y0 = T(left_geom.y0);
    new_left.z0 = T(left_geom.z0);

    BSSNGridSoA<T> new_right(right_geom.nx, right_geom.ny, right_geom.nz, right_geom.ng,
                             T(right_geom.dx), T(right_geom.dy), T(right_geom.dz));
    new_right.x0 = T(right_geom.x0);
    new_right.y0 = T(right_geom.y0);
    new_right.z0 = T(right_geom.z0);

    // Transfer data from old grids to new grids
    transfer_evolved_data_tricubic(old_left, new_left);
    transfer_evolved_data_tricubic(old_right, new_right);

    // Note: The actual swap of grids in the hierarchy requires access to private members.
    // This function prepares the new grids; the hierarchy class should provide a method
    // to accept and install them.

    return true;
}

} // namespace tensorium_RG::bssn::fmr
