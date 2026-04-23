#include "BSSNCudaGridOps.hpp"

#ifdef TENSORIUM_CUDA

#include <Tensorium/Backend/CUDA/Core/CudaRuntime.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace tensorium_RG::bssn::cuda {
namespace {

struct Bounds3D {
    std::size_t i0 = 0;
    std::size_t i1 = 0;
    std::size_t j0 = 0;
    std::size_t j1 = 0;
    std::size_t k0 = 0;
    std::size_t k1 = 0;
};

template <typename T> inline Bounds3D interior_bounds(const BSSNGridView<T> &grid, std::size_t padding) {
    Bounds3D bounds;
    grid.domain_bounds(bounds.i0, bounds.i1, bounds.j0, bounds.j1, bounds.k0, bounds.k1);
    bounds.i0 = (bounds.i0 + padding < bounds.i1) ? bounds.i0 + padding : bounds.i1;
    bounds.j0 = (bounds.j0 + padding < bounds.j1) ? bounds.j0 + padding : bounds.j1;
    bounds.k0 = (bounds.k0 + padding < bounds.k1) ? bounds.k0 + padding : bounds.k1;
    bounds.i1 = (bounds.i1 > padding) ? bounds.i1 - padding : bounds.i1;
    bounds.j1 = (bounds.j1 > padding) ? bounds.j1 - padding : bounds.j1;
    bounds.k1 = (bounds.k1 > padding) ? bounds.k1 - padding : bounds.k1;
    return bounds;
}

template <typename T> inline Bounds3D domain_bounds(const BSSNGridView<T> &grid) {
    Bounds3D bounds;
    grid.domain_bounds(bounds.i0, bounds.i1, bounds.j0, bounds.j1, bounds.k0, bounds.k1);
    return bounds;
}

inline dim3 make_launch_grid(std::size_t ni, std::size_t nj, std::size_t nk, const dim3 &block) {
    return dim3(static_cast<unsigned>((ni + block.x - 1) / block.x),
                static_cast<unsigned>((nj + block.y - 1) / block.y),
                static_cast<unsigned>((nk + block.z - 1) / block.z));
}

template <typename T> struct StageGridPack {
    using value_type = std::remove_const_t<T>;

    GridDims             dims{};
    Strides<value_type> st{};
    value_type           dx{};
    value_type           dy{};
    value_type           dz{};
    value_type           x0{};
    value_type           y0{};
    value_type           z0{};
    T                  *alpha = nullptr;
    T                  *chi = nullptr;
    T                  *K = nullptr;
    T                  *Theta = nullptr;
    T                  *beta[3] = {};
    T                  *B[3] = {};
    T                  *tildeGamma[3] = {};
    T                  *Z[3] = {};
    T                  *gamma_tilde[6] = {};
    T                  *gamma_tilde_inv[6] = {};
    T                  *A_tilde[6] = {};
    T                  *Ricci[6] = {};

    __host__ __device__ std::size_t idx(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return i * st.sx + j * st.sy + k * st.sz;
    }
};

template <typename T> struct StageRhsPack {
    using value_type = std::remove_const_t<T>;

    Strides<value_type> st{};
    T                  *alpha = nullptr;
    T                  *chi = nullptr;
    T                  *K = nullptr;
    T                  *Theta = nullptr;
    T                  *beta[3] = {};
    T                  *B[3] = {};
    T                  *gamma_tilde[6] = {};
    T                  *A_tilde[6] = {};
    T                  *tildeGamma[3] = {};
    T                  *Z[3] = {};

    __host__ __device__ std::size_t idx(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return i * st.sx + j * st.sy + k * st.sz;
    }
};

template <typename T> StageGridPack<T> make_stage_pack(BSSNGridView<T> view) {
    StageGridPack<T> pack;
    pack.dims = view.dims;
    pack.st = view.st;
    pack.dx = view.dx;
    pack.dy = view.dy;
    pack.dz = view.dz;
    pack.x0 = view.x0;
    pack.y0 = view.y0;
    pack.z0 = view.z0;
    pack.alpha = view.alpha.ptr();
    pack.chi = view.chi.ptr();
    pack.K = view.K.ptr();
    pack.Theta = view.Theta.ptr();
    for (int c = 0; c < 3; ++c) {
        pack.beta[c] = view.beta[c].ptr();
        pack.B[c] = view.B[c].ptr();
        pack.tildeGamma[c] = view.tildeGamma[c].ptr();
        pack.Z[c] = view.Z[c].ptr();
    }
    for (int s = 0; s < 6; ++s) {
        pack.gamma_tilde[s] = view.gamma_tilde[s].ptr();
        pack.gamma_tilde_inv[s] = view.gamma_tilde_inv[s].ptr();
        pack.A_tilde[s] = view.A_tilde[s].ptr();
        pack.Ricci[s] = view.Ricci[s].ptr();
    }
    return pack;
}

template <typename T> StageRhsPack<T> make_rhs_pack(BSSNRHSWorkspaceView<T> view) {
    StageRhsPack<T> pack;
    pack.st = view.st;
    pack.alpha = view.alpha.ptr();
    pack.chi = view.chi.ptr();
    pack.K = view.K.ptr();
    pack.Theta = view.Theta.ptr();
    for (int c = 0; c < 3; ++c) {
        pack.beta[c] = view.beta[c].ptr();
        pack.B[c] = view.B[c].ptr();
        pack.tildeGamma[c] = view.tildeGamma[c].ptr();
        pack.Z[c] = view.Z[c].ptr();
    }
    for (int s = 0; s < 6; ++s) {
        pack.gamma_tilde[s] = view.gamma_tilde[s].ptr();
        pack.A_tilde[s] = view.A_tilde[s].ptr();
    }
    return pack;
}

template <int Order, typename T>
__device__ __forceinline__ T d_axis_ptr_order(const T *p, ptrdiff_t stride, T inv_60d) {
    static_assert(Order == 4 || Order == 6, "Only 4th- and 6th-order derivatives are supported");
    if constexpr (Order == 4) {
        return (-p[2 * stride] + T(8) * p[stride] - T(8) * p[-stride] + p[-2 * stride]) *
               (T(5) * inv_60d);
    }
    return (p[3 * stride] - T(9) * p[2 * stride] + T(45) * p[stride] - T(45) * p[-stride] +
            T(9) * p[-2 * stride] - p[-3 * stride]) *
           inv_60d;
}

template <typename T>
__device__ __forceinline__ T d_axis_ptr(const T *p, ptrdiff_t stride, T inv_60d, int spatial_order) {
    return (spatial_order == 4) ? d_axis_ptr_order<4>(p, stride, inv_60d)
                                : d_axis_ptr_order<6>(p, stride, inv_60d);
}

template <int Order, typename T>
__device__ __forceinline__ T d2_axis_ptr_order(const T *p, ptrdiff_t stride, T inv_180d2) {
    static_assert(Order == 4 || Order == 6, "Only 4th- and 6th-order derivatives are supported");
    if constexpr (Order == 4) {
        return (-p[2 * stride] + T(16) * p[stride] - T(30) * p[0] + T(16) * p[-stride] -
                p[-2 * stride]) *
               (T(15) * inv_180d2);
    }
    return (T(2) * p[-3 * stride] - T(27) * p[-2 * stride] + T(270) * p[-stride] -
            T(490) * p[0] + T(270) * p[stride] - T(27) * p[2 * stride] +
            T(2) * p[3 * stride]) *
           inv_180d2;
}

template <typename T>
__device__ __forceinline__ T d2_axis_ptr(const T *p, ptrdiff_t stride, T inv_180d2,
                                         int spatial_order) {
    return (spatial_order == 4) ? d2_axis_ptr_order<4>(p, stride, inv_180d2)
                                : d2_axis_ptr_order<6>(p, stride, inv_180d2);
}

template <typename T>
__device__ __forceinline__ T upwind_axis_ptr(const T *p, ptrdiff_t stride, T inv_2d, T beta) {
    if (beta < T(0)) {
        const T dl = (T(1.0 / 30.0)) * p[-4 * stride] + (T(-4.0 / 15.0)) * p[-3 * stride] +
                     p[-2 * stride] + (T(-8.0 / 3.0)) * p[-stride] + (T(7.0 / 6.0)) * p[0] +
                     (T(4.0 / 5.0)) * p[stride] + (T(-1.0 / 15.0)) * p[2 * stride];
        return dl * inv_2d;
    }
    const T dr = (T(-1.0 / 30.0)) * p[4 * stride] + (T(4.0 / 15.0)) * p[3 * stride] -
                 p[2 * stride] + (T(8.0 / 3.0)) * p[stride] + (T(-7.0 / 6.0)) * p[0] +
                 (T(-4.0 / 5.0)) * p[-stride] + (T(1.0 / 15.0)) * p[-2 * stride];
    return dr * inv_2d;
}

template <typename T> __device__ __forceinline__ T ko6_axis_ptr(const T *p, ptrdiff_t stride) {
    return (p[-3 * stride] - T(6) * p[-2 * stride] + T(15) * p[-stride] - T(20) * p[0] +
            T(15) * p[stride] - T(6) * p[2 * stride] + p[3 * stride]) *
           T(1.0 / 64.0);
}

template <typename T>
__device__ __forceinline__ T dxy4_ptr(const T *p, ptrdiff_t sx, ptrdiff_t sy, T inv_144dxdy) {
    const ptrdiff_t sx2 = 2 * sx;
    const ptrdiff_t sy2 = 2 * sy;
    const T sum = p[-sx2 - sy2] - T(8) * p[-sx2 - sy] + T(8) * p[-sx2 + sy] - p[-sx2 + sy2] -
                  T(8) * p[-sx - sy2] + T(64) * p[-sx - sy] - T(64) * p[-sx + sy] +
                  T(8) * p[-sx + sy2] + T(8) * p[sx - sy2] - T(64) * p[sx - sy] +
                  T(64) * p[sx + sy] - T(8) * p[sx + sy2] - p[sx2 - sy2] +
                  T(8) * p[sx2 - sy] - T(8) * p[sx2 + sy] + p[sx2 + sy2];
    return sum * inv_144dxdy;
}

template <typename T>
__device__ __forceinline__ T dxz4_ptr(const T *p, ptrdiff_t sx, T inv_144dxdz) {
    const ptrdiff_t sx2 = 2 * sx;
    const T sum = p[-sx2 - 2] - T(8) * p[-sx2 - 1] + T(8) * p[-sx2 + 1] - p[-sx2 + 2] -
                  T(8) * p[-sx - 2] + T(64) * p[-sx - 1] - T(64) * p[-sx + 1] +
                  T(8) * p[-sx + 2] + T(8) * p[sx - 2] - T(64) * p[sx - 1] +
                  T(64) * p[sx + 1] - T(8) * p[sx + 2] - p[sx2 - 2] + T(8) * p[sx2 - 1] -
                  T(8) * p[sx2 + 1] + p[sx2 + 2];
    return sum * inv_144dxdz;
}

template <typename T>
__device__ __forceinline__ T dyz4_ptr(const T *p, ptrdiff_t sy, T inv_144dydz) {
    const ptrdiff_t sy2 = 2 * sy;
    const T sum = p[-sy2 - 2] - T(8) * p[-sy2 - 1] + T(8) * p[-sy2 + 1] - p[-sy2 + 2] -
                  T(8) * p[-sy - 2] + T(64) * p[-sy - 1] - T(64) * p[-sy + 1] +
                  T(8) * p[-sy + 2] + T(8) * p[sy - 2] - T(64) * p[sy - 1] +
                  T(64) * p[sy + 1] - T(8) * p[sy + 2] - p[sy2 - 2] + T(8) * p[sy2 - 1] -
                  T(8) * p[sy2 + 1] + p[sy2 + 2];
    return sum * inv_144dydz;
}

template <typename T> __device__ __forceinline__ T guard_chi_div_device(T chi, T chi_div_floor) {
    return (chi > chi_div_floor) ? chi : chi_div_floor;
}

template <int Order, typename T>
__device__ __forceinline__ void metric_inverse_divergence_components_ptr_order(
    const T *p_ginv_xx, const T *p_ginv_xy, const T *p_ginv_xz, const T *p_ginv_yy,
    const T *p_ginv_yz, const T *p_ginv_zz, ptrdiff_t sx, ptrdiff_t sy, T inv_60dx, T inv_60dy,
    T inv_60dz, T &out0, T &out1, T &out2) {
    out0 = d_axis_ptr_order<Order>(p_ginv_xx, sx, inv_60dx) +
           d_axis_ptr_order<Order>(p_ginv_xy, sy, inv_60dy) +
           d_axis_ptr_order<Order>(p_ginv_xz, 1, inv_60dz);
    out1 = d_axis_ptr_order<Order>(p_ginv_xy, sx, inv_60dx) +
           d_axis_ptr_order<Order>(p_ginv_yy, sy, inv_60dy) +
           d_axis_ptr_order<Order>(p_ginv_yz, 1, inv_60dz);
    out2 = d_axis_ptr_order<Order>(p_ginv_xz, sx, inv_60dx) +
           d_axis_ptr_order<Order>(p_ginv_yz, sy, inv_60dy) +
           d_axis_ptr_order<Order>(p_ginv_zz, 1, inv_60dz);
}

template <typename T>
__device__ __forceinline__ void recover_z_over_chi_components_from_gamma_ptr(
    const T *p_tildeGamma0, const T *p_tildeGamma1, const T *p_tildeGamma2, const T *p_ginv_xx,
    const T *p_ginv_xy, const T *p_ginv_xz, const T *p_ginv_yy, const T *p_ginv_yz,
    const T *p_ginv_zz, ptrdiff_t sx, ptrdiff_t sy, T inv_60dx, T inv_60dy, T inv_60dz,
    int spatial_order, T &z0, T &z1, T &z2) {
    T gamma_metric0 = T(0);
    T gamma_metric1 = T(0);
    T gamma_metric2 = T(0);
    if (spatial_order == 4) {
        metric_inverse_divergence_components_ptr_order<4>(p_ginv_xx, p_ginv_xy, p_ginv_xz,
                                                          p_ginv_yy, p_ginv_yz, p_ginv_zz, sx,
                                                          sy, inv_60dx, inv_60dy, inv_60dz,
                                                          gamma_metric0, gamma_metric1,
                                                          gamma_metric2);
    } else {
        metric_inverse_divergence_components_ptr_order<6>(p_ginv_xx, p_ginv_xy, p_ginv_xz,
                                                          p_ginv_yy, p_ginv_yz, p_ginv_zz, sx,
                                                          sy, inv_60dx, inv_60dy, inv_60dz,
                                                          gamma_metric0, gamma_metric1,
                                                          gamma_metric2);
    }

    gamma_metric0 = -gamma_metric0;
    gamma_metric1 = -gamma_metric1;
    gamma_metric2 = -gamma_metric2;

    z0 = T(0.5) * (p_tildeGamma0[0] - gamma_metric0);
    z1 = T(0.5) * (p_tildeGamma1[0] - gamma_metric1);
    z2 = T(0.5) * (p_tildeGamma2[0] - gamma_metric2);
}

template <typename T, typename Config>
__device__ __forceinline__ T ko_boundary_taper(const StageGridPack<const T> &grid, std::size_t i,
                                               std::size_t j, std::size_t k,
                                               const Config &cfg) {
    if (cfg.ko_boundary_width == 0)
        return T(1);

    const std::size_t i0 = grid.dims.ng;
    const std::size_t i1 = grid.dims.ng + grid.dims.nx;
    const std::size_t j0 = grid.dims.ng;
    const std::size_t j1 = grid.dims.ng + grid.dims.ny;
    const std::size_t k0 = grid.dims.ng;
    const std::size_t k1 = grid.dims.ng + grid.dims.nz;
    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return T(1);

    const std::size_t di = min(i - i0, (i1 - 1) - i);
    const std::size_t dj = min(j - j0, (j1 - 1) - j);
    const std::size_t dk = min(k - k0, (k1 - 1) - k);
    const std::size_t dist = min(di, min(dj, dk));
    if (dist >= cfg.ko_boundary_width)
        return T(1);

    const T x = T(dist) / T(cfg.ko_boundary_width);
    const T smooth = x * x * (T(3) - T(2) * x);
    return cfg.ko_boundary_floor + (T(1) - cfg.ko_boundary_floor) * smooth;
}

template <typename T, typename Config>
__device__ __forceinline__ T local_ko_scale(const StageGridPack<const T> &grid, std::size_t i,
                                            std::size_t j, std::size_t k, const Config &cfg) {
    return (cfg.ko_sigma / grid.dx) * ko_boundary_taper(grid, i, j, k, cfg);
}

template <typename T>
__device__ __forceinline__ void copy_state_at(StageGridPack<const T> src, StageGridPack<T> dst,
                                              std::size_t idx) {
    dst.alpha[idx] = src.alpha[idx];
    dst.chi[idx] = src.chi[idx];
    dst.K[idx] = src.K[idx];
    dst.Theta[idx] = src.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c][idx] = src.beta[c][idx];
        dst.B[c][idx] = src.B[c][idx];
        dst.tildeGamma[c][idx] = src.tildeGamma[c][idx];
        dst.Z[c][idx] = src.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s][idx] = src.gamma_tilde[s][idx];
        dst.A_tilde[s][idx] = src.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void copy_state_at(StageGridPack<const T> src, StageRhsPack<T> dst,
                                              std::size_t idx) {
    dst.alpha[idx] = src.alpha[idx];
    dst.chi[idx] = src.chi[idx];
    dst.K[idx] = src.K[idx];
    dst.Theta[idx] = src.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c][idx] = src.beta[c][idx];
        dst.B[c][idx] = src.B[c][idx];
        dst.tildeGamma[c][idx] = src.tildeGamma[c][idx];
        dst.Z[c][idx] = src.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s][idx] = src.gamma_tilde[s][idx];
        dst.A_tilde[s][idx] = src.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void accumulate_state_at(StageGridPack<const T> src, StageGridPack<T> dst,
                                                    T delta, std::size_t idx) {
    dst.alpha[idx] += delta * src.alpha[idx];
    dst.chi[idx] += delta * src.chi[idx];
    dst.K[idx] += delta * src.K[idx];
    dst.Theta[idx] += delta * src.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c][idx] += delta * src.beta[c][idx];
        dst.B[c][idx] += delta * src.B[c][idx];
        dst.tildeGamma[c][idx] += delta * src.tildeGamma[c][idx];
        dst.Z[c][idx] += delta * src.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s][idx] += delta * src.gamma_tilde[s][idx];
        dst.A_tilde[s][idx] += delta * src.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void accumulate_state_at(StageGridPack<const T> src, StageRhsPack<T> dst,
                                                    T delta, std::size_t idx) {
    dst.alpha[idx] += delta * src.alpha[idx];
    dst.chi[idx] += delta * src.chi[idx];
    dst.K[idx] += delta * src.K[idx];
    dst.Theta[idx] += delta * src.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c][idx] += delta * src.beta[c][idx];
        dst.B[c][idx] += delta * src.B[c][idx];
        dst.tildeGamma[c][idx] += delta * src.tildeGamma[c][idx];
        dst.Z[c][idx] += delta * src.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s][idx] += delta * src.gamma_tilde[s][idx];
        dst.A_tilde[s][idx] += delta * src.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void explicit_stage_update_at(StageGridPack<T> grid,
                                                         StageGridPack<const T> stage,
                                                         StageRhsPack<const T> rhs, T gam0, T gam1,
                                                         T beta_dt, std::size_t idx) {
    grid.alpha[idx] = gam0 * grid.alpha[idx] + gam1 * stage.alpha[idx] + beta_dt * rhs.alpha[idx];
    grid.chi[idx] = gam0 * grid.chi[idx] + gam1 * stage.chi[idx] + beta_dt * rhs.chi[idx];
    grid.K[idx] = gam0 * grid.K[idx] + gam1 * stage.K[idx] + beta_dt * rhs.K[idx];
    grid.Theta[idx] =
        gam0 * grid.Theta[idx] + gam1 * stage.Theta[idx] + beta_dt * rhs.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        grid.beta[c][idx] =
            gam0 * grid.beta[c][idx] + gam1 * stage.beta[c][idx] + beta_dt * rhs.beta[c][idx];
        grid.B[c][idx] = gam0 * grid.B[c][idx] + gam1 * stage.B[c][idx] + beta_dt * rhs.B[c][idx];
        grid.tildeGamma[c][idx] = gam0 * grid.tildeGamma[c][idx] +
                                  gam1 * stage.tildeGamma[c][idx] +
                                  beta_dt * rhs.tildeGamma[c][idx];
        grid.Z[c][idx] = gam0 * grid.Z[c][idx] + gam1 * stage.Z[c][idx] + beta_dt * rhs.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        grid.gamma_tilde[s][idx] = gam0 * grid.gamma_tilde[s][idx] +
                                   gam1 * stage.gamma_tilde[s][idx] +
                                   beta_dt * rhs.gamma_tilde[s][idx];
        grid.A_tilde[s][idx] = gam0 * grid.A_tilde[s][idx] + gam1 * stage.A_tilde[s][idx] +
                               beta_dt * rhs.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void explicit_stage_update_at(StageGridPack<T> grid,
                                                         StageRhsPack<const T> stage,
                                                         StageRhsPack<const T> rhs, T gam0, T gam1,
                                                         T beta_dt, std::size_t idx) {
    grid.alpha[idx] = gam0 * grid.alpha[idx] + gam1 * stage.alpha[idx] + beta_dt * rhs.alpha[idx];
    grid.chi[idx] = gam0 * grid.chi[idx] + gam1 * stage.chi[idx] + beta_dt * rhs.chi[idx];
    grid.K[idx] = gam0 * grid.K[idx] + gam1 * stage.K[idx] + beta_dt * rhs.K[idx];
    grid.Theta[idx] =
        gam0 * grid.Theta[idx] + gam1 * stage.Theta[idx] + beta_dt * rhs.Theta[idx];
    for (int c = 0; c < 3; ++c) {
        grid.beta[c][idx] =
            gam0 * grid.beta[c][idx] + gam1 * stage.beta[c][idx] + beta_dt * rhs.beta[c][idx];
        grid.B[c][idx] = gam0 * grid.B[c][idx] + gam1 * stage.B[c][idx] + beta_dt * rhs.B[c][idx];
        grid.tildeGamma[c][idx] = gam0 * grid.tildeGamma[c][idx] +
                                  gam1 * stage.tildeGamma[c][idx] +
                                  beta_dt * rhs.tildeGamma[c][idx];
        grid.Z[c][idx] = gam0 * grid.Z[c][idx] + gam1 * stage.Z[c][idx] + beta_dt * rhs.Z[c][idx];
    }
    for (int s = 0; s < 6; ++s) {
        grid.gamma_tilde[s][idx] = gam0 * grid.gamma_tilde[s][idx] +
                                   gam1 * stage.gamma_tilde[s][idx] +
                                   beta_dt * rhs.gamma_tilde[s][idx];
        grid.A_tilde[s][idx] = gam0 * grid.A_tilde[s][idx] + gam1 * stage.A_tilde[s][idx] +
                               beta_dt * rhs.A_tilde[s][idx];
    }
}

template <typename T>
__device__ __forceinline__ void compute_gauge_rhs_at(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                                     GaugeRHSCudaConfig<T> cfg, std::size_t i,
                                                     std::size_t j, std::size_t k,
                                                     std::size_t idx) {
    const ptrdiff_t sx = grid.st.sx;
    const ptrdiff_t sy = grid.st.sy;
    const T inv_60dx = T(1) / (T(60) * grid.dx);
    const T inv_60dy = T(1) / (T(60) * grid.dy);
    const T inv_60dz = T(1) / (T(60) * grid.dz);
    const T inv_2dx = T(1) / (T(2) * grid.dx);
    const T inv_2dy = T(1) / (T(2) * grid.dy);
    const T inv_2dz = T(1) / (T(2) * grid.dz);
    const T ko_scale = local_ko_scale(grid, i, j, k, cfg);

    const T alpha = grid.alpha[idx];
    const T chi = grid.chi[idx];
    const T K = grid.K[idx];
    const T theta = grid.Theta[idx];
    const T bx = grid.beta[0][idx];
    const T by = grid.beta[1][idx];
    const T bz = grid.beta[2][idx];

    const T *p_alpha = grid.alpha + idx;
    const T *p_chi = grid.chi + idx;
    const T *p_beta0 = grid.beta[0] + idx;
    const T *p_beta1 = grid.beta[1] + idx;
    const T *p_beta2 = grid.beta[2] + idx;
    const T *p_B0 = grid.B[0] + idx;
    const T *p_B1 = grid.B[1] + idx;
    const T *p_B2 = grid.B[2] + idx;
    const T *p_tg0 = grid.tildeGamma[0] + idx;
    const T *p_tg1 = grid.tildeGamma[1] + idx;
    const T *p_tg2 = grid.tildeGamma[2] + idx;

    const T adv_alpha = bx * upwind_axis_ptr(p_alpha, sx, inv_2dx, bx) +
                        by * upwind_axis_ptr(p_alpha, sy, inv_2dy, by) +
                        bz * upwind_axis_ptr(p_alpha, 1, inv_2dz, bz);
    const T diss_alpha =
        ko6_axis_ptr(p_alpha, sx) + ko6_axis_ptr(p_alpha, sy) + ko6_axis_ptr(p_alpha, 1);
    const T lapse_K = cfg.use_theta_in_lapse ? (K - T(2) * theta) : K;
    const T f = cfg.lapse_oplog * cfg.lapse_harmonicf + cfg.lapse_harmonic * alpha;
    rhs.alpha[idx] = cfg.lapse_advect * adv_alpha - cfg.slow_start_lapse_factor * f * alpha * lapse_K +
                     ko_scale * diss_alpha;

    const T adv_chi = bx * upwind_axis_ptr(p_chi, sx, inv_2dx, bx) +
                      by * upwind_axis_ptr(p_chi, sy, inv_2dy, by) +
                      bz * upwind_axis_ptr(p_chi, 1, inv_2dz, bz);
    const T div_beta = d_axis_ptr(p_beta0, sx, inv_60dx, cfg.spatial_order) +
                       d_axis_ptr(p_beta1, sy, inv_60dy, cfg.spatial_order) +
                       d_axis_ptr(p_beta2, 1, inv_60dz, cfg.spatial_order);
    const T diss_chi =
        ko6_axis_ptr(p_chi, sx) + ko6_axis_ptr(p_chi, sy) + ko6_axis_ptr(p_chi, 1);
    rhs.chi[idx] = adv_chi + (T(2.0 / 3.0)) * chi * (alpha * K - div_beta) + ko_scale * diss_chi;

    const bool use_beta_advection = (cfg.shift_advect != T(0));
    const bool use_B_shift_advection = cfg.use_shift_advection && use_beta_advection;
    const T rhs_gamma0 = rhs.tildeGamma[0][idx];
    const T rhs_gamma1 = rhs.tildeGamma[1][idx];
    const T rhs_gamma2 = rhs.tildeGamma[2][idx];

    const auto beta_rhs_component = [&](const T *p_beta_comp, T B_comp) -> T {
        const T diss =
            ko6_axis_ptr(p_beta_comp, sx) + ko6_axis_ptr(p_beta_comp, sy) + ko6_axis_ptr(p_beta_comp, 1);
        if (!use_beta_advection)
            return cfg.beta_B_coeff * B_comp + ko_scale * diss;
        const T adv = bx * upwind_axis_ptr(p_beta_comp, sx, inv_2dx, bx) +
                      by * upwind_axis_ptr(p_beta_comp, sy, inv_2dy, by) +
                      bz * upwind_axis_ptr(p_beta_comp, 1, inv_2dz, bz);
        return cfg.beta_B_coeff * B_comp + cfg.shift_advect * adv + ko_scale * diss;
    };

    rhs.beta[0][idx] = beta_rhs_component(p_beta0, grid.B[0][idx]);
    rhs.beta[1][idx] = beta_rhs_component(p_beta1, grid.B[1][idx]);
    rhs.beta[2][idx] = beta_rhs_component(p_beta2, grid.B[2][idx]);

    const auto B_rhs_component = [&](const T *p_B_comp, const T *p_tg_comp, T B_comp,
                                     T rhs_gamma_comp) -> T {
        const T diss =
            ko6_axis_ptr(p_B_comp, sx) + ko6_axis_ptr(p_B_comp, sy) + ko6_axis_ptr(p_B_comp, 1);
        if (!use_B_shift_advection)
            return rhs_gamma_comp - cfg.eta_coeff * B_comp + ko_scale * diss;

        const T adv_B = cfg.shift_advect *
                        (bx * upwind_axis_ptr(p_B_comp, sx, inv_2dx, bx) +
                         by * upwind_axis_ptr(p_B_comp, sy, inv_2dy, by) +
                         bz * upwind_axis_ptr(p_B_comp, 1, inv_2dz, bz));
        const T adv_Gamma = cfg.shift_advect *
                            (bx * upwind_axis_ptr(p_tg_comp, sx, inv_2dx, bx) +
                             by * upwind_axis_ptr(p_tg_comp, sy, inv_2dy, by) +
                             bz * upwind_axis_ptr(p_tg_comp, 1, inv_2dz, bz));
        return (rhs_gamma_comp - adv_Gamma) + adv_B - cfg.eta_coeff * B_comp + ko_scale * diss;
    };

    rhs.B[0][idx] = B_rhs_component(p_B0, p_tg0, grid.B[0][idx], rhs_gamma0);
    rhs.B[1][idx] = B_rhs_component(p_B1, p_tg1, grid.B[1][idx], rhs_gamma1);
    rhs.B[2][idx] = B_rhs_component(p_B2, p_tg2, grid.B[2][idx], rhs_gamma2);
}

template <typename T>
__device__ __forceinline__ T kappa1_times_lapse_device(T lapse,
                                                       const ScalarRHSCudaConfig<T> &cfg) {
    return cfg.covariant_z4 ? cfg.kappa1 : (cfg.kappa1 * lapse);
}

template <typename T>
__device__ __forceinline__ void compute_scalar_rhs_at(StageGridPack<const T> grid,
                                                      StageRhsPack<T> rhs,
                                                      const T *z4_conformal_trace,
                                                      ScalarRHSCudaConfig<T> cfg, std::size_t i,
                                                      std::size_t j, std::size_t k,
                                                      std::size_t idx) {
    constexpr int sym_row[6] = {0, 0, 0, 1, 1, 2};
    constexpr int sym_col[6] = {0, 1, 2, 1, 2, 2};

    const ptrdiff_t sx = grid.st.sx;
    const ptrdiff_t sy = grid.st.sy;
    const T inv_60dx = T(1) / (T(60) * grid.dx);
    const T inv_60dy = T(1) / (T(60) * grid.dy);
    const T inv_60dz = T(1) / (T(60) * grid.dz);
    const T inv_2dx = T(1) / (T(2) * grid.dx);
    const T inv_2dy = T(1) / (T(2) * grid.dy);
    const T inv_2dz = T(1) / (T(2) * grid.dz);
    const T inv_180dx2 = T(1) / (T(180) * grid.dx * grid.dx);
    const T inv_180dy2 = T(1) / (T(180) * grid.dy * grid.dy);
    const T inv_180dz2 = T(1) / (T(180) * grid.dz * grid.dz);
    const T inv_144dxdy = T(1) / (T(144) * grid.dx * grid.dy);
    const T inv_144dxdz = T(1) / (T(144) * grid.dx * grid.dz);
    const T inv_144dydz = T(1) / (T(144) * grid.dy * grid.dz);
    const T ko_scale = local_ko_scale(grid, i, j, k, cfg);

    const T alpha = grid.alpha[idx];
    const T chi = grid.chi[idx];
    const T chi_guarded = guard_chi_div_device(chi, cfg.chi_div_floor);
    const T inv_chi = T(1) / chi_guarded;
    const T K_val = grid.K[idx];
    const T theta = grid.Theta[idx];
    const T bx = grid.beta[0][idx];
    const T by = grid.beta[1][idx];
    const T bz = grid.beta[2][idx];

    const T *p_alpha = grid.alpha + idx;
    const T *p_chi = grid.chi + idx;
    const T *p_K = grid.K + idx;
    const T *p_theta = grid.Theta + idx;
    const T *p_tg0 = grid.tildeGamma[0] + idx;
    const T *p_tg1 = grid.tildeGamma[1] + idx;
    const T *p_tg2 = grid.tildeGamma[2] + idx;

    const T d_alpha[3] = {d_axis_ptr(p_alpha, sx, inv_60dx, cfg.spatial_order),
                          d_axis_ptr(p_alpha, sy, inv_60dy, cfg.spatial_order),
                          d_axis_ptr(p_alpha, 1, inv_60dz, cfg.spatial_order)};
    const T d_chi[3] = {d_axis_ptr(p_chi, sx, inv_60dx, cfg.spatial_order),
                        d_axis_ptr(p_chi, sy, inv_60dy, cfg.spatial_order),
                        d_axis_ptr(p_chi, 1, inv_60dz, cfg.spatial_order)};

    const T *p_gam[6];
    const T *p_ginv[6];
    const T *p_A[6];
    const T *p_R[6];
    T        gamma_phys[3][3];
    T        gamma_phys_inv[3][3];
    T        gamma_tilde_inv[3][3];
    T        A_mat[3][3];
    T        Ricci_mat[3][3];
    for (int s = 0; s < 6; ++s) {
        p_gam[s] = grid.gamma_tilde[s] + idx;
        p_ginv[s] = grid.gamma_tilde_inv[s] + idx;
        p_A[s] = grid.A_tilde[s] + idx;
        p_R[s] = grid.Ricci[s] + idx;

        const int row = sym_row[s];
        const int col = sym_col[s];
        const T gt = p_gam[s][0];
        const T gt_inv = p_ginv[s][0];
        const T Aval = p_A[s][0];
        const T Rval = p_R[s][0];

        gamma_tilde_inv[row][col] = gt_inv;
        gamma_tilde_inv[col][row] = gt_inv;

        const T g_phys = gt * inv_chi;
        const T g_phys_inv = gt_inv * chi;
        gamma_phys[row][col] = g_phys;
        gamma_phys[col][row] = g_phys;
        gamma_phys_inv[row][col] = g_phys_inv;
        gamma_phys_inv[col][row] = g_phys_inv;

        A_mat[row][col] = Aval;
        A_mat[col][row] = Aval;
        Ricci_mat[row][col] = Rval;
        Ricci_mat[col][row] = Rval;
    }

    const T g_xx = p_ginv[0][0];
    const T g_xy = p_ginv[1][0];
    const T g_xz = p_ginv[2][0];
    const T g_yy = p_ginv[3][0];
    const T g_yz = p_ginv[4][0];
    const T g_zz = p_ginv[5][0];

    const T A_xx = p_A[0][0];
    const T A_xy = p_A[1][0];
    const T A_xz = p_A[2][0];
    const T A_yy = p_A[3][0];
    const T A_yz = p_A[4][0];
    const T A_zz = p_A[5][0];

    const T row0_x = g_xx;
    const T row0_y = g_xy;
    const T row0_z = g_xz;
    const T row1_x = g_xy;
    const T row1_y = g_yy;
    const T row1_z = g_yz;
    const T row2_x = g_xz;
    const T row2_y = g_yz;
    const T row2_z = g_zz;

    const T tmp0_x = A_xx * row0_x + A_xy * row0_y + A_xz * row0_z;
    const T tmp0_y = A_xy * row0_x + A_yy * row0_y + A_yz * row0_z;
    const T tmp0_z = A_xz * row0_x + A_yz * row0_y + A_zz * row0_z;

    const T tmp1_x = A_xx * row1_x + A_xy * row1_y + A_xz * row1_z;
    const T tmp1_y = A_xy * row1_x + A_yy * row1_y + A_yz * row1_z;
    const T tmp1_z = A_xz * row1_x + A_yz * row1_y + A_zz * row1_z;

    const T tmp2_x = A_xx * row2_x + A_xy * row2_y + A_xz * row2_z;
    const T tmp2_y = A_xy * row2_x + A_yy * row2_y + A_yz * row2_z;
    const T tmp2_z = A_xz * row2_x + A_yz * row2_y + A_zz * row2_z;

    const T A_up_xx = row0_x * tmp0_x + row0_y * tmp0_y + row0_z * tmp0_z;
    const T A_up_xy = row0_x * tmp1_x + row0_y * tmp1_y + row0_z * tmp1_z;
    const T A_up_xz = row0_x * tmp2_x + row0_y * tmp2_y + row0_z * tmp2_z;
    const T A_up_yy = row1_x * tmp1_x + row1_y * tmp1_y + row1_z * tmp1_z;
    const T A_up_yz = row1_x * tmp2_x + row1_y * tmp2_y + row1_z * tmp2_z;
    const T A_up_zz = row2_x * tmp2_x + row2_y * tmp2_y + row2_z * tmp2_z;

    const T A_contract = A_xx * A_up_xx + A_yy * A_up_yy + A_zz * A_up_zz +
                         T(2) * (A_xy * A_up_xy + A_xz * A_up_xz + A_yz * A_up_yz);

    T z_over_chi0 = T(0);
    T z_over_chi1 = T(0);
    T z_over_chi2 = T(0);
    recover_z_over_chi_components_from_gamma_ptr(
        p_tg0, p_tg1, p_tg2, p_ginv[0], p_ginv[1], p_ginv[2], p_ginv[3], p_ginv[4], p_ginv[5],
        sx, sy, inv_60dx, inv_60dy, inv_60dz, cfg.spatial_order, z_over_chi0, z_over_chi1,
        z_over_chi2);
    const T z_phys0 = chi_guarded * z_over_chi0;
    const T z_phys1 = chi_guarded * z_over_chi1;
    const T z_phys2 = chi_guarded * z_over_chi2;

    const T R_conformal_base =
        g_xx * Ricci_mat[0][0] + g_yy * Ricci_mat[1][1] + g_zz * Ricci_mat[2][2] +
        T(2) * (g_xy * Ricci_mat[0][1] + g_xz * Ricci_mat[0][2] + g_yz * Ricci_mat[1][2]);
    const T R_conformal_z4 =
        z4_conformal_trace ? z4_conformal_trace[idx]
                           : -(z_over_chi0 * d_chi[0] + z_over_chi1 * d_chi[1] +
                               z_over_chi2 * d_chi[2]) /
                                 chi_guarded;
    const T R_scalar = chi_guarded * (R_conformal_base + R_conformal_z4);

    const T adv_theta = bx * upwind_axis_ptr(p_theta, sx, inv_2dx, bx) +
                        by * upwind_axis_ptr(p_theta, sy, inv_2dy, by) +
                        bz * upwind_axis_ptr(p_theta, 1, inv_2dz, bz);
    const T geom_source = T(0.5) *
                          (R_scalar - A_contract + T(2.0 / 3.0) * K_val * K_val -
                           T(2) * theta * K_val);
    const T Z_dot_dalpha = z_phys0 * d_alpha[0] + z_phys1 * d_alpha[1] + z_phys2 * d_alpha[2];
    const T damping =
        -kappa1_times_lapse_device(alpha, cfg) * (T(2) + cfg.kappa2) * theta;
    const T diss_theta =
        ko6_axis_ptr(p_theta, sx) + ko6_axis_ptr(p_theta, sy) + ko6_axis_ptr(p_theta, 1);
    rhs.Theta[idx] =
        alpha * geom_source + damping + adv_theta - Z_dot_dalpha + ko_scale * diss_theta;

    T d_g_phys[3][3][3];
    for (int s = 0; s < 6; ++s) {
        const int row = sym_row[s];
        const int col = sym_col[s];
        const T d_gt_x = d_axis_ptr(p_gam[s], sx, inv_60dx, cfg.spatial_order);
        const T d_gt_y = d_axis_ptr(p_gam[s], sy, inv_60dy, cfg.spatial_order);
        const T d_gt_z = d_axis_ptr(p_gam[s], 1, inv_60dz, cfg.spatial_order);
        const T gp = gamma_phys[row][col];
        const T val_x = (d_gt_x - gp * d_chi[0]) * inv_chi;
        const T val_y = (d_gt_y - gp * d_chi[1]) * inv_chi;
        const T val_z = (d_gt_z - gp * d_chi[2]) * inv_chi;
        d_g_phys[0][row][col] = val_x;
        d_g_phys[0][col][row] = val_x;
        d_g_phys[1][row][col] = val_y;
        d_g_phys[1][col][row] = val_y;
        d_g_phys[2][row][col] = val_z;
        d_g_phys[2][col][row] = val_z;
    }

    T Gamma_conn[3][3][3];
    for (int up = 0; up < 3; ++up) {
        for (int row = 0; row < 3; ++row) {
            for (int col = row; col < 3; ++col) {
                T sum = T(0);
                for (int l = 0; l < 3; ++l) {
                    sum += gamma_phys_inv[up][l] *
                           (d_g_phys[row][col][l] + d_g_phys[col][row][l] -
                            d_g_phys[l][row][col]);
                }
                const T val = T(0.5) * sum;
                Gamma_conn[up][row][col] = val;
                Gamma_conn[up][col][row] = val;
            }
        }
    }

    T hess[3][3];
    hess[0][0] = d2_axis_ptr(p_alpha, sx, inv_180dx2, cfg.spatial_order);
    hess[1][1] = d2_axis_ptr(p_alpha, sy, inv_180dy2, cfg.spatial_order);
    hess[2][2] = d2_axis_ptr(p_alpha, 1, inv_180dz2, cfg.spatial_order);
    hess[0][1] = dxy4_ptr(p_alpha, sx, sy, inv_144dxdy);
    hess[0][2] = dxz4_ptr(p_alpha, sx, inv_144dxdz);
    hess[1][2] = dyz4_ptr(p_alpha, sy, inv_144dydz);
    hess[1][0] = hess[0][1];
    hess[2][0] = hess[0][2];
    hess[2][1] = hess[1][2];

    T laplacian = T(0);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            T conn_grad = T(0);
            for (int l = 0; l < 3; ++l)
                conn_grad += Gamma_conn[l][row][col] * d_alpha[l];
            laplacian += gamma_phys_inv[row][col] * (hess[row][col] - conn_grad);
        }
    }

    const T dKhat_x = upwind_axis_ptr(p_K, sx, inv_2dx, bx) -
                      T(2) * upwind_axis_ptr(p_theta, sx, inv_2dx, bx);
    const T dKhat_y = upwind_axis_ptr(p_K, sy, inv_2dy, by) -
                      T(2) * upwind_axis_ptr(p_theta, sy, inv_2dy, by);
    const T dKhat_z = upwind_axis_ptr(p_K, 1, inv_2dz, bz) -
                      T(2) * upwind_axis_ptr(p_theta, 1, inv_2dz, bz);
    const T adv_khat = bx * dKhat_x + by * dKhat_y + bz * dKhat_z;
    const T quad = alpha * (A_contract + T(1.0 / 3.0) * K_val * K_val);
    const T z4c_term =
        kappa1_times_lapse_device(alpha, cfg) * (T(1) - cfg.kappa2) * theta;
    const T diss_khat = (ko6_axis_ptr(p_K, sx) - T(2) * ko6_axis_ptr(p_theta, sx)) +
                        (ko6_axis_ptr(p_K, sy) - T(2) * ko6_axis_ptr(p_theta, sy)) +
                        (ko6_axis_ptr(p_K, 1) - T(2) * ko6_axis_ptr(p_theta, 1));
    rhs.K[idx] = adv_khat - laplacian + quad + z4c_term + ko_scale * diss_khat;

    rhs.Z[0][idx] = T(0);
    rhs.Z[1][idx] = T(0);
    rhs.Z[2][idx] = T(0);
}

template <typename T> __device__ __forceinline__ T smooth_floor_device(T val, T floor) {
    const T delta = T(1.0e-10);
    return T(0.5) * (val + floor + sqrt((val - floor) * (val - floor) + delta));
}

template <typename T> __device__ __forceinline__ T khat_device(T K, T Theta) {
    return K - T(2) * Theta;
}

template <typename T> __device__ __forceinline__ T gamma_tilde_target_device(int component) {
    return (component == XX || component == YY || component == ZZ) ? T(1) : T(0);
}

template <typename T>
__device__ __forceinline__ T sponge_profile_value_device(std::size_t layer,
                                                         RHSSpongeCudaConfig<T> cfg) {
    if (layer >= cfg.width || cfg.width == 0 || cfg.strength <= T(0))
        return T(0);
    const double remaining =
        static_cast<double>(cfg.width - layer) / static_cast<double>(cfg.width);
    const double exponent = max(static_cast<double>(cfg.exponent), 1.0);
    return T(static_cast<double>(cfg.strength) * pow(remaining, exponent));
}

template <typename T>
__device__ __forceinline__ bool nearest_sponge_layer_device(StageGridPack<const T> grid,
                                                            std::size_t i, std::size_t j,
                                                            std::size_t k,
                                                            RHSSpongeCudaConfig<T> cfg,
                                                            std::size_t &layer_out) {
    if (!cfg.enabled || cfg.width == 0 || cfg.strength <= T(0))
        return false;

    const std::size_t i0 = grid.dims.ng;
    const std::size_t i1 = grid.dims.ng + grid.dims.nx;
    const std::size_t j0 = grid.dims.ng;
    const std::size_t j1 = grid.dims.ng + grid.dims.ny;
    const std::size_t k0 = grid.dims.ng;
    const std::size_t k1 = grid.dims.ng + grid.dims.nz;
    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return false;

    const std::size_t width_x = min(cfg.width, i1 - i0);
    const std::size_t width_y = min(cfg.width, j1 - j0);
    const std::size_t width_z = min(cfg.width, k1 - k0);

    std::size_t best = cfg.width;
    const auto consider = [&](bool enabled, std::size_t dist, std::size_t width) {
        if (!enabled || dist >= width)
            return;
        best = min(best, dist);
    };

    consider(cfg.active_face[0][0] && !cfg.reflective_face[0][0], i - i0, width_x);
    consider(cfg.active_face[0][1] && !cfg.reflective_face[0][1], (i1 - 1) - i, width_x);
    consider(cfg.active_face[1][0] && !cfg.reflective_face[1][0], j - j0, width_y);
    consider(cfg.active_face[1][1] && !cfg.reflective_face[1][1], (j1 - 1) - j, width_y);
    consider(cfg.active_face[2][0] && !cfg.reflective_face[2][0], k - k0, width_z);
    consider(cfg.active_face[2][1] && !cfg.reflective_face[2][1], (k1 - 1) - k, width_z);

    if (best >= cfg.width)
        return false;
    layer_out = best;
    return true;
}

template <typename T>
__device__ __forceinline__ T normal_derivative_device(const T *p, ptrdiff_t stride,
                                                      std::size_t pos, std::size_t lo,
                                                      std::size_t hi, T inv_h, T inv_2h,
                                                      bool outer) {
    if (!outer) {
        if (pos == lo && lo + 2 < hi)
            return -((-T(1.5) * p[0] + T(2) * p[stride] - T(0.5) * p[2 * stride]) * inv_h);
    } else if (pos + 1 == hi && lo + 2 < hi) {
        return (T(1.5) * p[0] - T(2) * p[-stride] + T(0.5) * p[-2 * stride]) * inv_h;
    }

    return (outer ? T(1) : T(-1)) * (p[stride] - p[-stride]) * inv_2h;
}

template <typename T>
__device__ __forceinline__ void blend_rhs_value(T &slot, T boundary_value, T collar_weight) {
    if (!isfinite(static_cast<double>(slot))) {
        slot = boundary_value;
        return;
    }
    slot = (T(1) - collar_weight) * slot + collar_weight * boundary_value;
}

template <typename T>
__global__ void copy_stage_reference_kernel(StageGridPack<const T> src, StageGridPack<T> dst,
                                            Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    copy_state_at(src, dst, src.idx(i, j, k));
}

template <typename T>
__global__ void copy_stage_reference_kernel(StageGridPack<const T> src, StageRhsPack<T> dst,
                                            Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    copy_state_at(src, dst, src.idx(i, j, k));
}

template <typename T>
__global__ void accumulate_stage_reference_kernel(StageGridPack<const T> src, StageGridPack<T> dst,
                                                  T delta, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    accumulate_state_at(src, dst, delta, src.idx(i, j, k));
}

template <typename T>
__global__ void accumulate_stage_reference_kernel(StageGridPack<const T> src, StageRhsPack<T> dst,
                                                  T delta, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    accumulate_state_at(src, dst, delta, src.idx(i, j, k));
}

template <typename T>
__global__ void compute_gauge_rhs_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                         GaugeRHSCudaConfig<T> cfg, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    compute_gauge_rhs_at(grid, rhs, cfg, i, j, k, grid.idx(i, j, k));
}

template <typename T>
__global__ void compute_scalar_rhs_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                          const T *z4_conformal_trace,
                                          ScalarRHSCudaConfig<T> cfg, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    compute_scalar_rhs_at(grid, rhs, z4_conformal_trace, cfg, i, j, k, grid.idx(i, j, k));
}

template <typename T>
__global__ void explicit_stage_update_kernel(StageGridPack<T> grid, StageGridPack<const T> stage,
                                             StageRhsPack<const T> rhs, T gam0, T gam1, T beta_dt,
                                             Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    explicit_stage_update_at(grid, stage, rhs, gam0, gam1, beta_dt, grid.idx(i, j, k));
}

template <typename T>
__global__ void explicit_stage_update_kernel(StageGridPack<T> grid, StageRhsPack<const T> stage,
                                             StageRhsPack<const T> rhs, T gam0, T gam1, T beta_dt,
                                             Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    explicit_stage_update_at(grid, stage, rhs, gam0, gam1, beta_dt, grid.idx(i, j, k));
}

template <typename T>
__global__ void enforce_algebraic_constraints_kernel(StageGridPack<T> grid, std::size_t total) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    constexpr double min_det_guard = 2.2250738585072014e-308;

    double gxx = static_cast<double>(grid.gamma_tilde[XX][idx]);
    double gxy = static_cast<double>(grid.gamma_tilde[XY][idx]);
    double gxz = static_cast<double>(grid.gamma_tilde[XZ][idx]);
    double gyy = static_cast<double>(grid.gamma_tilde[YY][idx]);
    double gyz = static_cast<double>(grid.gamma_tilde[YZ][idx]);
    double gzz = static_cast<double>(grid.gamma_tilde[ZZ][idx]);

    double det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
                 gxz * (gxy * gyz - gxz * gyy);
    if (det <= min_det_guard)
        det = min_det_guard;
    const double renorm = 1.0 / cbrt(det);
    gxx *= renorm;
    gxy *= renorm;
    gxz *= renorm;
    gyy *= renorm;
    gyz *= renorm;
    gzz *= renorm;

    const double det_scaled = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
                              gxz * (gxy * gyz - gxz * gyy);
    const double inv_det = (det_scaled > min_det_guard) ? (1.0 / det_scaled) : 0.0;

    const double gixx = (gyy * gzz - gyz * gyz) * inv_det;
    const double gixy = (gxz * gyz - gxy * gzz) * inv_det;
    const double gixz = (gxy * gyz - gxz * gyy) * inv_det;
    const double giyy = (gxx * gzz - gxz * gxz) * inv_det;
    const double giyz = (gxy * gxz - gxx * gyz) * inv_det;
    const double gizz = (gxx * gyy - gxy * gxy) * inv_det;

    double Axx = static_cast<double>(grid.A_tilde[XX][idx]);
    double Axy = static_cast<double>(grid.A_tilde[XY][idx]);
    double Axz = static_cast<double>(grid.A_tilde[XZ][idx]);
    double Ayy = static_cast<double>(grid.A_tilde[YY][idx]);
    double Ayz = static_cast<double>(grid.A_tilde[YZ][idx]);
    double Azz = static_cast<double>(grid.A_tilde[ZZ][idx]);

    const double trace =
        gixx * Axx + giyy * Ayy + gizz * Azz + 2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
    const double one_third_trace = trace / 3.0;
    Axx -= gxx * one_third_trace;
    Axy -= gxy * one_third_trace;
    Axz -= gxz * one_third_trace;
    Ayy -= gyy * one_third_trace;
    Ayz -= gyz * one_third_trace;
    Azz -= gzz * one_third_trace;

    grid.gamma_tilde[XX][idx] = static_cast<T>(gxx);
    grid.gamma_tilde[XY][idx] = static_cast<T>(gxy);
    grid.gamma_tilde[XZ][idx] = static_cast<T>(gxz);
    grid.gamma_tilde[YY][idx] = static_cast<T>(gyy);
    grid.gamma_tilde[YZ][idx] = static_cast<T>(gyz);
    grid.gamma_tilde[ZZ][idx] = static_cast<T>(gzz);

    grid.gamma_tilde_inv[XX][idx] = static_cast<T>(gixx);
    grid.gamma_tilde_inv[XY][idx] = static_cast<T>(gixy);
    grid.gamma_tilde_inv[XZ][idx] = static_cast<T>(gixz);
    grid.gamma_tilde_inv[YY][idx] = static_cast<T>(giyy);
    grid.gamma_tilde_inv[YZ][idx] = static_cast<T>(giyz);
    grid.gamma_tilde_inv[ZZ][idx] = static_cast<T>(gizz);

    grid.A_tilde[XX][idx] = static_cast<T>(Axx);
    grid.A_tilde[XY][idx] = static_cast<T>(Axy);
    grid.A_tilde[XZ][idx] = static_cast<T>(Axz);
    grid.A_tilde[YY][idx] = static_cast<T>(Ayy);
    grid.A_tilde[YZ][idx] = static_cast<T>(Ayz);
    grid.A_tilde[ZZ][idx] = static_cast<T>(Azz);
}

template <typename T> __global__ void alpha_floor_kernel(BSSNGridView<T> grid, T floor, std::size_t total) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    grid.alpha.ptr()[idx] = smooth_floor_device(grid.alpha.ptr()[idx], floor);
}

template <typename T> __global__ void chi_floor_kernel(BSSNGridView<T> grid, T floor, std::size_t total) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    grid.chi.ptr()[idx] = smooth_floor_device(grid.chi.ptr()[idx], floor);
}

template <typename T>
__global__ void apply_rhs_sommerfeld_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                            RHSSommerfeldCudaConfig<T> cfg, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;

    const std::size_t dist_ix1 = i - bounds.i0;
    const std::size_t dist_ox1 = (bounds.i1 - 1) - i;
    const std::size_t dist_ix2 = j - bounds.j0;
    const std::size_t dist_ox2 = (bounds.j1 - 1) - j;
    const std::size_t dist_ix3 = k - bounds.k0;
    const std::size_t dist_ox3 = (bounds.k1 - 1) - k;

    const bool use_face[6] = {
        cfg.face_enabled[0][0] && dist_ix1 < cfg.collar_width,
        cfg.face_enabled[0][1] && dist_ox1 < cfg.collar_width,
        cfg.face_enabled[1][0] && dist_ix2 < cfg.collar_width,
        cfg.face_enabled[1][1] && dist_ox2 < cfg.collar_width,
        cfg.face_enabled[2][0] && dist_ix3 < cfg.collar_width,
        cfg.face_enabled[2][1] && dist_ox3 < cfg.collar_width,
    };
    const std::size_t face_layer[6] = {dist_ix1, dist_ox1, dist_ix2, dist_ox2, dist_ix3,
                                       dist_ox3};

    std::size_t nearest_layer = cfg.collar_width;
    for (int face = 0; face < 6; ++face) {
        if (use_face[face])
            nearest_layer = min(nearest_layer, face_layer[face]);
    }
    if (nearest_layer >= cfg.collar_width)
        return;

    const std::size_t idx = grid.idx(i, j, k);
    const T x = grid.x0 + T(i - grid.dims.ng) * grid.dx;
    const T y = grid.y0 + T(j - grid.dims.ng) * grid.dy;
    const T z = grid.z0 + T(k - grid.dims.ng) * grid.dz;
    const T r = sqrt(x * x + y * y + z * z);
    const T inv_r = T(1) / max(r, T(1.0e-12));
    const T inv_dx = T(1) / grid.dx;
    const T inv_dy = T(1) / grid.dy;
    const T inv_dz = T(1) / grid.dz;
    const T inv_2dx = T(0.5) / grid.dx;
    const T inv_2dy = T(0.5) / grid.dy;
    const T inv_2dz = T(0.5) / grid.dz;

    T accum_alpha = T(0);
    T accum_chi = T(0);
    T accum_K = T(0);
    T accum_Theta = T(0);
    T accum_beta[3] = {T(0), T(0), T(0)};
    T accum_B[3] = {T(0), T(0), T(0)};
    T accum_tildeGamma[3] = {T(0), T(0), T(0)};
    T accum_Z[3] = {T(0), T(0), T(0)};
    T accum_gamma_tilde[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};
    T accum_A_tilde[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};
    int face_count = 0;

    const auto accumulate_face = [&](int axis, bool outer) {
        const ptrdiff_t stride = (axis == 0) ? grid.st.sx : (axis == 1 ? grid.st.sy : ptrdiff_t(1));
        const std::size_t pos = (axis == 0) ? i : (axis == 1 ? j : k);
        const std::size_t lo = (axis == 0) ? bounds.i0 : (axis == 1 ? bounds.j0 : bounds.k0);
        const std::size_t hi = (axis == 0) ? bounds.i1 : (axis == 1 ? bounds.j1 : bounds.k1);
        const T inv_h = (axis == 0) ? inv_dx : (axis == 1 ? inv_dy : inv_dz);
        const T inv_2h = (axis == 0) ? inv_2dx : (axis == 1 ? inv_2dy : inv_2dz);
        ++face_count;

        const auto outgoing_rhs = [&](const T *field, T asymptotic, T speed) -> T {
            const T *p = field + idx;
            return -speed * (normal_derivative_device(p, stride, pos, lo, hi, inv_h, inv_2h,
                                                      outer) +
                             (field[idx] - asymptotic) * inv_r);
        };

        accum_alpha += outgoing_rhs(grid.alpha, T(1), cfg.alpha_speed);
        accum_chi += outgoing_rhs(grid.chi, T(1), cfg.chi_speed);

        for (int a = 0; a < 3; ++a) {
            accum_beta[a] += outgoing_rhs(grid.beta[a], T(0), cfg.beta_speed);
            accum_B[a] += outgoing_rhs(grid.B[a], T(0), cfg.B_speed);
            accum_tildeGamma[a] += outgoing_rhs(grid.tildeGamma[a], T(0), cfg.tildeGamma_speed);
            accum_Z[a] += outgoing_rhs(grid.Z[a], T(0), cfg.Z_speed);
        }

        for (int s = 0; s < 6; ++s) {
            accum_gamma_tilde[s] +=
                outgoing_rhs(grid.gamma_tilde[s], gamma_tilde_target_device<T>(s),
                             cfg.gamma_tilde_speed);
            accum_A_tilde[s] += outgoing_rhs(grid.A_tilde[s], T(0), cfg.A_tilde_speed);
        }

        accum_Theta += outgoing_rhs(grid.Theta, T(0), cfg.Theta_speed);

        const T *p_K = grid.K + idx;
        const T *p_Theta = grid.Theta + idx;
        const T khat = khat_device(grid.K[idx], grid.Theta[idx]);
        const T d_khat =
            normal_derivative_device(p_K, stride, pos, lo, hi, inv_h, inv_2h, outer) -
            T(2) * normal_derivative_device(p_Theta, stride, pos, lo, hi, inv_h, inv_2h, outer);
        accum_K += -cfg.K_speed * (d_khat + khat * inv_r);
    };

    if (use_face[0] && face_layer[0] == nearest_layer)
        accumulate_face(0, false);
    if (use_face[1] && face_layer[1] == nearest_layer)
        accumulate_face(0, true);
    if (use_face[2] && face_layer[2] == nearest_layer)
        accumulate_face(1, false);
    if (use_face[3] && face_layer[3] == nearest_layer)
        accumulate_face(1, true);
    if (use_face[4] && face_layer[4] == nearest_layer)
        accumulate_face(2, false);
    if (use_face[5] && face_layer[5] == nearest_layer)
        accumulate_face(2, true);

    if (face_count == 0)
        return;

    const T inv_faces = T(1) / T(face_count);
    const double x_weight = static_cast<double>(cfg.collar_width - nearest_layer) /
                            static_cast<double>(cfg.collar_width);
    const T collar_weight =
        T(max(0.0, min(1.0, x_weight * x_weight * (3.0 - 2.0 * x_weight))));

    blend_rhs_value(rhs.alpha[idx], accum_alpha * inv_faces, collar_weight);
    blend_rhs_value(rhs.chi[idx], accum_chi * inv_faces, collar_weight);
    blend_rhs_value(rhs.K[idx], accum_K * inv_faces, collar_weight);
    blend_rhs_value(rhs.Theta[idx], accum_Theta * inv_faces, collar_weight);
    for (int a = 0; a < 3; ++a) {
        blend_rhs_value(rhs.beta[a][idx], accum_beta[a] * inv_faces, collar_weight);
        blend_rhs_value(rhs.B[a][idx], accum_B[a] * inv_faces, collar_weight);
        blend_rhs_value(rhs.tildeGamma[a][idx], accum_tildeGamma[a] * inv_faces, collar_weight);
        blend_rhs_value(rhs.Z[a][idx], accum_Z[a] * inv_faces, collar_weight);
    }
    for (int s = 0; s < 6; ++s) {
        blend_rhs_value(rhs.gamma_tilde[s][idx], accum_gamma_tilde[s] * inv_faces, collar_weight);
        blend_rhs_value(rhs.A_tilde[s][idx], accum_A_tilde[s] * inv_faces, collar_weight);
    }
}

template <typename T>
__global__ void apply_rhs_sponge_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                        RHSSpongeCudaConfig<T> cfg, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;

    std::size_t layer = 0;
    if (!nearest_sponge_layer_device(grid, i, j, k, cfg, layer))
        return;

    const T sigma = sponge_profile_value_device(layer, cfg);
    if (sigma <= T(0))
        return;

    const std::size_t idx = grid.idx(i, j, k);
    rhs.alpha[idx] += sigma * (T(1) - grid.alpha[idx]);
    rhs.chi[idx] += sigma * (T(1) - grid.chi[idx]);
    rhs.K[idx] += sigma * (T(0) - khat_device(grid.K[idx], grid.Theta[idx]));
    rhs.Theta[idx] += sigma * (T(0) - grid.Theta[idx]);
    for (int a = 0; a < 3; ++a) {
        rhs.beta[a][idx] += sigma * (T(0) - grid.beta[a][idx]);
        rhs.B[a][idx] += sigma * (T(0) - grid.B[a][idx]);
        rhs.tildeGamma[a][idx] += sigma * (T(0) - grid.tildeGamma[a][idx]);
        rhs.Z[a][idx] += sigma * (T(0) - grid.Z[a][idx]);
    }
    for (int s = 0; s < 6; ++s) {
        rhs.gamma_tilde[s][idx] += sigma * (gamma_tilde_target_device<T>(s) - grid.gamma_tilde[s][idx]);
        rhs.A_tilde[s][idx] += sigma * (T(0) - grid.A_tilde[s][idx]);
    }
}

template <typename T>
__global__ void stabilize_nonfinite_rhs_collar_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                                      std::size_t collar_width, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;

    const bool in_collar = (i - bounds.i0 < collar_width) || ((bounds.i1 - 1) - i < collar_width) ||
                           (j - bounds.j0 < collar_width) || ((bounds.j1 - 1) - j < collar_width) ||
                           (k - bounds.k0 < collar_width) || ((bounds.k1 - 1) - k < collar_width);
    if (!in_collar)
        return;

    const std::size_t idx = grid.idx(i, j, k);
    const auto reset_nonfinite = [&](T &slot) {
        if (!isfinite(static_cast<double>(slot)))
            slot = T(0);
    };
    reset_nonfinite(rhs.alpha[idx]);
    reset_nonfinite(rhs.chi[idx]);
    reset_nonfinite(rhs.K[idx]);
    reset_nonfinite(rhs.Theta[idx]);
    for (int c = 0; c < 3; ++c) {
        reset_nonfinite(rhs.beta[c][idx]);
        reset_nonfinite(rhs.B[c][idx]);
        reset_nonfinite(rhs.tildeGamma[c][idx]);
        reset_nonfinite(rhs.Z[c][idx]);
    }
    for (int s = 0; s < 6; ++s) {
        reset_nonfinite(rhs.gamma_tilde[s][idx]);
        reset_nonfinite(rhs.A_tilde[s][idx]);
    }
}

template <typename T>
__global__ void recompose_rhs_K_from_khat_kernel(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                                 Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    const std::size_t idx = grid.idx(i, j, k);
    rhs.K[idx] += T(2) * rhs.Theta[idx];
}

inline void throw_last_error(const char *context) {
    tensorium::cuda::throw_if_error(cudaGetLastError(), context);
}

template <typename T>
void launch_copy_stage_reference(StageGridPack<const T> src, StageGridPack<T> dst,
                                 const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    copy_stage_reference_kernel<<<launch_grid, block>>>(src, dst, bounds);
    throw_last_error("copy_stage_reference_kernel");
}

template <typename T>
void launch_copy_stage_reference(StageGridPack<const T> src, StageRhsPack<T> dst,
                                 const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    copy_stage_reference_kernel<<<launch_grid, block>>>(src, dst, bounds);
    throw_last_error("copy_stage_reference_kernel");
}

template <typename T>
void launch_accumulate_stage_reference(StageGridPack<const T> src, StageGridPack<T> dst, T delta,
                                       const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    accumulate_stage_reference_kernel<<<launch_grid, block>>>(src, dst, delta, bounds);
    throw_last_error("accumulate_stage_reference_kernel");
}

template <typename T>
void launch_accumulate_stage_reference(StageGridPack<const T> src, StageRhsPack<T> dst, T delta,
                                       const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    accumulate_stage_reference_kernel<<<launch_grid, block>>>(src, dst, delta, bounds);
    throw_last_error("accumulate_stage_reference_kernel");
}

template <typename T>
void launch_compute_gauge_rhs(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                              GaugeRHSCudaConfig<T> cfg, const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    compute_gauge_rhs_kernel<<<launch_grid, block>>>(grid, rhs, cfg, bounds);
    throw_last_error("compute_gauge_rhs_kernel");
}

template <typename T>
void launch_compute_scalar_rhs(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                               const T *z4_conformal_trace, ScalarRHSCudaConfig<T> cfg,
                               const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    compute_scalar_rhs_kernel<<<launch_grid, block>>>(grid, rhs, z4_conformal_trace, cfg, bounds);
    throw_last_error("compute_scalar_rhs_kernel");
}

template <typename T>
void launch_apply_rhs_sommerfeld(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                 RHSSommerfeldCudaConfig<T> cfg, const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    apply_rhs_sommerfeld_kernel<<<launch_grid, block>>>(grid, rhs, cfg, bounds);
    throw_last_error("apply_rhs_sommerfeld_kernel");
}

template <typename T>
void launch_apply_rhs_sponge(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                             RHSSpongeCudaConfig<T> cfg, const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    apply_rhs_sponge_kernel<<<launch_grid, block>>>(grid, rhs, cfg, bounds);
    throw_last_error("apply_rhs_sponge_kernel");
}

template <typename T>
void launch_stabilize_nonfinite_rhs_collar(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                           std::size_t collar_width, const Bounds3D &bounds) {
    if (collar_width == 0 || bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    stabilize_nonfinite_rhs_collar_kernel<<<launch_grid, block>>>(grid, rhs, collar_width, bounds);
    throw_last_error("stabilize_nonfinite_rhs_collar_kernel");
}

template <typename T>
void launch_recompose_rhs_K_from_khat(StageGridPack<const T> grid, StageRhsPack<T> rhs,
                                      const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    recompose_rhs_K_from_khat_kernel<<<launch_grid, block>>>(grid, rhs, bounds);
    throw_last_error("recompose_rhs_K_from_khat_kernel");
}

template <typename T>
void launch_explicit_stage_update(StageGridPack<T> grid, StageGridPack<const T> stage,
                                  StageRhsPack<const T> rhs, T gam0, T gam1, T beta_dt,
                                  const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    explicit_stage_update_kernel<<<launch_grid, block>>>(grid, stage, rhs, gam0, gam1, beta_dt,
                                                         bounds);
    throw_last_error("explicit_stage_update_kernel");
}

template <typename T>
void launch_explicit_stage_update(StageGridPack<T> grid, StageRhsPack<const T> stage,
                                  StageRhsPack<const T> rhs, T gam0, T gam1, T beta_dt,
                                  const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    explicit_stage_update_kernel<<<launch_grid, block>>>(grid, stage, rhs, gam0, gam1, beta_dt,
                                                         bounds);
    throw_last_error("explicit_stage_update_kernel");
}

template <typename T> void launch_enforce_algebraic_constraints(StageGridPack<T> grid) {
    const std::size_t total = grid.st.nx_tot * grid.st.ny_tot * grid.st.nz_tot;
    if (total == 0)
        return;
    constexpr std::size_t block_size = 256;
    const std::size_t blocks = (total + block_size - 1) / block_size;
    enforce_algebraic_constraints_kernel<<<static_cast<unsigned>(blocks), block_size>>>(grid, total);
    throw_last_error("enforce_algebraic_constraints_kernel");
}

template <typename T, typename Kernel>
void launch_floor_kernel(BSSNGridView<T> grid, T floor, Kernel kernel, const char *label) {
    const std::size_t total = grid.st.nx_tot * grid.st.ny_tot * grid.st.nz_tot;
    if (total == 0)
        return;
    constexpr std::size_t block_size = 256;
    const std::size_t blocks = (total + block_size - 1) / block_size;
    kernel<<<static_cast<unsigned>(blocks), block_size>>>(grid, floor, total);
    throw_last_error(label);
}

} // namespace

template <typename T>
void copy_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, std::size_t padding) {
    launch_copy_stage_reference(make_stage_pack(src), make_stage_pack(dst), interior_bounds(src, padding));
}

template <typename T>
void copy_stage_reference(BSSNGridView<const T> src, BSSNRHSWorkspaceView<T> dst,
                          std::size_t padding) {
    launch_copy_stage_reference(make_stage_pack(src), make_rhs_pack(dst), interior_bounds(src, padding));
}

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, T delta,
                                std::size_t padding) {
    launch_accumulate_stage_reference(make_stage_pack(src), make_stage_pack(dst), delta,
                                      interior_bounds(src, padding));
}

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNRHSWorkspaceView<T> dst, T delta,
                                std::size_t padding) {
    launch_accumulate_stage_reference(make_stage_pack(src), make_rhs_pack(dst), delta,
                                      interior_bounds(src, padding));
}

template <typename T>
void compute_gauge_rhs(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                       GaugeRHSCudaConfig<T> config, std::size_t padding) {
    launch_compute_gauge_rhs(make_stage_pack(grid), make_rhs_pack(rhs), config,
                             interior_bounds(grid, padding));
}

template <typename T>
void compute_scalar_rhs(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                        Field3DView<const T> z4_conformal_trace, ScalarRHSCudaConfig<T> config,
                        std::size_t padding) {
    launch_compute_scalar_rhs(make_stage_pack(grid), make_rhs_pack(rhs), z4_conformal_trace.ptr(),
                              config, interior_bounds(grid, padding));
}

template <typename T>
void apply_rhs_sommerfeld(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                          RHSSommerfeldCudaConfig<T> config) {
    launch_apply_rhs_sommerfeld(make_stage_pack(grid), make_rhs_pack(rhs), config,
                                domain_bounds(grid));
}

template <typename T>
void apply_rhs_sponge(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                      RHSSpongeCudaConfig<T> config) {
    launch_apply_rhs_sponge(make_stage_pack(grid), make_rhs_pack(rhs), config, domain_bounds(grid));
}

template <typename T>
void stabilize_nonfinite_rhs_collar(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                                    std::size_t collar_width) {
    launch_stabilize_nonfinite_rhs_collar(make_stage_pack(grid), make_rhs_pack(rhs), collar_width,
                                          domain_bounds(grid));
}

template <typename T>
void recompose_rhs_K_from_khat(BSSNGridView<const T> grid, BSSNRHSWorkspaceView<T> rhs,
                               std::size_t padding) {
    launch_recompose_rhs_K_from_khat(make_stage_pack(grid), make_rhs_pack(rhs),
                                     interior_bounds(grid, padding));
}

template <typename T> void enforce_algebraic_constraints(BSSNGridView<T> grid) {
    launch_enforce_algebraic_constraints(make_stage_pack(grid));
}

template <typename T>
void apply_explicit_stage_update(BSSNGridView<T> grid, BSSNGridView<const T> stage,
                                 BSSNRHSWorkspaceView<const T> rhs, T gam0, T gam1, T beta_dt,
                                 std::size_t padding) {
    launch_explicit_stage_update(make_stage_pack(grid), make_stage_pack(stage), make_rhs_pack(rhs),
                                 gam0, gam1, beta_dt, interior_bounds(grid, padding));
}

template <typename T>
void apply_explicit_stage_update(BSSNGridView<T> grid, BSSNRHSWorkspaceView<const T> stage,
                                 BSSNRHSWorkspaceView<const T> rhs, T gam0, T gam1, T beta_dt,
                                 std::size_t padding) {
    launch_explicit_stage_update(make_stage_pack(grid), make_rhs_pack(stage), make_rhs_pack(rhs),
                                 gam0, gam1, beta_dt, interior_bounds(grid, padding));
}

template <typename T> void apply_alpha_floor(BSSNGridView<T> grid, T floor) {
    launch_floor_kernel(grid, floor, alpha_floor_kernel<T>, "alpha_floor_kernel");
}

template <typename T> void apply_chi_floor(BSSNGridView<T> grid, T floor) {
    launch_floor_kernel(grid, floor, chi_floor_kernel<T>, "chi_floor_kernel");
}

template void copy_stage_reference<float>(BSSNGridView<const float>, BSSNGridView<float>, std::size_t);
template void copy_stage_reference<double>(BSSNGridView<const double>, BSSNGridView<double>,
                                           std::size_t);

template void copy_stage_reference<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>,
                                          std::size_t);
template void copy_stage_reference<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                           std::size_t);

template void accumulate_stage_reference<float>(BSSNGridView<const float>, BSSNGridView<float>, float,
                                                std::size_t);
template void accumulate_stage_reference<double>(BSSNGridView<const double>, BSSNGridView<double>,
                                                 double, std::size_t);

template void
accumulate_stage_reference<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>, float,
                                  std::size_t);
template void
accumulate_stage_reference<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                   double, std::size_t);

template void compute_gauge_rhs<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>,
                                       GaugeRHSCudaConfig<float>, std::size_t);
template void compute_gauge_rhs<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                        GaugeRHSCudaConfig<double>, std::size_t);

template void compute_scalar_rhs<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>,
                                        Field3DView<const float>, ScalarRHSCudaConfig<float>,
                                        std::size_t);
template void compute_scalar_rhs<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                         Field3DView<const double>, ScalarRHSCudaConfig<double>,
                                         std::size_t);

template void apply_rhs_sommerfeld<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>,
                                          RHSSommerfeldCudaConfig<float>);
template void apply_rhs_sommerfeld<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                           RHSSommerfeldCudaConfig<double>);

template void apply_rhs_sponge<float>(BSSNGridView<const float>, BSSNRHSWorkspaceView<float>,
                                      RHSSpongeCudaConfig<float>);
template void apply_rhs_sponge<double>(BSSNGridView<const double>, BSSNRHSWorkspaceView<double>,
                                       RHSSpongeCudaConfig<double>);

template void stabilize_nonfinite_rhs_collar<float>(BSSNGridView<const float>,
                                                    BSSNRHSWorkspaceView<float>, std::size_t);
template void stabilize_nonfinite_rhs_collar<double>(BSSNGridView<const double>,
                                                     BSSNRHSWorkspaceView<double>, std::size_t);

template void recompose_rhs_K_from_khat<float>(BSSNGridView<const float>,
                                               BSSNRHSWorkspaceView<float>, std::size_t);
template void recompose_rhs_K_from_khat<double>(BSSNGridView<const double>,
                                                BSSNRHSWorkspaceView<double>, std::size_t);

template void enforce_algebraic_constraints<float>(BSSNGridView<float>);
template void enforce_algebraic_constraints<double>(BSSNGridView<double>);

template void apply_explicit_stage_update<float>(BSSNGridView<float>, BSSNGridView<const float>,
                                                 BSSNRHSWorkspaceView<const float>, float, float,
                                                 float, std::size_t);
template void apply_explicit_stage_update<double>(BSSNGridView<double>, BSSNGridView<const double>,
                                                  BSSNRHSWorkspaceView<const double>, double,
                                                  double, double, std::size_t);

template void apply_explicit_stage_update<float>(BSSNGridView<float>,
                                                 BSSNRHSWorkspaceView<const float>,
                                                 BSSNRHSWorkspaceView<const float>, float, float,
                                                 float, std::size_t);
template void apply_explicit_stage_update<double>(BSSNGridView<double>,
                                                  BSSNRHSWorkspaceView<const double>,
                                                  BSSNRHSWorkspaceView<const double>, double,
                                                  double, double, std::size_t);

template void apply_alpha_floor<float>(BSSNGridView<float>, float);
template void apply_alpha_floor<double>(BSSNGridView<double>, double);

template void apply_chi_floor<float>(BSSNGridView<float>, float);
template void apply_chi_floor<double>(BSSNGridView<double>, double);

} // namespace tensorium_RG::bssn::cuda

#endif
