#include "BSSNCudaGridOps.hpp"

#ifdef TENSORIUM_CUDA

#include <Tensorium/Backend/CUDA/Core/CudaRuntime.hpp>

#include <cmath>
#include <cstddef>

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

inline dim3 make_launch_grid(std::size_t ni, std::size_t nj, std::size_t nk, const dim3 &block) {
    return dim3(static_cast<unsigned>((ni + block.x - 1) / block.x),
                static_cast<unsigned>((nj + block.y - 1) / block.y),
                static_cast<unsigned>((nk + block.z - 1) / block.z));
}

template <typename T> __device__ __forceinline__ void copy_state_at(BSSNGridView<const T> src,
                                                                    BSSNGridView<T> dst,
                                                                    std::size_t idx) {
    dst.alpha.ptr()[idx] = src.alpha.ptr()[idx];
    dst.chi.ptr()[idx] = src.chi.ptr()[idx];
    dst.K.ptr()[idx] = src.K.ptr()[idx];
    dst.Theta.ptr()[idx] = src.Theta.ptr()[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c].ptr()[idx] = src.beta[c].ptr()[idx];
        dst.B[c].ptr()[idx] = src.B[c].ptr()[idx];
        dst.tildeGamma[c].ptr()[idx] = src.tildeGamma[c].ptr()[idx];
        dst.Z[c].ptr()[idx] = src.Z[c].ptr()[idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s].ptr()[idx] = src.gamma_tilde[s].ptr()[idx];
        dst.A_tilde[s].ptr()[idx] = src.A_tilde[s].ptr()[idx];
    }
}

template <typename T>
__device__ __forceinline__ void accumulate_state_at(BSSNGridView<const T> src, BSSNGridView<T> dst,
                                                    T delta, std::size_t idx) {
    dst.alpha.ptr()[idx] += delta * src.alpha.ptr()[idx];
    dst.chi.ptr()[idx] += delta * src.chi.ptr()[idx];
    dst.K.ptr()[idx] += delta * src.K.ptr()[idx];
    dst.Theta.ptr()[idx] += delta * src.Theta.ptr()[idx];
    for (int c = 0; c < 3; ++c) {
        dst.beta[c].ptr()[idx] += delta * src.beta[c].ptr()[idx];
        dst.B[c].ptr()[idx] += delta * src.B[c].ptr()[idx];
        dst.tildeGamma[c].ptr()[idx] += delta * src.tildeGamma[c].ptr()[idx];
        dst.Z[c].ptr()[idx] += delta * src.Z[c].ptr()[idx];
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s].ptr()[idx] += delta * src.gamma_tilde[s].ptr()[idx];
        dst.A_tilde[s].ptr()[idx] += delta * src.A_tilde[s].ptr()[idx];
    }
}

template <typename T> __device__ __forceinline__ T smooth_floor_device(T val, T floor) {
    const T delta = T(1.0e-10);
    return T(0.5) * (val + floor + sqrt((val - floor) * (val - floor) + delta));
}

template <typename T>
__global__ void copy_stage_reference_kernel(BSSNGridView<const T> src, BSSNGridView<T> dst,
                                            Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    copy_state_at(src, dst, src.alpha.idx(i, j, k));
}

template <typename T>
__global__ void accumulate_stage_reference_kernel(BSSNGridView<const T> src, BSSNGridView<T> dst,
                                                  T delta, Bounds3D bounds) {
    const std::size_t i = bounds.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t j = bounds.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t k = bounds.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bounds.i1 || j >= bounds.j1 || k >= bounds.k1)
        return;
    accumulate_state_at(src, dst, delta, src.alpha.idx(i, j, k));
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

inline void throw_last_error(const char *context) {
    tensorium::cuda::throw_if_error(cudaGetLastError(), context);
}

template <typename T>
void launch_copy_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst,
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
void launch_accumulate_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, T delta,
                                       const Bounds3D &bounds) {
    if (bounds.i0 >= bounds.i1 || bounds.j0 >= bounds.j1 || bounds.k0 >= bounds.k1)
        return;
    const dim3 block(8, 4, 4);
    const dim3 launch_grid =
        make_launch_grid(bounds.i1 - bounds.i0, bounds.j1 - bounds.j0, bounds.k1 - bounds.k0, block);
    accumulate_stage_reference_kernel<<<launch_grid, block>>>(src, dst, delta, bounds);
    throw_last_error("accumulate_stage_reference_kernel");
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
    launch_copy_stage_reference(src, dst, interior_bounds(src, padding));
}

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, T delta,
                                std::size_t padding) {
    launch_accumulate_stage_reference(src, dst, delta, interior_bounds(src, padding));
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

template void accumulate_stage_reference<float>(BSSNGridView<const float>, BSSNGridView<float>, float,
                                                std::size_t);
template void accumulate_stage_reference<double>(BSSNGridView<const double>, BSSNGridView<double>,
                                                 double, std::size_t);

template void apply_alpha_floor<float>(BSSNGridView<float>, float);
template void apply_alpha_floor<double>(BSSNGridView<double>, double);

template void apply_chi_floor<float>(BSSNGridView<float>, float);
template void apply_chi_floor<double>(BSSNGridView<double>, double);

} // namespace tensorium_RG::bssn::cuda

#endif
