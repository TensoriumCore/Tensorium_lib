#pragma once

#include "BSSNGridSoA.hpp"

#include <type_traits>

#ifdef TENSORIUM_CUDA
#    include <cuda_runtime.h>
#    define TENSORIUM_BSSN_HD __host__ __device__
#else
#    define TENSORIUM_BSSN_HD
#endif

namespace tensorium_RG::bssn {

template <typename T> struct Field3DView {
    using value_type = std::remove_const_t<T>;

    T *data = nullptr;
    Strides<value_type> st{};

    TENSORIUM_BSSN_HD T *ptr() noexcept { return data; }
    TENSORIUM_BSSN_HD const T *ptr() const noexcept { return data; }

    TENSORIUM_BSSN_HD size_t idx(size_t i, size_t j, size_t k) const noexcept {
        return i * st.sx + j * st.sy + k * st.sz;
    }
};

template <typename T> struct BSSNGridView {
    using value_type = std::remove_const_t<T>;

    GridDims              dims{};
    Strides<value_type>   st{};
    Field3DView<T>        alpha;
    Field3DView<T>        chi;
    Field3DView<T>        K;
    Field3DView<T>        Theta;
    Field3DView<T>        beta[3];
    Field3DView<T>        B[3];
    Field3DView<T>        tildeGamma[3];
    Field3DView<T>        Z[3];
    Field3DView<T>        gamma_tilde[6];
    Field3DView<T>        gamma_tilde_inv[6];
    Field3DView<T>        A_tilde[6];
    Field3DView<T>        Ricci[6];
    value_type            dx{};
    value_type            dy{};
    value_type            dz{};
    value_type            x0{};
    value_type            y0{};
    value_type            z0{};

    TENSORIUM_BSSN_HD void domain_bounds(size_t &i0, size_t &i1, size_t &j0, size_t &j1,
                                         size_t &k0, size_t &k1) const noexcept {
        i0 = dims.ng;
        i1 = dims.ng + dims.nx;
        j0 = dims.ng;
        j1 = dims.ng + dims.ny;
        k0 = dims.ng;
        k1 = dims.ng + dims.nz;
    }

    TENSORIUM_BSSN_HD void coords(size_t i, size_t j, size_t k, value_type &x, value_type &y,
                                  value_type &z) const noexcept {
        x = x0 + (i - dims.ng) * dx;
        y = y0 + (j - dims.ng) * dy;
        z = z0 + (k - dims.ng) * dz;
    }
};

template <typename T> inline Field3DView<T> make_field_view(Field3D<T> &field) {
    return {field.ptr(), field.st};
}

template <typename T> inline Field3DView<const T> make_field_view(const Field3D<T> &field) {
    return {field.ptr(), field.st};
}

template <typename T> inline BSSNGridView<T> make_view(BSSNGridSoA<T> &grid) {
    BSSNGridView<T> view;
    view.dims = grid.dims;
    view.st = grid.st;
    view.alpha = make_field_view(grid.alpha);
    view.chi = make_field_view(grid.chi);
    view.K = make_field_view(grid.K);
    view.Theta = make_field_view(grid.Theta);
    for (int c = 0; c < 3; ++c) {
        view.beta[c] = make_field_view(grid.beta[c]);
        view.B[c] = make_field_view(grid.B[c]);
        view.tildeGamma[c] = make_field_view(grid.tildeGamma[c]);
        view.Z[c] = make_field_view(grid.Z[c]);
    }
    for (int s = 0; s < 6; ++s) {
        view.gamma_tilde[s] = make_field_view(grid.gamma_tilde[s]);
        view.gamma_tilde_inv[s] = make_field_view(grid.gamma_tilde_inv[s]);
        view.A_tilde[s] = make_field_view(grid.A_tilde[s]);
        view.Ricci[s] = make_field_view(grid.Ricci[s]);
    }
    view.dx = grid.dx;
    view.dy = grid.dy;
    view.dz = grid.dz;
    view.x0 = grid.x0;
    view.y0 = grid.y0;
    view.z0 = grid.z0;
    return view;
}

template <typename T> inline BSSNGridView<const T> make_view(const BSSNGridSoA<T> &grid) {
    BSSNGridView<const T> view;
    view.dims = grid.dims;
    view.st = grid.st;
    view.alpha = make_field_view(grid.alpha);
    view.chi = make_field_view(grid.chi);
    view.K = make_field_view(grid.K);
    view.Theta = make_field_view(grid.Theta);
    for (int c = 0; c < 3; ++c) {
        view.beta[c] = make_field_view(grid.beta[c]);
        view.B[c] = make_field_view(grid.B[c]);
        view.tildeGamma[c] = make_field_view(grid.tildeGamma[c]);
        view.Z[c] = make_field_view(grid.Z[c]);
    }
    for (int s = 0; s < 6; ++s) {
        view.gamma_tilde[s] = make_field_view(grid.gamma_tilde[s]);
        view.gamma_tilde_inv[s] = make_field_view(grid.gamma_tilde_inv[s]);
        view.A_tilde[s] = make_field_view(grid.A_tilde[s]);
        view.Ricci[s] = make_field_view(grid.Ricci[s]);
    }
    view.dx = grid.dx;
    view.dy = grid.dy;
    view.dz = grid.dz;
    view.x0 = grid.x0;
    view.y0 = grid.y0;
    view.z0 = grid.z0;
    return view;
}

} // namespace tensorium_RG::bssn

#undef TENSORIUM_BSSN_HD
