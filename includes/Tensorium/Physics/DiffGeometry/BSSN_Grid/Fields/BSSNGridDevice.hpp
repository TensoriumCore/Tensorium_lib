#pragma once

#include "BSSNGridViews.hpp"

#include <Tensorium/Backend/CUDA/Core/DeviceBuffer.hpp>

#include <cstddef>
#include <stdexcept>

namespace tensorium_RG::bssn {

template <typename T> inline bool strides_match(const Strides<T> &lhs, const Strides<T> &rhs) noexcept {
    return lhs.sx == rhs.sx && lhs.sy == rhs.sy && lhs.sz == rhs.sz && lhs.nx_tot == rhs.nx_tot &&
           lhs.ny_tot == rhs.ny_tot && lhs.nz_tot == rhs.nz_tot;
}

inline bool dims_match(const GridDims &lhs, const GridDims &rhs) noexcept {
    return lhs.nx == rhs.nx && lhs.ny == rhs.ny && lhs.nz == rhs.nz && lhs.ng == rhs.ng;
}

template <typename T> struct DeviceField3D {
    tensorium::cuda::DeviceBuffer<T> data;
    Strides<T>                       st{};

    void allocate_like(const Field3D<T> &field) {
        st = field.st;
        data.resize(elements());
    }

    void allocate_with_strides(const Strides<T> &strides) {
        st = strides;
        data.resize(elements());
    }

    [[nodiscard]] std::size_t elements() const noexcept {
        return st.nx_tot * st.ny_tot * st.nz_tot;
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return data.bytes(); }

    [[nodiscard]] T *ptr() noexcept { return data.data(); }

    [[nodiscard]] const T *ptr() const noexcept { return data.data(); }

    [[nodiscard]] size_t idx(size_t i, size_t j, size_t k) const noexcept {
        return i * st.sx + j * st.sy + k * st.sz;
    }

    void zero() { data.zero(); }

    void copy_from_host(const Field3D<T> &field) {
        if (!strides_match(st, field.st))
            allocate_like(field);
        data.copy_from_host(field.ptr(), elements());
    }

    void copy_to_host(Field3D<T> &field) const {
        const std::size_t host_elements = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
        if (!strides_match(st, field.st) || host_elements != elements()) {
            throw std::invalid_argument("DeviceField3D::copy_to_host layout mismatch");
        }
        data.copy_to_host(field.ptr(), host_elements);
    }

    [[nodiscard]] Field3DView<T> view() noexcept { return {ptr(), st}; }

    [[nodiscard]] Field3DView<const T> view() const noexcept { return {ptr(), st}; }
};

template <typename T> struct BSSNRHSWorkspaceDevice {
    DeviceField3D<T> alpha;
    DeviceField3D<T> chi;
    DeviceField3D<T> K;
    DeviceField3D<T> Theta;
    DeviceField3D<T> beta[3];
    DeviceField3D<T> B[3];
    DeviceField3D<T> gamma_tilde[6];
    DeviceField3D<T> A_tilde[6];
    DeviceField3D<T> tildeGamma[3];
    DeviceField3D<T> Z[3];

    void allocate_like(const BSSNGridSoA<T> &grid) {
        alpha.allocate_like(grid.alpha);
        chi.allocate_like(grid.chi);
        K.allocate_like(grid.K);
        Theta.allocate_like(grid.Theta);
        for (int c = 0; c < 3; ++c) {
            beta[c].allocate_like(grid.beta[c]);
            B[c].allocate_like(grid.B[c]);
            tildeGamma[c].allocate_like(grid.tildeGamma[c]);
            Z[c].allocate_like(grid.Z[c]);
        }
        for (int s = 0; s < 6; ++s) {
            gamma_tilde[s].allocate_like(grid.gamma_tilde[s]);
            A_tilde[s].allocate_like(grid.A_tilde[s]);
        }
    }

    void zero() {
        alpha.zero();
        chi.zero();
        K.zero();
        Theta.zero();
        for (int c = 0; c < 3; ++c) {
            beta[c].zero();
            B[c].zero();
            tildeGamma[c].zero();
            Z[c].zero();
        }
        for (int s = 0; s < 6; ++s) {
            gamma_tilde[s].zero();
            A_tilde[s].zero();
        }
    }

    [[nodiscard]] BSSNRHSWorkspaceView<T> view() noexcept {
        BSSNRHSWorkspaceView<T> out;
        out.st = alpha.st;
        out.alpha = alpha.view();
        out.chi = chi.view();
        out.K = K.view();
        out.Theta = Theta.view();
        for (int c = 0; c < 3; ++c) {
            out.beta[c] = beta[c].view();
            out.B[c] = B[c].view();
            out.tildeGamma[c] = tildeGamma[c].view();
            out.Z[c] = Z[c].view();
        }
        for (int s = 0; s < 6; ++s) {
            out.gamma_tilde[s] = gamma_tilde[s].view();
            out.A_tilde[s] = A_tilde[s].view();
        }
        return out;
    }

    [[nodiscard]] BSSNRHSWorkspaceView<const T> view() const noexcept {
        BSSNRHSWorkspaceView<const T> out;
        out.st = alpha.st;
        out.alpha = alpha.view();
        out.chi = chi.view();
        out.K = K.view();
        out.Theta = Theta.view();
        for (int c = 0; c < 3; ++c) {
            out.beta[c] = beta[c].view();
            out.B[c] = B[c].view();
            out.tildeGamma[c] = tildeGamma[c].view();
            out.Z[c] = Z[c].view();
        }
        for (int s = 0; s < 6; ++s) {
            out.gamma_tilde[s] = gamma_tilde[s].view();
            out.A_tilde[s] = A_tilde[s].view();
        }
        return out;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        std::size_t total = alpha.bytes() + chi.bytes() + K.bytes() + Theta.bytes();
        for (int c = 0; c < 3; ++c)
            total += beta[c].bytes() + B[c].bytes() + tildeGamma[c].bytes() + Z[c].bytes();
        for (int s = 0; s < 6; ++s)
            total += gamma_tilde[s].bytes() + A_tilde[s].bytes();
        return total;
    }
};

template <typename T> struct BSSNGridDevice {
    GridDims         dims{};
    Strides<T>       st{};
    DeviceField3D<T> alpha;
    DeviceField3D<T> chi;
    DeviceField3D<T> K;
    DeviceField3D<T> Theta;
    DeviceField3D<T> beta[3];
    DeviceField3D<T> B[3];
    DeviceField3D<T> tildeGamma[3];
    DeviceField3D<T> Z[3];
    DeviceField3D<T> gamma_tilde[6];
    DeviceField3D<T> gamma_tilde_inv[6];
    DeviceField3D<T> A_tilde[6];
    DeviceField3D<T> Ricci[6];
    T                dx{};
    T                dy{};
    T                dz{};
    T                x0{};
    T                y0{};
    T                z0{};

    BSSNGridDevice() = default;

    explicit BSSNGridDevice(const BSSNGridSoA<T> &grid) { allocate_like(grid); }

    void allocate_like(const BSSNGridSoA<T> &grid) {
        dims = grid.dims;
        st = grid.st;
        dx = grid.dx;
        dy = grid.dy;
        dz = grid.dz;
        x0 = grid.x0;
        y0 = grid.y0;
        z0 = grid.z0;

        alpha.allocate_like(grid.alpha);
        chi.allocate_like(grid.chi);
        K.allocate_like(grid.K);
        Theta.allocate_like(grid.Theta);
        for (int c = 0; c < 3; ++c) {
            beta[c].allocate_like(grid.beta[c]);
            B[c].allocate_like(grid.B[c]);
            tildeGamma[c].allocate_like(grid.tildeGamma[c]);
            Z[c].allocate_like(grid.Z[c]);
        }
        for (int s = 0; s < 6; ++s) {
            gamma_tilde[s].allocate_like(grid.gamma_tilde[s]);
            gamma_tilde_inv[s].allocate_like(grid.gamma_tilde_inv[s]);
            A_tilde[s].allocate_like(grid.A_tilde[s]);
            Ricci[s].allocate_like(grid.Ricci[s]);
        }
    }

    void copy_from_host(const BSSNGridSoA<T> &grid) {
        if (!strides_match(st, grid.st) || !dims_match(dims, grid.dims))
            allocate_like(grid);
        dims = grid.dims;
        st = grid.st;
        dx = grid.dx;
        dy = grid.dy;
        dz = grid.dz;
        x0 = grid.x0;
        y0 = grid.y0;
        z0 = grid.z0;

        alpha.copy_from_host(grid.alpha);
        chi.copy_from_host(grid.chi);
        K.copy_from_host(grid.K);
        Theta.copy_from_host(grid.Theta);
        for (int c = 0; c < 3; ++c) {
            beta[c].copy_from_host(grid.beta[c]);
            B[c].copy_from_host(grid.B[c]);
            tildeGamma[c].copy_from_host(grid.tildeGamma[c]);
            Z[c].copy_from_host(grid.Z[c]);
        }
        for (int s = 0; s < 6; ++s) {
            gamma_tilde[s].copy_from_host(grid.gamma_tilde[s]);
            gamma_tilde_inv[s].copy_from_host(grid.gamma_tilde_inv[s]);
            A_tilde[s].copy_from_host(grid.A_tilde[s]);
            Ricci[s].copy_from_host(grid.Ricci[s]);
        }
    }

    void copy_to_host(BSSNGridSoA<T> &grid) const {
        if (!dims_match(dims, grid.dims) || !strides_match(st, grid.st)) {
            throw std::invalid_argument("BSSNGridDevice::copy_to_host layout mismatch");
        }
        grid.x0 = x0;
        grid.y0 = y0;
        grid.z0 = z0;
        alpha.copy_to_host(grid.alpha);
        chi.copy_to_host(grid.chi);
        K.copy_to_host(grid.K);
        Theta.copy_to_host(grid.Theta);
        for (int c = 0; c < 3; ++c) {
            beta[c].copy_to_host(grid.beta[c]);
            B[c].copy_to_host(grid.B[c]);
            tildeGamma[c].copy_to_host(grid.tildeGamma[c]);
            Z[c].copy_to_host(grid.Z[c]);
        }
        for (int s = 0; s < 6; ++s) {
            gamma_tilde[s].copy_to_host(grid.gamma_tilde[s]);
            gamma_tilde_inv[s].copy_to_host(grid.gamma_tilde_inv[s]);
            A_tilde[s].copy_to_host(grid.A_tilde[s]);
            Ricci[s].copy_to_host(grid.Ricci[s]);
        }
    }

    [[nodiscard]] BSSNGridView<T> view() noexcept {
        BSSNGridView<T> out;
        out.dims = dims;
        out.st = st;
        out.alpha = alpha.view();
        out.chi = chi.view();
        out.K = K.view();
        out.Theta = Theta.view();
        for (int c = 0; c < 3; ++c) {
            out.beta[c] = beta[c].view();
            out.B[c] = B[c].view();
            out.tildeGamma[c] = tildeGamma[c].view();
            out.Z[c] = Z[c].view();
        }
        for (int s = 0; s < 6; ++s) {
            out.gamma_tilde[s] = gamma_tilde[s].view();
            out.gamma_tilde_inv[s] = gamma_tilde_inv[s].view();
            out.A_tilde[s] = A_tilde[s].view();
            out.Ricci[s] = Ricci[s].view();
        }
        out.dx = dx;
        out.dy = dy;
        out.dz = dz;
        out.x0 = x0;
        out.y0 = y0;
        out.z0 = z0;
        return out;
    }

    [[nodiscard]] BSSNGridView<const T> view() const noexcept {
        BSSNGridView<const T> out;
        out.dims = dims;
        out.st = st;
        out.alpha = alpha.view();
        out.chi = chi.view();
        out.K = K.view();
        out.Theta = Theta.view();
        for (int c = 0; c < 3; ++c) {
            out.beta[c] = beta[c].view();
            out.B[c] = B[c].view();
            out.tildeGamma[c] = tildeGamma[c].view();
            out.Z[c] = Z[c].view();
        }
        for (int s = 0; s < 6; ++s) {
            out.gamma_tilde[s] = gamma_tilde[s].view();
            out.gamma_tilde_inv[s] = gamma_tilde_inv[s].view();
            out.A_tilde[s] = A_tilde[s].view();
            out.Ricci[s] = Ricci[s].view();
        }
        out.dx = dx;
        out.dy = dy;
        out.dz = dz;
        out.x0 = x0;
        out.y0 = y0;
        out.z0 = z0;
        return out;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        std::size_t total = alpha.bytes() + chi.bytes() + K.bytes() + Theta.bytes();
        for (int c = 0; c < 3; ++c)
            total += beta[c].bytes() + B[c].bytes() + tildeGamma[c].bytes() + Z[c].bytes();
        for (int s = 0; s < 6; ++s) {
            total += gamma_tilde[s].bytes() + gamma_tilde_inv[s].bytes() + A_tilde[s].bytes() +
                     Ricci[s].bytes();
        }
        return total;
    }
};

template <typename T> struct BSSNCUDAStepperState {
    BSSNGridDevice<T>         grid;
    BSSNRHSWorkspaceDevice<T> stage_state;
    BSSNRHSWorkspaceDevice<T> rhs_workspace;
    DeviceField3D<T>          theta_ricciz4_trace_cache;

    void allocate_like(const BSSNGridSoA<T> &prototype) {
        grid.allocate_like(prototype);
        stage_state.allocate_like(prototype);
        rhs_workspace.allocate_like(prototype);
        theta_ricciz4_trace_cache.allocate_like(prototype.alpha);
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return grid.bytes() + stage_state.bytes() + rhs_workspace.bytes() +
               theta_ricciz4_trace_cache.bytes();
    }
};

} // namespace tensorium_RG::bssn
