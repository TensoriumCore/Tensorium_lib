
#pragma once

#include "Tensorium_Grid/GridSetup.hpp"

namespace tensorium_RG {

template <typename T> class BSSNGridSoA {
  public:
    GridDims   dims;
    Strides<T> st;

    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;

    Field3D<T> beta[3];
    Field3D<T> tildeGamma[3];

    Field3D<T> gamma_tilde[6];
    Field3D<T> gamma_tilde_inv[6];
    Field3D<T> A_tilde[6];

    Field3D<T> Gamma_tilde[27];
    T          dx, dy, dz;
    Field3D<T> Ricci[6];

    BSSNGridSoA(size_t nx, size_t ny, size_t nz, size_t ng, T dx_, T dy_, T dz_)
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

        for (int i = 0; i < 3; ++i) {
            alloc_field(beta[i]);
            alloc_field(tildeGamma[i]);
        }

        for (int s = 0; s < 6; ++s) {
            alloc_field(gamma_tilde[s]);
            alloc_field(gamma_tilde_inv[s]);
            alloc_field(A_tilde[s]);
        }
        for (int q = 0; q < 27; ++q)
            alloc_field(Gamma_tilde[q]);
		for (int r = 0; r < 6; ++r)
			alloc_field(Ricci[r]);
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

    inline void coords(size_t i, size_t j, size_t k, T &x, T &y, T &z) const noexcept {
        x = x0 + (i - dims.ng) * dx;
        y = y0 + (j - dims.ng) * dy;
        z = z0 + (k - dims.ng) * dz;
    }
};

template <typename Boundary, typename T> inline void apply_halos_grid(BSSNGridSoA<T> &G) {
    const auto &D = G.dims;

    Boundary::apply(G.alpha, D);
    Boundary::apply(G.chi, D);
    Boundary::apply(G.K, D);

    for (int i = 0; i < 3; ++i) {
        Boundary::apply(G.beta[i], D);
        Boundary::apply(G.tildeGamma[i], D);
    }

    for (int s = 0; s < 6; ++s) {
        Boundary::apply(G.gamma_tilde[s], D);
        Boundary::apply(G.gamma_tilde_inv[s], D);
        Boundary::apply(G.A_tilde[s], D);
    }
    for (int q = 0; q < 27; ++q)
        Boundary::apply(G.Gamma_tilde[q], D);
}

} // namespace tensorium_RG
