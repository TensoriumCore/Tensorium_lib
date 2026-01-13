
#pragma once

#include <algorithm>

#include "Tensorium_Grid/Grid/GridLayout.hpp"

/**
 * @file BSSNGridSoA.hpp
 * @brief Structure-of-arrays storage for all BSSN state variables.
 * @details
 * This header defines the aligned storage container used by every evolution, geometry, and
 * constraint kernel in the @ref BSSN_Grid module.  Each tensor component is stored in a dedicated
 * `Field3D` so loops enjoy unit stride in the fastest-varying dimension (z).  The constructor
 * allocates halos of width `ng` on all sides; interior kernels are responsible for skipping the
 * padding mandated by the 4th-order stencils.  Derived caches (inverse metric, Christoffels, Ricci,
 * constraints) are colocated with the primary state to avoid repeated allocations.
 *
 * @note Units and sign conventions follow the module overview in @ref BSSN_Grid.  Physical metrics
 * and tensors are obtained from the conformal data through \f$\gamma_{ij}=\chi^{-1}\tilde{\gamma}_{ij}\f$.
 */

namespace tensorium_RG {

/**
 * @tparam T floating-point storage type (double by default in production runs).
 * @brief Structure-of-arrays grid that holds all evolved, cache, and diagnostic fields.
 * @details
 * Fields are grouped thematically: scalar gauges (`alpha`, `chi`, `K`), vectors (`beta`, `B`,
 * `tildeGamma`), symmetric tensors stored in 6-component packed form (\f$\tilde{\gamma}_{ij}\f$,
 * \f$\tilde{\gamma}^{ij}\f$, \f$\tilde{A}_{ij}\f$), and auxiliary caches (full \f$\tilde{\Gamma}^i_{\ jk}\f$,
 * Ricci, constraints).  Each `Field3D` shares the same stride metadata so pointer arithmetic is
 * identical regardless of the component being accessed.  The grid remembers the physical spacing
 * `(dx,dy,dz)` and origin `(x0,y0,z0)` so derivative kernels can translate between index space and
 * physical coordinates when sampling diagnostics or initial data.
 */
template <typename T> class BSSNGridSoA {
  public:
    GridDims   dims;
    Strides<T> st;

    Field3D<T> alpha;
    Field3D<T> chi;
    Field3D<T> K;

    Field3D<T> beta[3];
    Field3D<T> B[3];
    Field3D<T> tildeGamma[3];

    Field3D<T> gamma_tilde[6];
    Field3D<T> gamma_tilde_inv[6];
    Field3D<T> A_tilde[6];

    Field3D<T> Gamma_tilde[27];
    T          dx, dy, dz;
    Field3D<T> Ricci[6];

    Field3D<T> Hc;
    Field3D<T> Mc[3];
    Field3D<T> Cc[3];
    /**
     * @brief Construct a grid with halo-aware allocation.
     * @param nx,ny,nz Number of physical cells along x/y/z.
     * @param ng Halo/ghost width in cells (must be ≥2 to support the 4th-order stencils documented
     *           in `Derivatives/BSSNGridDerivatives.hpp`).
     * @param dx_,dy_,dz_ Grid spacing in each direction.
     *
     * The constructor pads the fastest-varying (z) dimension to SIMD-friendly widths through
     * `pad_simd<T>` and allocates every tensor component with identical strides.  Vector and tensor
     * caches are zeroed only where correctness requires (e.g., \f$B^i\f$ start at zero for the
     * Gamma-driver).  Projection routines rely on these allocations being exclusive ownership, so
     * copy construction is intentionally omitted.
     */
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

        auto alloc_zero_field = [&](Field3D<T> &f) {
            alloc_field(f);
            std::fill_n(f.ptr(), nx_tot * ny_tot * nz_tot, T(0));
        };

        alloc_field(alpha);
        alloc_field(chi);
        alloc_field(K);

        for (int i = 0; i < 3; ++i) {
            alloc_field(beta[i]);
            alloc_zero_field(B[i]);
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

        alloc_field(Hc);
        for (int q = 0; q < 3; ++q) {
            alloc_field(Mc[q]);
            alloc_field(Cc[q]);
        }
    }

    /**
     * @brief Return the inclusive-exclusive bounds of the physical domain inside the halo buffer.
     * @param[out] i0,i1,j0,j1,k0,k1 Filled with the start/end indices that correspond to the first
     *             cell adjacent to the lower guard zone and the first guard cell above the domain.
     * @details The BSSN kernels shrink these bounds by up to two cells so that 4th-order centered
     * derivatives (±2) and KO6 dissipation (±3) never dereference halos unless explicitly desired.
     */
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

    /**
     * @brief Map an index triple to physical coordinates.
     * @param i,j,k Logical indices including halos.
     * @param[out] x,y,z Cartesian coordinates with origin `(x0,y0,z0)`.
     *
     * @note The coordinate mapping is used by initial-data builders and diagnostics to place radial
     * masks.  All users share the same conventions documented in @ref BSSN_Grid.
     */
    inline void coords(size_t i, size_t j, size_t k, T &x, T &y, T &z) const noexcept {
        x = x0 + (i - dims.ng) * dx;
        y = y0 + (j - dims.ng) * dy;
        z = z0 + (k - dims.ng) * dz;
    }
};

} // namespace tensorium_RG
