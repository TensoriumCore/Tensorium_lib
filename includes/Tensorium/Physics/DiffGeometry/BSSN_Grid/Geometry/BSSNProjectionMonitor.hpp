#pragma once

#include "BSSNProjection.hpp"
#include "../Constraints/BSSNConstraintMonitoring.hpp"
#include "../Constraints/BSSNConstraintsGrid.hpp"

/**
 * @file BSSNProjectionMonitor.hpp
 * @brief Combined helper that enforces algebraic constraints, recomputes \f$R_{ij}\f$, and reports
 *        diagnostic norms.
 */

namespace tensorium_RG::bssn {

namespace detail {
template <typename T>
inline void subtract_z4c_gamma(const BSSNGridSoA<T> &G, Field3D<T> C[3]) {
    const size_t total = G.alpha.st.nx_tot * G.alpha.st.ny_tot * G.alpha.st.nz_tot;
    const T     *gixx = G.gamma_tilde_inv[XX].ptr();
    const T     *gixy = G.gamma_tilde_inv[XY].ptr();
    const T     *gixz = G.gamma_tilde_inv[XZ].ptr();
    const T     *giyy = G.gamma_tilde_inv[YY].ptr();
    const T     *giyz = G.gamma_tilde_inv[YZ].ptr();
    const T     *gizz = G.gamma_tilde_inv[ZZ].ptr();
    const T     *Zx = G.Z[0].ptr();
    const T     *Zy = G.Z[1].ptr();
    const T     *Zz = G.Z[2].ptr();

#pragma omp parallel for
    for (size_t idx = 0; idx < total; ++idx) {
        const double raise0 = static_cast<double>(gixx[idx]) * static_cast<double>(Zx[idx]) +
                               static_cast<double>(gixy[idx]) * static_cast<double>(Zy[idx]) +
                               static_cast<double>(gixz[idx]) * static_cast<double>(Zz[idx]);
        const double raise1 = static_cast<double>(gixy[idx]) * static_cast<double>(Zx[idx]) +
                               static_cast<double>(giyy[idx]) * static_cast<double>(Zy[idx]) +
                               static_cast<double>(giyz[idx]) * static_cast<double>(Zz[idx]);
        const double raise2 = static_cast<double>(gixz[idx]) * static_cast<double>(Zx[idx]) +
                               static_cast<double>(giyz[idx]) * static_cast<double>(Zy[idx]) +
                               static_cast<double>(gizz[idx]) * static_cast<double>(Zz[idx]);

        C[0].ptr()[idx] = static_cast<T>(static_cast<double>(C[0].ptr()[idx]) - 2.0 * raise0);
        C[1].ptr()[idx] = static_cast<T>(static_cast<double>(C[1].ptr()[idx]) - 2.0 * raise1);
        C[2].ptr()[idx] = static_cast<T>(static_cast<double>(C[2].ptr()[idx]) - 2.0 * raise2);
    }
}
} // namespace detail

/**
 * @brief Project the BSSN state and immediately recompute Hamiltonian, momentum, and Gamma constraints.
 * @param r_min,r_max Radial window for constraint evaluation (useful for puncture data).
 * @param padding Interior padding forwarded to `project_bssn_after_update` and monitor kernels.
 * @return Aggregated statistics of \f$H\f$ gathered by `compute_constraint_monitor`.
 */
template <typename T>
inline ConstraintMonitorStats project_and_monitor(BSSNGridSoA<T> &G, Field3D<T> &H,
                                                  Field3D<T> M[3], Field3D<T> C[3], double r_min,
                                                  double r_max, double xc, double yc, double zc,
                                                  size_t padding = 4) {
    project_bssn_after_update(G, padding);
    compute_bssn_constraints(G, G.Ricci, H, M, C, r_min, r_max, xc, yc, zc);
    detail::subtract_z4c_gamma(G, C);
    return compute_constraint_monitor(G, H, padding);
}

} // namespace tensorium_RG::bssn
