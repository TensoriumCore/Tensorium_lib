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
    return compute_constraint_monitor(G, H, padding);
}

} // namespace tensorium_RG::bssn
