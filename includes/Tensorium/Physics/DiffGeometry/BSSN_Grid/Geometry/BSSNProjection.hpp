#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "BSSNGamma.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>
#include <cmath>

/**
 * @file BSSNProjection.hpp
 * @brief Enforce the algebraic BSSN constraints \f$\det\tilde{\gamma}=1\f$ and
 * \f$\mathrm{tr}\,\tilde{A}=0\f$.
 * @details
 * RK stages drift away from the conformal manifold because `compute_rhs_gamma_tilde` and
 * `compute_rhs_A_tilde` treat \f$\tilde{\gamma}_{ij}\f$ and \f$\tilde{A}_{ij}\f$ as unconstrained
 * tensors. `project_bssn_state` rescales the metric and subtracts traces so the invariants
 * monitored in `BSSNInvariants.hpp` remain within tolerance.  Optional knobs let callers skip
 * expensive steps when experimenting with alternative stabilization techniques.
 */

namespace tensorium_RG::bssn {

/// @brief Fine-grained control over which invariants get enforced during projection.
struct ProjectionConfig {
    size_t padding = 4;                    // guard cells before touching interior
    bool   renormalize_metric = true;      // enforce det(gamma_tilde)=1
    bool   project_A_tilde = true;         // enforce trace-free A_tilde
    bool   recompute_inverse = true;       // refresh gamma_tilde_inv from gamma_tilde
    bool   resync_contracted_gamma = true; // recompute stored Gamma^i from metric
};

namespace detail {

inline double det3(double gxx, double gxy, double gxz, double gyy, double gyz, double gzz) {
    return gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
           gxz * (gxy * gyz - gxz * gyy);
}

} // namespace detail

template <typename T>
inline void project_conformal_metric_and_A(BSSNGridSoA<T> &G, size_t padding = 4,
                                          bool renormalize_metric = true,
                                          bool project_A_tilde = true) {
    if (!renormalize_metric && !project_A_tilde)
        return;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.gamma_tilde[XX].idx(i, j, k);

                double gxx, gxy, gxz, gyy, gyz, gzz;
                tensorium_RG::load_sym6(G.gamma_tilde, id, gxx, gxy, gxz, gyy, gyz, gzz);

                const double det = detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
                double       scale = 1.0;
                if (renormalize_metric && det > 0.0)
                    scale = 1.0 / std::cbrt(det);

                if (renormalize_metric && std::isfinite(scale)) {
                    gxx *= scale;
                    gxy *= scale;
                    gxz *= scale;
                    gyy *= scale;
                    gyz *= scale;
                    gzz *= scale;
                    tensorium_RG::store_sym6(G.gamma_tilde, id, (T)gxx, (T)gxy, (T)gxz, (T)gyy,
                                             (T)gyz, (T)gzz);
                } else {
                    scale = 1.0;
                }

                tensorium_RG::load_sym6(G.gamma_tilde, id, gxx, gxy, gxz, gyy, gyz, gzz);

                double inv_det = detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
                inv_det = (inv_det != 0.0) ? (1.0 / inv_det) : 0.0;
                const double gixx = (gyy * gzz - gyz * gyz) * inv_det;
                const double gixy = (gxz * gyz - gxy * gzz) * inv_det;
                const double gixz = (gxy * gyz - gxz * gyy) * inv_det;
                const double giyy = (gxx * gzz - gxz * gxz) * inv_det;
                const double giyz = (gxy * gxz - gxx * gyz) * inv_det;
                const double gizz = (gxx * gyy - gxy * gxy) * inv_det;

                double Axx, Axy, Axz, Ayy, Ayz, Azz;
                tensorium_RG::load_sym6(G.A_tilde, id, Axx, Axy, Axz, Ayy, Ayz, Azz);

                if (renormalize_metric && std::isfinite(scale)) {
                    Axx *= scale;
                    Axy *= scale;
                    Axz *= scale;
                    Ayy *= scale;
                    Ayz *= scale;
                    Azz *= scale;
                }

                if (!project_A_tilde) {
                    tensorium_RG::store_sym6(G.A_tilde, id, (T)Axx, (T)Axy, (T)Axz, (T)Ayy,
                                             (T)Ayz, (T)Azz);
                    continue;
                }

                const double trace = gixx * Axx + giyy * Ayy + gizz * Azz +
                                     2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
                const double one_third_trace = trace / 3.0;
                Axx -= gxx * one_third_trace;
                Axy -= gxy * one_third_trace;
                Axz -= gxz * one_third_trace;
                Ayy -= gyy * one_third_trace;
                Ayz -= gyz * one_third_trace;
                Azz -= gzz * one_third_trace;

                tensorium_RG::store_sym6(G.A_tilde, id, (T)Axx, (T)Axy, (T)Axz, (T)Ayy, (T)Ayz,
                                         (T)Azz);
            }
        }
    }
}

/**
 * @brief Apply determinant renormalization, \f$\tilde{A}\f$ trace-free projection, and
 * \f$\tilde{\Gamma}^i\f$ resynchronization.
 * @param cfg Controls which projections are executed; defaults enforce all invariants inside the
 *            interior `padding` region.
 *
 * @details
 * - Metric renormalization rescales \f$\tilde{\gamma}_{ij}\f$ by \f$\det(\tilde{\gamma})^{-1/3}\f$
 * so the new determinant equals 1.
 * - Trace-free projection subtracts
 * \f$\tfrac{1}{3}\tilde{\gamma}_{ij}\tilde{\gamma}^{mn}\tilde{A}_{mn}\f$ from
 *   \f$\tilde{A}_{ij}\f$.
 * - Optionally recompute \f$\tilde{\gamma}^{ij}\f$ from the renormalized metric and resynchronize
 *   \f$\tilde{\Gamma}^i\f$ via `metric_inverse_divergence`.
 */
template <typename T>
inline void project_bssn_state(BSSNGridSoA<T> &G, const ProjectionConfig &cfg = {}) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(cfg.padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

    if (cfg.renormalize_metric || cfg.project_A_tilde)
        project_conformal_metric_and_A(G, cfg.padding, cfg.renormalize_metric, cfg.project_A_tilde);

    if (cfg.recompute_inverse) {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i) {
            for (size_t j = j0; j < j1; ++j) {
                for (size_t k = k0; k < k1; ++k) {
                    const size_t id = G.gamma_tilde[XX].idx(i, j, k);
                    double       gxx, gxy, gxz, gyy, gyz, gzz;
                    tensorium_RG::load_sym6(G.gamma_tilde, id, gxx, gxy, gxz, gyy, gyz, gzz);
                    const double det = detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
                    const double inv_det = (det != 0.0) ? (1.0 / det) : 0.0;
                    const double gixx = (gyy * gzz - gyz * gyz) * inv_det;
                    const double gixy = (gxz * gyz - gxy * gzz) * inv_det;
                    const double gixz = (gxy * gyz - gxz * gyy) * inv_det;
                    const double giyy = (gxx * gzz - gxz * gxz) * inv_det;
                    const double giyz = (gxy * gxz - gxx * gyz) * inv_det;
                    const double gizz = (gxx * gyy - gxy * gxy) * inv_det;
                    tensorium_RG::store_sym6(G.gamma_tilde_inv, id, (T)gixx, (T)gixy, (T)gixz,
                                             (T)giyy, (T)giyz, (T)gizz);
                }
            }
        }
    }

    if (cfg.resync_contracted_gamma) {
#pragma omp parallel for collapse(2)
        for (size_t i = i0; i < i1; ++i) {
            for (size_t j = j0; j < j1; ++j) {
                for (size_t k = k0; k < k1; ++k) {
                    const size_t id = G.gamma_tilde[XX].idx(i, j, k);
                    T            div[3];
                    metric_inverse_divergence(G, i, j, k, div);
                    G.tildeGamma[0].ptr()[id] = -div[0];
                    G.tildeGamma[1].ptr()[id] = -div[1];
                    G.tildeGamma[2].ptr()[id] = -div[2];
                }
            }
        }
    }
}

/// @brief Convenience wrapper that enforces all invariants after every RK stage.
template <typename T> inline void project_bssn_after_update(BSSNGridSoA<T> &G, size_t padding = 4) {
    ProjectionConfig cfg;
    cfg.padding = padding;
    cfg.renormalize_metric = true;
    cfg.project_A_tilde = true;
    cfg.recompute_inverse = true;
    cfg.resync_contracted_gamma = true;
    project_bssn_state(G, cfg);
}

template <typename T> inline void project_z4c_state(BSSNGridSoA<T> &G, const ProjectionConfig &cfg = {}) {
    project_bssn_state(G, cfg);
}

template <typename T> inline void project_z4c_after_update(BSSNGridSoA<T> &G, size_t padding = 4) {
    project_bssn_after_update(G, padding);
}

} // namespace tensorium_RG::bssn
