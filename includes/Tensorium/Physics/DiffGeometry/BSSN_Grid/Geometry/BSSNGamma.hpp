#pragma once

#include "../Derivatives/BSSNGridDerivatives.hpp"
#include "../Fields/BSSNGridSoA.hpp"

/**
 * @file BSSNGamma.hpp
 * @brief Contracted conformal connection diagnostics and construction utilities.
 * @details
 * \f$\tilde{\Gamma}^i = -\partial_j \tilde{\gamma}^{ij}\f$ in the BSSN formalism.  The code maintains
 * both an evolved \f$\tilde{\Gamma}^i\f$ and a diagnostic divergence of \f$\tilde{\gamma}^{ij}\f$ to track the
 * algebraic Gamma constraint \f$C^i\f$.  These helpers evaluate the divergence without mutating the
 * stored field so `compute_bssn_constraints` can compare the two and quantify violations.
 */

namespace tensorium_RG::bssn {

namespace detail {

#pragma omp declare simd notinbranch
template <typename T>
inline void metric_inverse_divergence_ptr(const T *p_ginv_xx, const T *p_ginv_xy,
                                          const T *p_ginv_xz, const T *p_ginv_yy,
                                          const T *p_ginv_yz, const T *p_ginv_zz,
                                          ptrdiff_t sx, ptrdiff_t sy, double inv_12dx,
                                          double inv_12dy, double inv_12dz, T out[3]) {
    using namespace tensorium_RG::fd;
    out[0] = Dx_ptr(p_ginv_xx, sx, inv_12dx) + Dy_ptr(p_ginv_xy, sy, inv_12dy) +
             Dz_ptr(p_ginv_xz, inv_12dz);
    out[1] = Dx_ptr(p_ginv_xy, sx, inv_12dx) + Dy_ptr(p_ginv_yy, sy, inv_12dy) +
             Dz_ptr(p_ginv_yz, inv_12dz);
    out[2] = Dx_ptr(p_ginv_xz, sx, inv_12dx) + Dy_ptr(p_ginv_yz, sy, inv_12dy) +
             Dz_ptr(p_ginv_zz, inv_12dz);
}

} // namespace detail

// Computes ∂_j γ̃^{ij} without mutating G.tildeGamma.  The Γ constraint compares
// this diagnostic derivative against the stateful vector field stored in the
// grid, so that violations remain observable even if Γ̃^i is evolved
// independently.
/**
 * @brief Evaluate \f$\partial_j\tilde{\gamma}^{ij}\f$ using the 4th-order derivative operators.
 * @details Called during Gamma-constraint monitoring as well as projection steps to re-synchronize
 * the evolved \f$\tilde{\Gamma}^i\f$ with the metric after renormalization.
 */
template <typename T>
inline void metric_inverse_divergence(const BSSNGridSoA<T> &G, size_t i, size_t j, size_t k,
                                      T out[3]) {
    const size_t id = G.gamma_tilde_inv[XX].idx(i, j, k);
    const auto  *p_xx = G.gamma_tilde_inv[XX].ptr() + id;
    const auto  *p_xy = G.gamma_tilde_inv[XY].ptr() + id;
    const auto  *p_xz = G.gamma_tilde_inv[XZ].ptr() + id;
    const auto  *p_yy = G.gamma_tilde_inv[YY].ptr() + id;
    const auto  *p_yz = G.gamma_tilde_inv[YZ].ptr() + id;
    const auto  *p_zz = G.gamma_tilde_inv[ZZ].ptr() + id;
    const auto   sx = G.gamma_tilde_inv[XX].st.sx;
    const auto   sy = G.gamma_tilde_inv[XX].st.sy;
    const double inv_12dx = 1.0 / (60.0 * G.dx);
    const double inv_12dy = 1.0 / (60.0 * G.dy);
    const double inv_12dz = 1.0 / (60.0 * G.dz);

    detail::metric_inverse_divergence_ptr(p_xx, p_xy, p_xz, p_yy, p_yz, p_zz, sx, sy, inv_12dx,
                                          inv_12dy, inv_12dz, out);
}

/**
 * @brief Convenience wrapper that produces \f$\tilde{\Gamma}^i\f$ from the metric and stores the
 *        negative divergence as required by the BSSN definition.
 * @note Avoid calling right before constraint evaluation, otherwise \f$C^i\f$ will trivially vanish.
 */
template <typename T>
inline void compute_contracted_gamma_from_metric(BSSNGridSoA<T> &G, size_t i, size_t j,
                                                 size_t k, T out[3]) {
    metric_inverse_divergence(G, i, j, k, out);
    out[0] = -out[0];
    out[1] = -out[1];
    out[2] = -out[2];
}

} // namespace tensorium_RG::bssn
