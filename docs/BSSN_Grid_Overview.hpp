/**
 * @defgroup BSSN_Grid BSSN Grid Module
 * @brief BSSN-based numerical relativity infrastructure for evolving Einstein's equations.
 *
 * @details
 * This module implements the Baumgarte–Shapiro–Shibata–Nakamura (BSSN) formulation of the
 * 3+1 Einstein equations:
 *
 * \f[
 * R_{ij} = \tilde{R}_{ij} + R^{\chi}_{ij}
 * \f]
 *
 * with the conformal decomposition:
 *
 * \f[
 * \gamma_{ij} = \chi^{-1} \tilde{\gamma}_{ij}, \quad \det(\tilde{\gamma}) = 1
 * \f]
 *
 * Stored variables:
 * - Conformal metric \f$ \tilde{\gamma}_{ij} \f$
 * - Conformal factor \f$ \chi = e^{-4\phi} \f$
 * - Trace-free curvature \f$ \tilde{A}_{ij} \f$
 * - Mean curvature \f$ K \f$
 * - Contracted Christoffels \f$ \tilde{\Gamma}^i \f$
 * - Gauge: lapse \f$ \alpha \f$, shift \f$ \beta^i \f$
 *
 * Architecture:
 * - Structure-of-Arrays (SoA) grid layout
 * - Halo / ghost zones for finite differences
 * - Cached geometric tensors (\tilde{\Gamma}, Ricci, constraints)
 * - RK4 time integration
 *
 * Numerics:
 * - Centered finite differences
 * - Kreiss--Oliger dissipation
 * - Constraint monitoring
 *
 * @note This documentation follows the conventions of Alcubierre (2008).
 */
