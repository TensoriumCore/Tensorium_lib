/**
 * @file BSSN_GRID_MODULE.md
 * @brief High-level module overview for the Tensorium BSSN grid implementation.
 * @defgroup BSSN_Grid BSSN Grid Infrastructure
 * @details
 * ## Purpose
 * The `BSSN_Grid` module implements the vacuum Baumgarte–Shapiro–Shibata–Nakamura (BSSN)
 * formulation on a Cartesian structured grid stored in structure-of-arrays form.  Every header
 * under `includes/Tensorium/Physics/DiffGeometry/BSSN_Grid` participates in this module and
 * therefore inherits the conventions stated here.
 *
 * ## Mathematical formulation
 * We evolve the conformal variables \f$\{\chi,\tilde{\gamma}_{ij},\tilde{A}_{ij},K,\tilde{\Gamma}^i\}\f$
 * coupled to the gauge \f$\{\alpha,\beta^i,B^i\}\f$.  Spatial indices \f$i,j,k,\dots\in\{0,1,2\}\f$
 * and spacetime indices \f$\mu,\nu\in\{0,1,2,3\}\f$.  The conformal factor satisfies
 * \f$\chi=e^{-4\phi}\f$, so \f$\tilde{\gamma}_{ij}=\chi\,\gamma_{ij}\f$ has unit determinant and the
 * physical metric is \f$\gamma_{ij}=\chi^{-1}\,\tilde{\gamma}_{ij}\f$.  We follow the sign conventions of
 * Baumgarte–Shapiro (2009) and use geometric units \f$G=c=1\f$.  The evolution system implemented in
 * the RHS kernels corresponds to the standard vacuum BSSN equations:
 * \f[
 * \begin{aligned}
 * (\partial_t-\mathcal{L}_\beta)\chi &= \tfrac{2}{3}\chi\,(\alpha K-\partial_k\beta^k),\\
 * (\partial_t-\mathcal{L}_\beta)\tilde{\gamma}_{ij} &= -2\alpha\tilde{A}_{ij},\\
 * (\partial_t-\mathcal{L}_\beta)K &= -\gamma^{ij}D_iD_j\alpha + \alpha\left( \tilde{A}_{ij}\tilde{A}^{ij} + \tfrac{1}{3}K^2 \right),\\
 * (\partial_t-\mathcal{L}_\beta)\tilde{A}_{ij} &= \chi\left[-D_iD_j\alpha + \alpha R_{ij}\right]^{TF} + \alpha\left(K\tilde{A}_{ij}-2\tilde{A}_{ik}\tilde{A}^k_{\ j}\right),\\
 * (\partial_t-\mathcal{L}_\beta)\tilde{\Gamma}^i &= -2\tilde{A}^{ij}\partial_j\alpha + 2\alpha\left(\tilde{\Gamma}^i_{\ jk}\tilde{A}^{jk} - \tfrac{2}{3}\tilde{\gamma}^{ij}\partial_j K\right) + \tilde{\gamma}^{jk}\partial_j\partial_k\beta^i + \tfrac{1}{3}\tilde{\gamma}^{ij}\partial_j\partial_k\beta^k \\
 * &\quad - \tilde{\Gamma}^k\partial_k\beta^i + \tfrac{2}{3}\tilde{\Gamma}^i\partial_k\beta^k.
 * \end{aligned}
 * \f]
 * The Ricci tensor splits into conformal and conformal-factor parts,
 * \f$R_{ij}=\tilde{R}_{ij}+R^{\chi}_{ij}\f$, with explicit finite-difference realizations described in
 * `Geometry/BSSNRicci.hpp`.
 *
 * ## Gauge conditions
 *  - **1+log slicing**: \f$(\partial_t-\mathcal{L}_\beta)\alpha=-2\alpha K\f$.
 *  - **Gamma-driver shift**: \f$(\partial_t-\mathcal{L}_\beta)\beta^i = \frac{3}{4} B^i\f$ and
 *    \f$(\partial_t-\mathcal{L}_\beta) B^i = (\partial_t\tilde{\Gamma}^i-\mathcal{L}_\beta\tilde{\Gamma}^i) - \eta B^i\f$.
 *    The damping parameter \f$\eta\f$ and the \f$3/4\f$ factor are configurable via
 *    `GaugeParameters`.
 *
 * ## Stored fields
 * `Fields/BSSNGridSoA.hpp` allocates aligned `Field3D` arrays for all evolved and diagnostic
 * quantities:
 * - Primary state: \f$\alpha,\beta^i,B^i,\chi,K,\tilde{\gamma}_{ij},\tilde{A}_{ij},\tilde{\Gamma}^i\f$.
 * - Derived caches: \f$\tilde{\gamma}^{ij}\f$, \f$\tilde{\Gamma}^i_{\ jk}\f$ (27 components), Ricci tensor,
 *   and constraint monitors \f$H,M_i,C_i\f$.
 * All arrays share identical strides to enable SIMD-friendly loops.
 *
 * ## Constraints and invariants
 * Besides the Hamiltonian \f$H = R + K^2 - K_{ij}K^{ij}\f$ and the momentum constraint
 * \f$M_i = D_j K^j_{\ i} - D_i K\f$, the module tracks algebraic conditions
 * \f$\det\tilde{\gamma}=1\f$, \f$\tilde{\gamma}^{ij}\tilde{A}_{ij}=0\f$, and the Gamma constraint
 * \f$C^i = \tilde{\Gamma}^i + \partial_j\tilde{\gamma}^{ij}\f$.  Projection and monitoring helpers
 * keep these quantities bounded and provide tolerances based on stencil order.
 *
 * ## Architecture overview
 * 1. **SoA grid**: `BSSNGridSoA` stores halo-augmented fields with uniform strides and holds
 *    domain metadata (origin, spacing, padding).
 * 2. **Derivative operators**: `Derivatives/BSSNGridDerivatives.hpp` exposes 4th-order centered
 *    stencils for first/second derivatives, one-sided upwind advection operators, and KO6
 *    dissipation.  Ghost-zone width must be at least two cells.
 * 3. **Geometry builders**: Christoffels, Ricci tensor, invariants, and projections live under
 *    `Geometry/`.  Contracted connections depend on derivatives of \f$\tilde{\gamma}_{ij}\f$ and
 *    inverse metrics; Ricci requires both metric and conformal-factor derivatives.
 * 4. **Constraint evaluators**: `Constraints/` assembles \f$H, M_i, C_i\f$ and statistical monitors.
 * 5. **Evolution kernels**: `Evolution/` contains RHS builders for each field, organized so that
 *    advection terms reuse upwind stencils and geometric terms consume cached tensors.
 * 6. **Time integrator**: `TimeIntegration/BSSNRK4.hpp` orchestrates halo filling, geometry refresh,
 *    RHS evaluation, and constraint monitoring through four Runge–Kutta stages.
 *
 * Dependencies follow the graph: derivatives → (metric inverses, Christoffels) → Ricci & constraints → RHS
 * → RK stepper → projection/monitor.  `project_bssn_after_update` enforces
 * \f$\det\tilde{\gamma}=1\f$ and \f$\mathrm{tr}\,\tilde{A}=0\f$ after every partial state update.
 *
 * ## Boundary conditions and halos
 * The grid allocates `ng` halo cells around the physical domain.  Kernels restrict loops to
 * `padding≥2` interior points so that the 4th-order stencils (±2) access valid data.  Calling code is
 * responsible for populating halos through `apply_halos_grid<Boundary>`; the provided `BoundaryClamp`
 * clamps guard values to their nearest interior neighbor, emulating radiative-free (approximate
 * Dirichlet) boundaries.  Outflow or symmetry operators can be plugged in by specializing the
 * `Boundary` functor.
 *
 * ## Numerical methods
 * - 4th-order centered differences for \f$\partial_i\f$ and \f$\partial_i\partial_j\f$.
 * - 3-point upwind advection (`Dx_upwind`, etc.) driven by the sign of each \f$\beta^i\f$ component.
 * - Kreiss–Oliger dissipation (order 6) with configurable coefficient \f$\sigma=0.1\f$ in RHS builders.
 * - RK4 time integration; CFL timestep estimated via `compute_dt_cfl` by combining lapse and shift
 *   speeds.
 * - Ricci computation recomputes \f$\tilde{\Gamma}^i_{\ jk}\f$ on demand to maintain coherence with
 *   `project_bssn_after_update`.
 *
 * ## How to extend
 * 1. **Matter sources**: insert \f$4\pi(\rho+S)\f$ and \f$8\pi S_{ij}\f$ terms into the RHS kernels.
 *    Augment `BSSNGridSoA` with stress-energy fields and document their normalization.
 * 2. **Gauge choices**: add new slicing or shift drivers by extending `GaugeParameters` and writing
 *    alternative RHS helpers.  Reference this module page for consistent notation.
 * 3. **Stencil order**: implement higher-order operators inside `Derivatives/` and update
 *    `compute_invariant_tolerances` so projection tolerances track the new truncation error.
 * 4. **Boundary conditions**: implement a new `Boundary` functor with the same `apply(Field3D,GridDims)`
 *    signature and pass it to `apply_halos_grid` / the RK stepper template parameter.
 * 5. **Additional diagnostics**: hook into `BSSNRKStepper::set_constraint_callback` to stream custom
 *    telemetry or to trigger AMR / filtering.
 *
 * ## Units and conventions
 * All headers referenced by this module assume geometric units \f$G=c=1\f$, spacelike signature
 * \f$(+,+,+)\f$, and the definition \f$\chi=e^{-4\phi}\f$.  Physical metrics and tensors are recovered
 * via \f$\gamma_{ij}=\chi^{-1}\tilde{\gamma}_{ij}\f$, \f$K_{ij}=\chi^{-1}(\tilde{A}_{ij}+\tfrac{1}{3}\gamma_{ij}K)\f$.
 * Refer back to this group (`@ref BSSN_Grid`) before introducing new notation in downstream headers.
 */
