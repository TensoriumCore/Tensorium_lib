# BSSN Grid Subsystem

## 1. Overview
`BSSN_Grid` implements a CCZ4/Z4c-enhanced BSSN scheme.  The conformally rescaled Einstein variables are supplemented with the CCZ4 scalar \(\Theta\) and vector \(Z_i\) so that constraint violations propagate and damp.  The module provides:
- Structure-of-arrays storage plus halo-aware indexing (`BSSNGridSoA.hpp`).
- RHS kernels for every geometric, constraint, and gauge variable using sixth-order finite differences and shared KO6 dissipation.
- Ricci builders that inject the \(\Delta R^{\mathrm{Z4}}_{ij}\) correction before sourcing \(\tilde{A}_{ij}\), \(K\), and \(\Theta\).
- Projection/monitoring utilities that enforce \(\det\tilde{\gamma}=1\), \(\mathrm{Tr}\tilde{A}=0\), and emit Hamiltonian/momentum/Gamma norms as well as \(||\Theta||_2|| and \(||Z||_2||.
Only vacuum sources are currently supported, but the RHS stubs highlight where matter terms would be added.

## 2. State Variables
For every grid point the module stores:
- **Lapse \(\alpha\)** — controls the slicing of spacetime.
- **Shift \(\beta^i\)** — advects the spatial coordinates.
- **Gamma-driver auxiliary \(B^i\)** — first-order variable that damps the shift.
- **Conformal factor \(\chi\)** — equal to \(e^{-4\phi}\), used to rescale the physical metric.
- **Conformal metric \(\tilde{\gamma}_{ij}\)** and its inverse \(\tilde{\gamma}^{ij}\)** — unit-determinant rescaling of the spatial metric.
- **Trace-free conformal extrinsic curvature \(\tilde{A}_{ij}\)**.
- **Mean curvature \(K\)** — trace of the physical extrinsic curvature (RHSs mostly use \(\hat{K}=K-2\Theta\)).
- **Conformal connection functions \(\tilde{\Gamma}^i\)** — contracted Christoffel symbols.
- **Constraint fields** — \(\Theta\) and \(Z_i\) from CCZ4/Z4c.
Each field is stored in a `Field3D` with identical strides (`BSSNGridSoA.hpp`).

## 3. Evolution Equations
The implemented right-hand sides (padding-aware, allocation-free) correspond to the standard BSSN vacuum system:
- **Conformal factor**
  ```math
  \partial_t \chi = \beta^k \partial_k \chi + \frac{2}{3} \chi \left( \alpha K - \partial_k \beta^k \right).
  ```
- **Conformal metric**
  ```math
  \partial_t \tilde{\gamma}_{ij} = \beta^k \partial_k \tilde{\gamma}_{ij} + \tilde{\gamma}_{ik} \partial_j \beta^k + \tilde{\gamma}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{\gamma}_{ij} \partial_k \beta^k - 2 \alpha \tilde{A}_{ij}.
  ```
- **Trace-free extrinsic curvature**
  ```math
  \partial_t \tilde{A}_{ij} = \beta^k \partial_k \tilde{A}_{ij} + \tilde{A}_{ik} \partial_j \beta^k + \tilde{A}_{jk} \partial_i \beta^k - \frac{2}{3} \tilde{A}_{ij} \partial_k \beta^k - \left(D_i D_j \alpha\right)^{TF} + \alpha \left(R_{ij}^{TF} - 8\pi S_{ij}^{TF}\right),
  ```
  with \(R_{ij}\) built from the conformal metric, \(D_i\) the covariant derivative compatible with the physical metric, and the trace-free projector enforced explicitly; matter terms are zero in the current vacuum implementation.
- **Mean curvature** (implemented with \(K\) storage and \(\hat{K}\) sources)
  ```math
  \partial_t K = \beta^k \partial_k K - \gamma^{ij} D_i D_j \alpha + \alpha \left( \tilde{A}_{ij} \tilde{A}^{ij} + \frac{1}{3} \hat{K}^2 \right) + \alpha \kappa_1 (1-\kappa_2) \Theta.
  ```
- **CCZ4 scalar**
  ```math
  \partial_t \Theta = \beta^k \partial_k \Theta + \frac{\alpha}{2}\left(R + 2 D_k Z^k - \tilde{A}_{ij}\tilde{A}^{ij} + \frac{2}{3} K^2\right) - \alpha \kappa_1 (2 + \kappa_2) \Theta.
  ```
- **CCZ4 vector**
  ```math
  \partial_t Z_i = \beta^k \partial_k Z_i + \alpha \left(D_j \tilde{A}^j_{\ i} + 6 \tilde{A}_{ij} \partial^j \phi - \frac{2}{3} D_i K\right) - \alpha \kappa_z Z_i.
  ```
- **Conformal connection functions**
  ```math
  \begin{aligned}
  \partial_t \tilde{\Gamma}^i &= \beta^k \partial_k \tilde{\Gamma}^i - \tilde{\Gamma}^k \partial_k \beta^i + \frac{2}{3} \tilde{\Gamma}^i \partial_k \beta^k \\
  &\quad + \tilde{\gamma}^{jk} \partial_j \partial_k \beta^i + \frac{1}{3} \tilde{\gamma}^{ij} \partial_j \partial_k \beta^k - 2 \tilde{A}^{ij} \partial_j \alpha \\
  &\quad + 2 \alpha \left( \tilde{\Gamma}^i_{\ jk} \tilde{A}^{jk} - \frac{2}{3} \tilde{\gamma}^{ij} \partial_j K \right).
  \end{aligned}
  ```
- **Gauge system**
  - 1+log slicing:
    ```math
    \partial_t \alpha = \beta^k \partial_k \alpha - 2 \alpha K.
    ```
  - Gamma-driver shift and auxiliary:
    ```math
    \partial_t \beta^i = \beta^k \partial_k \beta^i + \frac{3}{4} B^i,
    \qquad
    \partial_t B^i = \beta^k \partial_k B^i + \partial_t \tilde{\Gamma}^i - \eta B^i.
    ```
  The code evaluates \(\partial_t \tilde{\Gamma}^i\) first and feeds it directly into the driver RHS.

## 4. Gauge System
- \(B^i\) converts the second-order Gamma-driver into two first-order equations, providing control over the damping of coordinate drifts.
- Damping parameters \(\eta\), \(\kappa_1\), \(\kappa_2\), and \(\kappa_z\) plus the shared KO6 strength are configurable through `GaugeParameters` so individual runs can tune both shift/lapse and Z4 damping.
- Advection by the shift (\(\beta^k \partial_k \beta^i\) and \(\beta^k \partial_k B^i\)) is included to preserve hyperbolicity and to keep the gauge locally transported with the physical flow.

## 5. Geometric Quantities
- \(\tilde{\gamma}^{ij}\) is obtained by explicitly inverting \(\tilde{\gamma}_{ij}\) per grid point; both tensors are stored simultaneously (`BSSNGridSoA`).
- \(\tilde{\Gamma}^i_{\ jk}\) is computed from spatial derivatives of the conformal metric using `BSSNCHristoffelTilde.hpp`, and \(\tilde{\Gamma}^i\) is the negative divergence of \(\tilde{\gamma}^{ij}\).
- The Ricci tensor is assembled in `BSSNRicci.hpp` by combining conformal metric derivatives, the conformal factor gradient, and the stored Christoffels; `compute_RicciZ4` adds \(\Delta R^{\mathrm{Z4}}_{ij} = D_i Z_j + D_j Z_i - \tfrac{2}{3}\gamma_{ij} D_k Z^k\) before sourcing \(\tilde{A}_{ij}\), \(K\), and \(\Theta\).

## 6. Projection Conditions
The conformal metric must maintain unit determinant and \(\tilde{A}_{ij}\) must remain trace-free.  `project_bssn_after_update` enforces:
- \(\det(\tilde{\gamma}_{ij}) = 1\) via rescaling.
- \(\mathrm{Tr}\,\tilde{A} = \tilde{\gamma}^{ij} \tilde{A}_{ij} = 0\) via explicit subtraction of the trace.
- Recompute \(\tilde{\gamma}^{ij}\) from the renormalized metric and resync \(\tilde{\Gamma}^i\) using the metric divergence.
This helper runs after every RK stage so CCZ4 variables remain consistent with the conformal constraints.

## 7. Constraints
- **Hamiltonian** \(H = R + K^2 - K_{ij} K^{ij}\).
- **Momentum** \(M_i = D_j K^j_{\ i} - D_i K\).
- **Gamma** \(C_i = \tilde{\Gamma}^i + \partial_j \tilde{\gamma}^{ij} + 2 Z^i\).
`BSSNConstraintsGrid.hpp` evaluates \(H\), \(M_i\), and \(C_i\) on the interior region.  `BSSNConstraintMonitoring.hpp` condenses them into L∞/L² norms, tracks \(|\det \tilde{\gamma} - 1|\) and \(|\mathrm{Tr}\,\tilde{A}|\), and (optionally) reports \(||\Theta||_2|| and \(||Z||_2|| to verify that CCZ4 damping remains effective.

## 8. Ghost Zones and Padding
Fourth-order finite differences require two guard cells.  `clamped_lower` and `clamped_upper` limit interior loops so that \(i \in [\mathrm{ng}+2, \mathrm{ng}+n-2]\).  Before each RHS evaluation, callers must refresh halo values (periodic, clamp, or problem-specific) using `apply_halos_grid`.  This separation lets the RK driver decide when halo exchanges occur.

## 9. Code Structure
- `Fields/` — `BSSNGridSoA.hpp` defines the structure-of-arrays storage and utility routines.
- `Evolution/` — `BSSNEvolution*.hpp` contain the RHS kernels for \(\chi, \tilde{\gamma}_{ij}, \tilde{A}_{ij}, K, \tilde{\Gamma}^i\) and the gauge.
- `Geometry/` — Christoffel builders, Ricci calculators, projections, and monitoring helpers.
- `Constraints/` — Hamiltonian, momentum, and Gamma constraint evaluation plus statistics.
- `InitialData/` — Minkowski, isotropic Schwarzschild, and Bowen–York puncture datasets initialize all state fields and precompute geometric quantities.

For additional usage context, see the tests and demos in `Tests/bssn/`.
