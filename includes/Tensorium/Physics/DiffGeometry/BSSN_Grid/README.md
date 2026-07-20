# BSSN_Grid Technical Guide

This document is the implementation-level reference for the `BSSN_Grid` subsystem in Tensorium.
It describes the mathematical formulation, the discretization and time integration strategy,
the boundary/halo logic, the initialization pipelines, and the diagnostics used in production runs.

The intent is to keep the theory and the code paths aligned. Every section points to concrete
headers so you can verify behavior directly.

## 1. Scope and Design Goals

`BSSN_Grid` is a structured-grid, finite-difference numerical relativity module for vacuum
Einstein evolution in a BSSN/Z4c-style formulation.

What it currently targets:

- 3D Cartesian grids with ghost zones.
- High-order finite differences (order 6 default, order 4 optional).
- Moving-puncture style gauge evolution.
- Constraint-aware evolution with Z4 variables (`Theta`, `Z^i`).
- CPU execution with OpenMP parallel loops.

What it does not yet provide as a full integrated runtime:

- End-to-end AMR integration in this module.
- End-to-end MPI-distributed BSSN runtime in this module.
- Matter source terms in the evolution equations (vacuum-only RHS at present).

Core paths:

- State and storage: `Fields/BSSNGridSoA.hpp`
- Derivatives: `Derivatives/BSSNGridDerivatives.hpp`
- Evolution kernels: `Evolution/BSSNEvolution*.hpp`
- Geometry/projection: `Geometry/*.hpp`
- Constraints/monitoring: `Constraints/*.hpp`
- Time integrator: `TimeIntegration/BSSNRK4.hpp`
- Initialization: `InitialData/BSSNInitialData.hpp`

## 2. State Vector and Conventions

### 2.1 Evolved variables

At each cell, the evolved variables are:

- Scalars:
  - `alpha` (lapse)
  - `chi` (conformal factor, with `gamma_ij = chi^{-1} * gamma_tilde_ij`)
  - `K` (trace of extrinsic curvature)
  - `Theta` (Z4 scalar constraint variable)
- Vectors:
  - `beta^i` (shift)
  - `B^i` (Gamma-driver auxiliary)
  - `tildeGamma^i` (contracted conformal connection)
  - `Z^i` (Z4 vector, contravariant)
- Symmetric tensors in packed 6-component form:
  - `gamma_tilde_ij`
  - `gamma_tilde^ij`
  - `A_tilde_ij`

### 2.2 Packed symmetric index convention

The 6 packed components follow:

- `0 -> XX`
- `1 -> XY`
- `2 -> XZ`
- `3 -> YY`
- `4 -> YZ`
- `5 -> ZZ`

### 2.3 Geometry relations

The implementation consistently uses:

- `gamma_ij = chi^{-1} * gamma_tilde_ij`
- `gamma^ij = chi * gamma_tilde^ij`
- `Khat = K - 2*Theta` in parts of the RHS assembly

Helper routines:

- `Khat(...)`: `Evolution/BSSNEvolutionCommon.hpp`
- `guard_chi_div(...)`: `Evolution/BSSNEvolutionCommon.hpp`

## 3. Mathematical Formulation in Code

The subsystem is BSSN with Z4c-driven terms added in the evolved connection and constraints.
The exact algebra follows the kernel implementations; this section summarizes the principal form.

### 3.1 Conformal factor

Implemented in `Evolution/BSSNEvolutionChi.hpp`:

- `d_t chi = beta^k d_k chi + (2/3) chi (alpha K - d_k beta^k) + KO(chi)`

### 3.2 Conformal metric

Implemented in `Evolution/BSSNEvolutionGammaTilde.hpp`:

- `d_t gamma_tilde_ij = adv + Lie_beta(gamma_tilde)_ij - (2/3) gamma_tilde_ij d_k beta^k`
- source part includes `-2 alpha A_tilde_ij`
- KO dissipation is added component-wise

### 3.3 Trace-free extrinsic curvature

Implemented in `Evolution/BSSNEvolutionATilde.hpp`:

- Uses the trace-free combination of `chi[-D_i D_j alpha + alpha R_ij]`
- Adds quadratic term `alpha (K A_tilde_ij - 2 A_tilde_{ik} A_tilde^k_j)`
- Includes Lie/advection transport and KO dissipation
- Adds Z4 Ricci correction via `compute_RicciZ4(...)`

### 3.4 Mean curvature / Khat handling

Implemented in `Evolution/BSSNEvolutionK.hpp` and recomposed in `TimeIntegration/BSSNRK4.hpp`:

- The kernel computes RHS for `Khat = K - 2 Theta`.
- The stepper later reconstructs `RHS(K) = RHS(Khat) + 2*RHS(Theta)`.

This split is deliberate and keeps the Z4-coupled source terms consistent.

### 3.5 Contracted connection evolution

Implemented in `Evolution/BSSNEvolutionGamma.hpp`:

- Includes advection, stretching, shift-Laplacian pieces, lapse-gradient coupling, and geometric source terms.
- Includes Z4-driven couplings (`Theta`, reconstructed `Z/chi`, `kappa*` parameters).
- `Z/chi` is reconstructed from `tildeGamma - gamma_metric` on the RHS path.

### 3.6 Z4 scalar and vector

Implemented in `Evolution/BSSNEvolutionZ4C.hpp`:

- `compute_rhs_Theta(...)` builds a geometric source from Ricci scalar, `A_tilde` contraction, `K`, and reconstructed Z4 couplings, then adds damping/advection/KO.
- `compute_rhs_Z(...)` is diagnostic-only: the stored contravariant `Z^i` field is reconstructed from `tildeGamma` and its RHS is identically zero.

## 4. Gauge System

Gauge controls live in `Evolution/BSSNEvolutionGauge.hpp` through `GaugeParameters<T>`.

### 4.1 Lapse

`compute_rhs_alpha(...)` supports a generalized source of the form:

- advection term
- multiplicative factor `f = lapse_oplog*lapse_harmonicf + lapse_harmonic*alpha`
- source `- f * alpha * lapse_K` where `lapse_K` can use `Khat` if `use_theta_in_lapse=true`

Optional slow-start damping is available through:

- `slow_start_lapse`
- `ssl_damping_amp`
- `ssl_damping_time`
- `ssl_damping_index`

### 4.2 Shift and B-driver

Two modes are implemented:

- Direct-shift mode (`use_direct_shift_rhs=true`):
  - `beta^i` evolved directly from `Gamma`/advection/damping/harmonic couplings.
  - `B^i` RHS is set to zero.
- Gamma-driver mode (`use_direct_shift_rhs=false`):
  - `beta^i` includes `beta_B_coeff * B^i` plus advection.
  - `B^i` evolves from `rhs_Gamma`, advection, and damping.

Important practical point:

- `use_direct_shift_rhs` defaults to `false` in `GaugeParameters`.

## 5. Geometry and Algebraic Constraints

### 5.1 Contracted Gamma and Christoffels

- `Geometry/BSSNCHristoffelTilde.hpp`:
  - `compute_tildeGamma_symbols(...)` builds local `tildeGamma^i_{jk}`.
  - `compute_tildeGamma_contracted(...)` recomputes `tildeGamma^i = -d_j gamma_tilde^{ij}`.

### 5.2 Ricci decomposition

`Geometry/BSSNRicci.hpp` computes:

- `R_ij = R_tilde_ij + R_chi_ij`

with high-order derivatives and explicit conformal contributions.

### 5.3 Projection and invariant enforcement

`Geometry/BSSNProjection.hpp` provides:

- determinant renormalization (`det(gamma_tilde)=1`)
- trace projection (`tr(A_tilde)=0`)
- inverse recomputation
- optional re-sync of contracted Gamma

Key entry points:

- `project_bssn_state(...)`
- `project_bssn_after_update(...)`

Invariants are checked via `Geometry/BSSNInvariants.hpp` helpers used across init, Ricci, and constraints.

## 6. Spatial Discretization

All derivative operators are centralized in `Derivatives/BSSNGridDerivatives.hpp`.

### 6.1 Orders

Runtime-selectable maximum order:

- Order 6 (default)
- Order 4 (optional)

API:

- `set_max_spatial_derivative_order(int)`
- `max_spatial_derivative_order()`

### 6.2 First and second derivatives

- Centered high-order stencils for `Dx,Dy,Dz` and `Dxx,Dyy,Dzz`.
- Mixed derivatives:
  - compact order-2 (`Dxy`, `Dxz`, `Dyz`)
  - order-4 tensor-product (`Dxy4`, `Dxz4`, `Dyz4`), used in high-sensitivity geometry pieces.

### 6.3 Upwind advection

Advection uses sign-dependent biased stencils:

- `Dx_upwind_ptr`, `Dy_upwind_ptr`, `Dz_upwind_ptr`

These are used in transport terms of gauge and geometry variables.

### 6.4 KO dissipation

KO operator:

- `KO6_axis_ptr(...)`

Applied in most RHS kernels as:

- `(ko_sigma / dx) * (KO_x + KO_y + KO_z)`

The module currently keeps KO scaling constant (`update_ko_scale` sets scale to `1.0`).

## 7. Time Integration (Low-Storage 4-Stage RK)

Implemented in `TimeIntegration/BSSNRK4.hpp`.

### 7.1 CFL estimate

`compute_dt_cfl(...)` uses:

- `min(dx,dy,dz)`
- `max |beta|` on interior
- gauge speed factor

`dt = cfl * min_dx / (max_beta + gauge_speed)`

### 7.2 Stage coefficients

The stepper uses the fixed 4-stage coefficient arrays:

- `gam0_ref`
- `gam1_ref`
- `beta_ref`
- `delta_ref`

These are embedded in `BSSNRK4.hpp` and define the low-storage update.

### 7.3 Stage pipeline

Per stage, the sequence is:

1. Build/update stage reference state.
2. Apply solution halos and rebuild geometry (`prepare_state_for_rhs`).
3. Evaluate all RHS kernels.
4. Optional RHS Sommerfeld correction on boundary surfaces.
5. Recompose `rhs(K)` from `rhs(Khat)`.
6. Apply explicit low-storage update.
7. Enforce algebraic constraints.
8. Optional smooth floors on `alpha` and `chi`.

Post-step, diagnostics and constraint monitors are evaluated.

## 8. Halos and Boundary Conditions

Implemented in `Grid/BSSNGridOperations.hpp`.

### 8.1 Boundary functors

Available boundary functors include:

- `BoundaryClamp`
- `BoundarySponge`
- `BoundaryRadiative`

`apply_halos_grid<Boundary>(...)` applies all physical/halo passes for every field in `BSSNGridSoA`.

### 8.2 Radiative boundary behavior

`BoundaryRadiative` supports:

- per-face RHS Sommerfeld masks (`rhs_sommerfeld_face[axis][side]`)
- per-face reflective parity masks (`reflective_face[axis][side]`)
- runtime configuration of the outer-boundary collar and compatibility knobs from the stepper

Vector and tensor parity on reflective faces is component-aware. For non-reflective faces, the
solution ghosts are extrapolated rather than filled with a radiative Sommerfeld state, matching the
GRChombo-style split where outgoing behavior is imposed on the RHS instead of on the state halos.

### 8.3 RHS-level Sommerfeld correction

Rather than imposing Sommerfeld data in the solution halos, the RK stepper can apply a direct RHS
Sommerfeld operator near boundary surfaces:

- enabled by `GaugeParameters::apply_rhs_sommerfeld`
- implemented in `BSSNRK4::apply_rhs_sommerfeld(...)`
- the operator mirrors GRChombo's local Cartesian Sommerfeld form on every boundary point:
  - `rhs(u) = -sum_i (\partial_i u) x^i / r + (u_inf - u) / r`
  - one-sided second-order derivatives are used on the outermost grid cells
- the bulk RHS sweep skips active radiative faces by the effective Sommerfeld collar, so cells whose
  high-order stencils would touch extrapolated ghosts are replaced by the outgoing RHS instead
- `K` is still treated through `Khat = K - 2 Theta` internally, then recomposed after the RHS pass
- Tensorium still does not evolve GRChombo-style grown-grid boundary ghosts, so the outermost
  physical surface remains the closest analogue to GRChombo's RHS boundary fill

## 9. Initial Data Pipelines

Implemented in `InitialData/BSSNInitialData.hpp` and `Solvers/BSSNConstrainSolver.hpp`.

### 9.1 Analytic initializers

Provided initializers include:

- `minkowski(...)`
- `isotropic schwarzschild` (single/binary puncture style)
- `kerr_schild_single(...)`

### 9.2 Bowen-York binary puncture

`binary_bowen_york_puncture_init(...)` uses:

- Bowen-York `A_tilde` assembly (`fill_Atilde_bowen_york_binary`)
- optional Lichnerowicz solve (`solve_lichnerowicz_u_SOR`)

The Lichnerowicz solver is red-black SOR on a 2nd-order Laplacian discretization.

### 9.3 Interpolated puncture initialization

`binary_bowen_york_puncture_interpolated_init(...)` has two runtime branches:

- If `TENSORIUM_HAS_TWOPUNCTURES_C` is enabled:
  - solve external spectral puncture backend
  - required upstream TwoPuncturesC code: `https://bitbucket.org/bernuzzi/twopuncturesc/src/master/`
  - expected dependency checkout: `git clone https://bitbucket.org/bernuzzi/twopuncturesc.git ../TwoPuncturesC`
  - interpolate to Cartesian grid
  - map into BSSN state and reproject
- Otherwise:
  - generate a seed Bowen-York grid
  - trilinear interpolation onto target grid

Both paths finish with projection/invariant checks and geometry rebuild.

## 10. Constraint Evaluation and Monitoring

### 10.1 Constraint fields

`Constraints/BSSNConstraintsGrid.hpp` computes:

- Hamiltonian constraint `H`
- Momentum constraint `M_i`
- Gamma constraint surrogate from `tildeGamma + div(gamma_tilde_inv)`

### 10.2 Reduced monitoring stats

`Constraints/BSSNConstraintMonitoring.hpp` computes:

- `max_H`, `l2_H`
- `l2_theta`, `l2_Z`, `l2_M`
- `max_trace_A`, `max_det_drift`
- `samples`

The RK stepper calls these diagnostics and can forward them via callback hooks.

## 11. Runtime Controls and Stability Knobs

Most evolution controls are in `GaugeParameters<T>`.

Frequently tuned knobs:

- Gauge:
  - `use_direct_shift_rhs`
  - `shift_eta`, `shift_Gamma`, `shift_advect`
  - `lapse_oplog`, `lapse_harmonic*`, `slow_start_lapse*`
- Z4/constraint damping:
  - `kappa1`, `kappa2`, `kappa3`, `kappa_z`
  - `evolve_Z` compatibility knob for the auxiliary/exported `Z_i` field
- Numerical stabilization:
  - `ko_sigma`
  - `chi_div_floor`
  - `alpha_floor`, `chi_floor`
- Boundary coupling:
  - `apply_rhs_sommerfeld`
  - per-face masks in `BoundaryRadiative`

## 12. Performance Model

### 12.1 Memory layout

`BSSNGridSoA` is structure-of-arrays with aligned allocations and shared strides.
This favors contiguous component loops and predictable streaming behavior.

### 12.2 Parallelism

Most hot loops use:

- `#pragma omp parallel for collapse(...)`

Thread-level parallelism is CPU/OpenMP-centric.

### 12.3 Kernel timing

`TimeIntegration/BSSNPerfTimers.hpp` instruments kernels (`Gamma`, `A_tilde`, `Theta`, etc.).
`BSSNRK4` prints timers per step when enabled.

## 13. Practical Workflow

A typical evolution loop using this subsystem is:

1. Allocate `BSSNGridSoA` with chosen `nx,ny,nz,ng,dx,dy,dz`.
2. Fill initial data (`binary_bowen_york...` or another initializer).
3. Configure `GaugeParameters` and boundary face masks.
4. Build `BSSNRKStepper<T, BoundaryRadiative>`.
5. Compute `dt` from `compute_dt_cfl(...)`.
6. Repeatedly call `step(grid, dt, step_index)`.
7. Export slices/constraints/trackers via callbacks and test harness utilities.

## 14. Current Limitations and Extension Targets

Current limitations:

- Single-grid architecture in this module (no integrated AMR hierarchy update).
- No integrated multi-rank BSSN runtime path yet.
- Vacuum-only RHS terms (no matter coupling path wired through kernels).

Natural extension targets:

- Matter source integration in RHS.
- AMR prolongation/restriction for BSSN fields.
- Fully integrated MPI halo exchange for distributed runs.
- Higher-level problem setup API (config-driven run construction).

## 15. Source Map (Quick Navigation)

- `Fields/BSSNGridSoA.hpp`
- `Derivatives/BSSNGridDerivatives.hpp`
- `Evolution/BSSNEvolutionChi.hpp`
- `Evolution/BSSNEvolutionGammaTilde.hpp`
- `Evolution/BSSNEvolutionATilde.hpp`
- `Evolution/BSSNEvolutionK.hpp`
- `Evolution/BSSNEvolutionGamma.hpp`
- `Evolution/BSSNEvolutionZ4C.hpp`
- `Evolution/BSSNEvolutionGauge.hpp`
- `Geometry/BSSNRicci.hpp`
- `Geometry/BSSNProjection.hpp`
- `Constraints/BSSNConstraintsGrid.hpp`
- `Constraints/BSSNConstraintMonitoring.hpp`
- `InitialData/BSSNInitialData.hpp`
- `Solvers/BSSNConstrainSolver.hpp`
- `TimeIntegration/BSSNRK4.hpp`

## 16. Notes on Code-Theory Consistency

This guide intentionally tracks the implementation rather than presenting an idealized textbook system.
When behavior appears different from a canonical equation set, trust the kernel source first:

- evolution details: `Evolution/*.hpp`
- projection and invariants: `Geometry/BSSNProjection.hpp`, `Geometry/BSSNInvariants.hpp`
- step orchestration: `TimeIntegration/BSSNRK4.hpp`

That is the authoritative behavior for current Tensorium runs.

## Test for a two puncture initial data (no merge but BBH drift over lapse):

```bash
  OMP_NUM_THREADS=24 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_DYNAMIC=FALSE \
  TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT=1 \
  TENSORIUM_MOVING_PUNCTURE_GRID_N=112 \
  TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH=10.0 \
  TENSORIUM_MOVING_PUNCTURE_SEPARATION=3.2 \
  TENSORIUM_MOVING_PUNCTURE_MOMENTUM=0.85 \
  TENSORIUM_MOVING_PUNCTURE_RADIAL_MOMENTUM=0.10 \
  TENSORIUM_MOVING_PUNCTURE_CFL=0.09 \
  TENSORIUM_MOVING_PUNCTURE_KO_SIGMA=0.1 \
  TENSORIUM_MOVING_PUNCTURE_STEPS=4000 \
  TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_IX3=0 \
  TENSORIUM_MOVING_PUNCTURE_REFLECTIVE_OX3=0 \
  TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_IX3=1 \
  TENSORIUM_MOVING_PUNCTURE_RHS_SOMMERFELD_OX3=1 \
  TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_ON_DRIFT=1 \
  TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_MODE=chi \
  TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_CELLS=1.5 \
  TENSORIUM_MOVING_PUNCTURE_TRACKER_DRIFT_WARN_CELLS=0.8 \
  TENSORIUM_MOVING_PUNCTURE_STATE_LOG_STRIDE=1 \
  TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_EXPORT_STRIDE=2 \
  TENSORIUM_MOVING_PUNCTURE_EXPORT_CONSTRAINT_SLICES=1 \
  TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_SLICE_STRIDE=2 \
  ./build/Tests/TensoriumTests --test bssn.viz.moving_puncture_interpolate
```

Moving-puncture boundary safety knobs:

- `TENSORIUM_MOVING_PUNCTURE_ALLOW_REFLECTIVE_BC=1` is now required to opt into reflective faces.
- `TENSORIUM_MOVING_PUNCTURE_FAIL_ON_GAUGE_BC_MISMATCH=1` turns the boundary characteristic
  warning into a hard error when a face has no radiative gauge-speed budget.
- `TENSORIUM_MOVING_PUNCTURE_ENABLE_SPONGE=0/1` toggles the optional outer sponge layer. It is off
  by default for GRChombo-style moving-puncture runs.
- `TENSORIUM_MOVING_PUNCTURE_SPONGE_WIDTH`, `..._SPONGE_STRENGTH`, `..._SPONGE_EXPONENT`
  tune the damping profile.
- `TENSORIUM_MOVING_PUNCTURE_RADIATIVE_COLLAR_WIDTH` controls how many physical layers near each
  radiative face are overwritten by the outgoing RHS collar instead of using ghost-fed bulk RHS.
- `TENSORIUM_MOVING_PUNCTURE_KO_BOUNDARY_WIDTH` and `..._KO_BOUNDARY_FLOOR` taper KO near the
  boundary to avoid filtering directly on radiative ghost closures.
- Moving-puncture runners also log `[bc.char]` summaries that estimate face-wise gauge speeds
  from the initial lapse profile and compare them to the configured radiative budget.
