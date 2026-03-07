# Tensorium — DiffGeometry Module

This directory provides the core components for differential geometry computations in general relativity. It supports both symbolic 4D analytical calculations and the foundations for ADM/BSSN-based 3+1 numerical simulations.

## Purpose

This module enables:
- Representation and manipulation of the spacetime metric
- Computation of Christoffel symbols (first and second kind)
- Construction of curvature tensors: Riemann, Ricci, and Ricci scalar
- BSSN-related helpers (`BSSN/`) and the production grid evolution path (`BSSN_Grid/`)

## Structure

```
DiffGeometry/
├── Metric.hpp              // Spacetime metric representation
├── ChristoffelSymbol.hpp   // Computation of Γ^k_{ij}
├── RicciTensor.hpp         // Ricci tensor computation
├── RiemannTensor.hpp       // Full Riemann tensor computation
├── Tensor.hpp              // Basic tensor structures
├── BSSN/                   // Legacy/pointwise BSSN helpers
└── BSSN_Grid/              // Main finite-difference BSSN/Z4c grid implementation
```

## Features

- `Metric` provides the metric components, inverse metric, and determinant.
- `ChristoffelSymbol` computes the Christoffel symbols from metric derivatives.
- `RicciTensor` constructs the Ricci tensor by contracting the Riemann tensor.
- `RiemannTensor` provides full access to the Riemann curvature tensor \( R^\mu_{\ \nu\rho\sigma} \).
- `BSSN/` provides pointwise and helper routines.
- `BSSN_Grid/` is the active path for current evolution tests and moving-puncture workflows.

## Internal Dependencies

- Uses `Tensor.hpp` from `Core/` for generic tensor representation.
- Uses `DerivateND` from `Core/` for partial derivatives (via centered difference or spectral methods).
- Fully self-contained, with no external library dependency.

## Current Supported Metrics

- Minkowski (flat spacetime)
- Schwarzschild (non-rotating black hole)
- Kerr (rotating black hole)
- Kerr-Schild
- FLRW

## Example: Compute Riemann and Ricci Tensors from Kerr Metric

```cpp
constexpr size_t dim = 4;

tensorium::Vector<double> X(dim);
X(0) = 0.0;
X(1) = 10.0;
X(2) = M_PI / 2.0;
X(3) = 0.0;

tensorium::Tensor<double, 2> g({dim, dim});
tensorium::Tensor<double, 2> g_inv({dim, dim});

tensorium_RG::Metric<double> metric("kerr", 1.0, 0.8);
metric(X, g);

g_inv = tensorium::inv_mat_tensor(g);

auto gamma = tensorium::compute_christoffel(X, 1e-5, g, g_inv, metric);
auto Riemann = tensorium::compute_riemann_tensor<double>(X, 1e-5, metric);
auto Ricci = tensorium::contract_tensor<0, 2>(Riemann); // Ricci^μ_μ = R_{ρσ}
```

## Status

The module is stable for 4D analytical computations. Symbolic initializations and automation through the `Symbolics/` and BSSN modules are under active development.

## References

- M. Alcubierre, *Introduction to 3+1 Numerical Relativity*
- T. W. Baumgarte & S. L. Shapiro, *Numerical Relativity: Solving Einstein's Equations on the Computer*
