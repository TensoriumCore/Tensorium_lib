# Tensorium — BSSN Module

This directory contains the building blocks for the BSSN (Baumgarte–Shapiro–Shibata–Nakamura) formulation used in numerical relativity. The headers implement helper routines to compute conformal variables, derivatives and curvature terms from a given spacetime metric.

## Header Overview

- `BSSNSetup.hpp` – defines `BSSNGrid` and the `BSSN` class. It initializes all BSSN variables at a spatial point: lapse, shift, conformal metric \(\tilde{\gamma}_{ij}\), its derivatives and inverse, Christoffel symbols, the extrinsic curvature \(K_{ij}\) and the trace–free tensor \(\tilde{A}_{ij}\).
- `BSSNMetricUtils.hpp` – utilities for generating conformal metric fields on a 3D grid (e.g. `generate_conformal_metric_field`).
- `BSSNDerivatives.hpp` – finite difference helpers to compute partial derivatives of scalars, vectors and tensors.
- `BSSNChristoffel.hpp` – computes the conformal Christoffel symbols \(\tilde{\Gamma}^{k}_{ij}\) and provides routines to differentiate a 5‑D metric field.
- `BSSNTildeChristoffel.hpp` – contracts the conformal Christoffel symbols with the inverse metric to form \(\tilde{\Gamma}^{i}\).
- `BSSNContractedChristoffel.hpp` – evaluates the contracted connection \(\Gamma^{i}_{ij} = -\tfrac{3}{2}\,\partial_j \ln\chi\).
- `BSSNextrinTensor.hpp` – computes the extrinsic curvature tensor \(K_{ij}\) from metric time derivatives, lapse and shift.
- `BSSNAtildeTensor.hpp` – builds the trace‑free conformal extrinsic curvature tensor \(\tilde{A}_{ij}\).

## Status

The BSSN implementation is experimental and currently limited to initializing variables at a single grid point. Evolution equations and advanced gauge conditions are not yet available, and derivative operators rely on straightforward finite differences. APIs may change as development continues.

## Usage

`setup_BSSN_grid` from `includes/Tensorium/Functionnal/FunctionnalRG.hpp` provides a simple entry point:

```cpp
using tensorium::Vector;
Vector<double> X(3);       // (x, y, z)
X(0) = 10.0; X(1) = 0.0; X(2) = 0.0;

tensorium_RG::Metric<double> metric("kerr", 1.0, 0.8);
double dx = 0.1, dy = 0.1, dz = 0.1;

auto bssn = tensorium::setup_BSSN_grid(X, metric, dx, dy, dz);

const auto& gamma_tilde = bssn.grid.gamma_tilde[0];
const auto& A_tilde     = bssn.grid.A_tildeTensor[0];
```

The returned `tensorium_RG::BSSN` object stores all conformal and curvature quantities for that spatial coordinate.
