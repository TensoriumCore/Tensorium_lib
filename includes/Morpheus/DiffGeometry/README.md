# Morpheus — DiffGeometry Module

This directory provides the core components for differential geometry computations in general relativity. It serves as the foundation for implementing the ADM and BSSN formalisms used in numerical relativity.

## Purpose

The module enables:
- Representation and manipulation of the spacetime metric
- Computation of Christoffel symbols (first and second kind)
- Construction of curvature tensors: Riemann, Ricci, and Ricci scalar
- Implementation of the BSSN (conformal) formalism setup for GRID 3+1 problems (To be done)

## Structure

```
DiffGeometry/
├── Metric.hpp              // Spacetime metric representation
├── ChristoffelSymbol.hpp   // Computation of Γ^k_{ij}
├── RicciTensor.hpp         // Ricci tensor computation
├── RiemannTensor.hpp       // Full Riemann tensor computation
├── Tensor.hpp              // Basic tensor structures
└── BSSN/                   // Conformal BSSN variables and evolution
```

## Features

- `Metric` provides the metric components, inverse metric, and determinant.
- `ChristoffelSymbol` computes Christoffel symbols from metric derivatives.
- `RicciTensor` builds the Ricci tensor from Christoffel symbols.
- `RiemannTensor` constructs the full Riemann curvature tensor.
- `BSSN/` (TODO) contains conformal metric decomposition, trace-free extrinsic curvature, and conformal connection functions.

## Internal Dependencies

- Depends on `Tensor.hpp` from `Core/` for multidimensional tensor structures.
- Uses `DerivateND` from `Core/` for partial derivatives (finite difference or spectral).
- No external libraries are used.

## Status

The module is functional and actively used in BSSN-based 3+1 simulations. Symbolic support for automatic tensor generation (via `Symbolics/`) is under development.

## Example Usage to compute Riemann Tensor and contract to Ricci in Kerr metric 

```cpp
	constexpr size_t dim = 4;

	morpheus::Vector<double> X(dim);
	X(0) = 0.0; 
	X(1) = 10.0;
	X(2) = M_PI / 2.0;
	X(3) = 0.0;

	morpheus::Tensor<double, 2> g({dim, dim});
	morpheus::Tensor<double, 2> g_inv({dim, dim});

	morpheus_RG::Metric<double> metric("kerr", 1.0, 0.8);
	metric(X, g);

	g_inv = morpheus::inv_mat_tensor(g); 
	auto gamma = morpheus::compute_christoffel(X, 1e-5, g, g_inv, metric);
	auto R = morpheus::compute_riemann_tensor<double>(X, 1e-5, morpheus_RG::Metric<double>("kerr", 1.0, 0.8));
	morpheus::contract_tensor<0, 1>(R);
```

