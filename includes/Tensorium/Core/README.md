
# Tensorium — Core Module

This directory contains the core linear algebra structures and operations of the Tensorium library. It provides the foundational vector, matrix, tensor, and derivative tools used across all higher-level modules, including symbolic computation, spectral methods, and numerical relativity.

## Purpose

The module offers:
- Generic, aligned data structures for `Vector`, `Matrix`, and `Tensor` with SIMD support
- Fundamental linear algebra operations (addition, scaling, dot product, etc.)
- Matrix multiplication, transposition, inversion, and LU decomposition
- Generalized tensors of arbitrary rank with contraction and tensor product
- First and fourth-order derivative schemes (finite difference and spectral)

## Structure

```
Core/
├── Vector.hpp         // Fixed-size aligned vectors with SIMD operations
├── Matrix.hpp         // Dense matrix with optimized arithmetic and inversion
├── Tensor.hpp         // Arbitrary-rank tensor with contraction and layout
├── LinearSolver.hpp   // Gaussian, Jacobi and LU solvers
├── Derivate.hpp       // Centered finite difference and spectral derivatives
```

## Features

### Vector
- Generic `Vector<T>` with aligned storage
- Operations: add, subtract, scale, dot product, cross product (3D), norm
- Support for SIMD acceleration (SSE/AVX/AVX512)

### Matrix
- `Matrix<T>` with dense layout and cache-aware optimizations
- SIMD-aware matrix multiplication with unrolling and OpenMP support
- LU decomposition, determinant, trace, inverse, transpose

### Tensor
- `Tensor<T, Rank>` with full support for arbitrary-rank tensor algebra
- Compile-time rank enforcement
- Operations: contraction, tensor product, slicing, transposition

### Derivative
- `Derivate<T>` and `DerivateND<T, Rank>` structures
- First-order and fourth-order centered finite difference derivatives
- Optional support for spectral methods via `Spectral.hpp`

### Linear Solvers
- Solvers for linear systems: Gaussian elimination, Jacobi iterations
- Designed to work with both matrices and tensors

## Internal Dependencies

- Depends on `SIMD/` for vectorization and aligned allocation
- Uses optional BLAS acceleration (`TENSORIUM_USE_CBLAS`) when available
- Compatible with `Symbolics/` and `DiffGeometry/` layers

## Example Usage

```cpp
tensorium::Matrix<double> A(3, 3);
tensorium::Vector<double> b(3);

// Fill A and b...

auto x = tensorium::gauss_solve(A, b);

tensorium::Matrix<double> B = A.transpose();
auto det = tensorium::det_mat(A);

tensorium::Tensor<double, 3> T({4, 4, 4});
auto T2 = tensorium::contract_tensor<0, 2>(T);
```

## Status

The module is stable and used across all parts of the Tensorium library. It is designed for performance-critical numerical code and supports both analytical and simulation backends.

## Notes

- Matrix sizes are expected to fit in cache (optimized kernels exist for 4×4, 8×8, 16×16).
- Tensor operations are layout-aware for compatibility with differential geometry modules.
