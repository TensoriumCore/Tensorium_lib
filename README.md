# Morpheus_lib

**Morpheus_lib** is a high-performance C++ linear algebra library with SIMD vectorization (AVX2/FMA) and optional Python bindings via `pybind11`.

It provides fast and efficient implementations of `Vector` and `Matrix` operations using SIMD instructions, custom allocators for aligned memory, and highly optimized algorithms.

## Features

- Header-only SIMD linear algebra library
- Aligned memory allocation using `posix_memalign`
- Optimized `Vector` and `Matrix` operations (add, sub, scale, dot, norms, etc.)
- AVX2/FMA acceleration (automatically dispatches to the best supported ISA)
- Matrix × Matrix multiplication optimized with blocking, unrolling and OpenMP
- Python bindings using `pybind11` for interactive usage and rapid testing
- Optional comparison with BLAS (`cblas_sgemm`, `cblas_dgemm`)

## Build Instructions

### Requirements

- C++17 compiler with AVX2/FMA support
- CMake ≥ 3.16
- Python ≥ 3.8 (for bindings)
- `pybind11`
- `OpenBLAS` or another BLAS backend (optional, for benchmarks)

### Build C++ Library and Python Module

```bash
mkdir pybuild
cd pybuild
cmake ..
make -j4
```
The Python module will be created as a .so file in the pybuild/ directory.
### Exemple using in C++
```cpp
#include "Vector.hpp"
#include "Matrix.hpp"

int main() {
    morpheus::Vector<float> v1 = {1.0f, 2.0f, 3.0f};
    morpheus::Vector<float> v2 = {4.0f, 5.0f, 6.0f};

    v1.add(v2);
    v1.print(); // prints: [5, 7, 9]

    morpheus::Matrix<float> A(512, 512), B(512, 512);
    auto C = A.mul_mat(B);
}
```
### Example using in Python
```python
from morpheus import Vector, add, scl

a = Vector([1.0, 2.0, 3.0])
b = Vector([4.0, 5.0, 6.0])

c = add(a, b)
print("a + b =", list(c))

d = scl(a, 2.5)
print("a * 2.5 =", list(d))
```


