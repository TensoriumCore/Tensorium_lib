# Morpheus_lib

**Morpheus_lib** is a high-performance, header-only C++ linear algebra library with AVX512/AVX2/FMA SIMD acceleration and native Python bindings via `pybind11`.

It is designed to be fast, portable, and efficient in both C++ and Python environments, making it usable for scientific computing, numerical simulations, and real-time applications.

## Highlights

- Optimized `Vector` and `Matrix` classes with aligned memory
- AVX2/FMA SIMD acceleration (fallback on SSE when needed)
- Custom allocator using `posix_memalign` for proper vectorization
- Matrix multiplication optimized with blocking, unrolling, and OpenMP
- Python bindings using `pybind11` for seamless integration with Python
- Optional benchmark against BLAS (OpenBLAS, MKL)

## Build Instructions

### Requirements

- C++17 compiler with AVX2/FMA support (Intel compilers will be added later)
- fopenmp
- MPI
- CMake ≥ 3.16
- Python ≥ 3.8 (for Python bindings)
- `pybind11` installed (`pacman -S python-pybind11` on Arch, or `pip install pybind11 --user`)
- OpenBLAS (optional, for benchmarking with BLAS)


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

### Run tests
```bash
chmod +x setup.sh && ./setup.sh
```
