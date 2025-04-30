# Morpheus_lib

![Screenshot from 2025-04-29 14-43-09](https://github.com/user-attachments/assets/41772890-d460-4f72-bda1-8cfc23eddfa3)


**Morpheus_lib** is a high-performance scientific C++ library designed for demanding computational domains such as **numerical relativity**, **machine learning (ML)**, **deep learning (DL)**, **artificial intelligence (AI)**, and **scientific simulations**.

It provides a modern, extensible infrastructure for efficient vector, matrix, and tensor computations by leveraging:
- **SIMD acceleration** (SSE, AVX2, AVX512),
- **Multithreading** with OpenMP,
- And soon, **distributed computing** via MPI.

The core philosophy of Morpheus_lib is to combine:
- **Raw performance**, through low-level SIMD optimization,
- **Modularity and clarity**, using a modern, header-only C++17 design,
- **Python interoperability**, via PyBind11, for seamless integration with scientific Python workflows.

This library is built with the goal of empowering projects that require both speed and flexibility, such as:
- Simulating curved spacetime and relativistic matter (e.g. BSSN formalism, GRHD, GRMHD),
- Custom neural network training and inference on CPU,
- Fast manipulation of large scientific datasets and image matrices,
- Research and education projects needing intuitive yet high-performance numerical tools.

## Highlights

- Optimized `Tensor`, `Vector` and `Matrix` classes with aligned memory
- AVX2/FMA SIMD acceleration (fallback on SSE when needed)
- Custom allocator using `posix_memalign` for proper vectorization
- OpenMP and MPI support
- Matrix/Tensor multiplication optimized with blocking, unrolling, and OpenMP
- Python bindings using `pybind11` for seamless integration with Python
- A symbolic parser to compute problems with a LaTex structure (in comming)
- Optional benchmark against BLAS (OpenBLAS, MKL)

## TODO
- Symbolic LaTeX parser
- Tensor operators
- General relativity / differential geometry classes dans methods
- Full MPI support 
- SSE fallback (curently working on)
- Spectral Methdods (Chebychev/Fourrier)
- Backward FDM
- Some optimizations

## Build Instructions

### Requirements

- C++17 compiler with AVX2/FMA support or AVX512 if avalaible on your plateform (Intel compilers will be added later)
- fopenmp
- MPI
- libmemkind-dev (if you are using Xeon Phi knight landing CPU)
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

### Build C++ only for special targets and options

```bash
make                # Default AVX2
make help	    # Show differents compile options 
make AVX512=true    # AVX512
make USE_KNL=true   # MCDRAM Memkind HBW (Xeon phi KNL)
make DEBUG=true     # debug symbols
make VERBOSE=true   # VERBOSE log
make benchmark      # BLAS vs Morpheus mat_mult benchmark
```

The Python module will be created as a .so file in the pybuild/ directory.
### Exemple using in C++
```cpp
#include "Morpheus.hpp"

int main() {
	Vector<float> v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	Vector<float> v2 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

	std::cout << "\n[v1] + [v2]:\n";
	morpheus::add_vec(v1, v2).print();

	std::cout << "\n[v1] - [v2]:\n";
	morpheus::sub_vec(v1, v2).print();

	std::cout << "\n[v1] * 0.5:\n";
	morpheus::scl_vec(v1, 0.5f).print();

	Matrix<float> m1(2, 8); 
	Matrix<float> m2(2, 8);

	for (size_t i = 0; i < m1.rows; ++i)
		for (size_t j = 0; j < m1.cols; ++j) {
			m1(i, j) = i * 10 + j;
			m2(i, j) = 1.0f;
		}

	std::cout << "\n[m1] + [m2]:\n";
	morpheus::add_mat(m1, m2).print();

	std::cout << "\n[m1] - [m2]:\n";
	morpheus::sub_mat(m1, m2).print();

	std::cout << "\n[m1] * 2.0:\n";
	morpheus::scl_mat(m1, 2.0f).print();
}
```
### Example using in Python
```python
from morpheus import *

matA = Matrix(2, 3)
matA.fill([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

matB = Matrix(2, 3)
matB.fill([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

print("matA + matB =")
morph.add_mat(matA, matB).print()

print("matA - matB =")
morph.sub_mat(matA, matB).print()

print("matA * 2.0 =")
morph.scl_mat(matA, 2.0).print()

v = Vector([1.0, 2.0, 3.0])
v2 = Vector([4.0, 5.0, 6.0])

print("v =", v)
print("len(v) =", len(v))
print("v + v2 =", morph.add_vec(v, v2))
print("v - v2 =", morph.sub_vec(v, v2))
print("v * 2.0 =", morph.scl_vec(v, 2.0))
print("dot(v, v2) =", morph.dot_vec(v, v2))
print("norm_1(v) =", morph.norm_1(v))
print("norm_2(v) =", morph.norm_2(v))
print("norm_inf(v) =", morph.norm_inf(v))
print("cosine(v, v2) =", morph.cosine(v, v2))
print("lerp(v, v2, 0.5) =", morph.lerp(v, v2, 0.5))
```


### Run tests
```bash
chmod +x setup.sh && ./setup.sh
```


