![Nouveau projet](https://github.com/user-attachments/assets/5f75f1f9-999d-410b-971e-ba3bd5e8b5e9)
# Tensorium
### !!!DISCLAMER!!! 
Tensorium is still in the early development phase, and many of its features work, but I'm not yet convinced of the solidity of some of them (especially the tensor manipulations).
The python binding is usable without any other python librairy, but I'm still working on it to make it all clean and usable using a simple pip3 install (see the Jupiter Notebook).
note: macos version does not compile at the moment because Apple always want to be different (in clang too...)

**Tensorium_lib** is a high-performance scientific C++ library designed for demanding computational domains such as **numerical relativity**, **machine learning (ML)**, **deep learning (DL)** and general **scientific simulations**.

Here is the full documentation : https://at0m741.github.io/Tensorium_lib/

It provides a modern, extensible infrastructure for efficient vector, matrix, and tensor computations by leveraging:
- **SIMD acceleration** (SSE, AVX2, AVX512),
- **Multithreading** with OpenMP,
- And soon, **distributed computing** via MPI.

The core philosophy of Tensorium_lib is to combine:
- **Raw performance**, through low-level SIMD optimization,
- **Modularity and clarity**, using a modern, header-only C++17 design,
- **Python interoperability**, via PyBind11, for seamless integration with scientific Python workflows.

This library is built with the goal of empowering projects that require both speed and flexibility, such as:
- Simulating curved spacetime and relativistic matter (e.g. BSSN formalism, GRHD, GRMHD),
- Custom neural network training and inference on CPU (not really atm),
- Fast manipulation of large scientific datasets and image matrices (not atm),
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
- Python ≥ 3.10 (for Python bindings)
- `pybind11` installed (`pacman -S python-pybind11` on Arch, or `pip install pybind11 --user`)
- OpenBLAS (optional, for benchmarking with BLAS)

## Nix shell command for LLVM plugin dependencies

```bash
nix-shell -p numactl zlib udev openmpi clang llvmPackages.openmp tree openblas  python312Packages.pybind11 llvmPackages_17.clang llvmPackages_17.llvm llvmPackages_17.libclang cloc doxygen graphviz bear llvmPackages_17.mlir
```
if you are on Macos :
```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

### Build C++ Library and Python Module

```bash
mkdir pybuild
cd pybuild
cmake ..
make -j4
```
Then you can use it as the .ipynb show

### Build C++ only for special targets and options

```bash
make                # Default AVX2
make help	    # Show differents compile options 
make AVX512=true    # AVX512
make USE_KNL=true   # MCDRAM Memkind HBW (Xeon phi KNL)
make DEBUG=true     # debug symbols
make VERBOSE=true   # VERBOSE log
make benchmark      # BLAS vs Tensorium mat_mult benchmark
```

The Python module will be created as a .so file in the pybuild/ directory.
### Exemple using in C++
```cpp
#include "Tensorium.hpp"

int main() {
	Vector<float> v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	Vector<float> v2 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

	std::cout << "\n[v1] + [v2]:\n";
	tensorium::add_vec(v1, v2).print();

	std::cout << "\n[v1] - [v2]:\n";
	tensorium::sub_vec(v1, v2).print();

	std::cout << "\n[v1] * 0.5:\n";
	tensorium::scl_vec(v1, 0.5f).print();

	Matrix<float> m1(2, 8); 
	Matrix<float> m2(2, 8);

	for (size_t i = 0; i < m1.rows; ++i)
		for (size_t j = 0; j < m1.cols; ++j) {
			m1(i, j) = i * 10 + j;
			m2(i, j) = 1.0f;
		}

	std::cout << "\n[m1] + [m2]:\n";
	tensorium::add_mat(m1, m2).print();

	std::cout << "\n[m1] - [m2]:\n";
	tensorium::sub_mat(m1, m2).print();

	std::cout << "\n[m1] * 2.0:\n";
	tensorium::scl_mat(m1, 2.0f).print();
}
```
### Example using in Python
```python
from tensorium import *

matA = Matrix(2, 3)
matA.fill([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

matB = Matrix(2, 3)
matB.fill([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

print("matA + matB =")
tns.add_mat(matA, matB).print()

print("matA - matB =")
tns.sub_mat(matA, matB).print()

print("matA * 2.0 =")
tns.scl_mat(matA, 2.0).print()

v = Vector([1.0, 2.0, 3.0])
v2 = Vector([4.0, 5.0, 6.0])

print("v =", v)
print("len(v) =", len(v))
print("v + v2 =", tns.add_vec(v, v2))
print("v - v2 =", tns.sub_vec(v, v2))
print("v * 2.0 =", tns.scl_vec(v, 2.0))
print("dot(v, v2) =", tns.dot_vec(v, v2))
print("norm_1(v) =", tns.norm_1(v))
print("norm_2(v) =", tns.norm_2(v))
print("norm_inf(v) =", tns.norm_inf(v))
print("cosine(v, v2) =", tns.cosine(v, v2))
print("lerp(v, v2, 0.5) =", tns.lerp(v, v2, 0.5))
```




