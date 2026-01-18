![Nouveau projet](https://github.com/user-attachments/assets/5f75f1f9-999d-410b-971e-ba3bd5e8b5e9)
# Tensorium_lib
> !!!DISCLAMER!!! 
Tensorium_lib is still in the early development phase, and many of its features work, but I'm not yet convinced of the solidity of some of them (especially the tensor manipulations).
The python binding is usable without any other python librairy, but I'm still working on it to make it all clean and usable using a simple pip3 install (see the Jupiter Notebook).

**Tensorium_lib** is a high-performance scientific C++ library designed for demanding computational domains such as **numerical relativity**, **machine learning (ML)**, **deep learning (DL)** and general **scientific simulations**.

## Documentation 

> Here is the full documentation : https://tensoriumcore.github.io/Tensorium_lib/

### BSSN Grid
The `includes/Tensorium/Physics/DiffGeometry/BSSN_Grid` module contains the structured-array storage, evolution kernels, and constraint monitoring for the vacuum BSSN formulation equipped with the 1+log + Gamma-driver gauge.  A detailed description of the state variables, evolution equations, gauge system, projections, and diagnostics is provided in [`includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md`](includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md).

## Highlight
It provides a modern, extensible infrastructure for efficient vector, matrix, and tensor computations by leveraging:
- **SIMD acceleration** (SSE, AVX2, AVX512, Neon and soon Apple AMX),
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

##  Requirements

>  **Recommended:** build and use with **LLVM/Clang** for maximum performance.

###  Core Dependencies
- **C++17/20 compiler** with `AVX2` / `FMA` or `ARM Neon` support  
  → `AVX512` is automatically detected and enabled if available  
  → Recommended: **Clang ≥ 17** or **LLVM ≥ 20**  
- **OpenMP** (`fopenmp`)
- **MPI** (for distributed parallelism)
- **libmemkind-dev** *(required only for Intel Xeon Phi Knight Landing CPUs)*
- **CMake ≥ 3.16**
- **Python ≥ 3.10** (for Python bindings)
- **pybind11**  
  - Arch Linux: `sudo pacman -S python-pybind11`  
  - Other: `pip install pybind11 --user`
- **OpenBLAS** *(optional)* — used for benchmarking against BLAS kernels

---
## Build Instructions

### Recommended LLVM/Clang Toolchain

Clang/LLVM 20+ delivers the best SIMD + OpenMP performance on x86_64 and AArch64.

#### Install LLVM/Clang (Linux example)

```bash
# Clone the official LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir llvm-build-release && cd llvm-build-release

# Configure the build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld;lldb;openmp" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-20

# Build & install
ninja -j$(nproc)
sudo ninja install
```
Then you can compile the Tensorium_lib. If you want to use it on your own projects, simply change the Test rule to Srcs (or another) and set the recommended options in the CmakeLists.txt file in the `
Tests` folder, or add a src rule and create a src folder :
then
```cmake
###inside the main CmakeLists.txt
if(BUILD_SRCS)
  add_subdirectory(SRCS)
endif()
```
### Configure & Build

```bash
git clone https://github.com/TensoriumCore/Tensorium_lib.git
cd Tensorium_lib
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS=ON \
  -DTENSORIUM_BSSN_PROFILE_KERNELS=ON
cmake --build build -j
```

Key CMake switches:

| Option | Default | Description |
| --- | --- | --- |
| `TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS` | OFF | Extra CCZ4/BSSN consistency checks (slower, useful during development). |
| `TENSORIUM_BSSN_PROFILE_KERNELS` | OFF | Records per-kernel timing information for the CCZ4 RHS. |
| `BUILD_TESTING` | ON | Controls the test suites under `Tests/`. |
| `BUILD_PYBIND` | ON | Builds the Python module in `pybuild/`. |

The Python extension ends up in `pybuild/` and can be imported by setting `PYTHONPATH=pybuild` or by running `pip install -e .`.

### Running the Test Suites

After building, run the aggregated tests:

```bash
cd Tensorium_lib
ctest --test-dir build --output-on-failure
```

Common filters:

- `ctest -R bssn` – run all BSSN/CCZ4 regression tests.
- `ctest -R ccz4_schwarzschild_stability` – long-running Schwarzschild CCZ4 check.
- `./build/Tests/TensoriumTests --list` – list individual tests.

### Generating Documentation

The Doxygen configuration lives at the project root:

```bash
doxygen Doxyfile
```

HTML docs are emitted to `docs/html/index.html`. The BSSN/CCZ4 subsystem also ships a dedicated README (`includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md`) describing the equations and state layout.

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
- Tensor operators
- Multiple kernels for Tensors/Matrix (optimized for several sizes)
- General relativity / differential geometry classes dans methods (BSSN)
- CUDA runtime kernels for critical kernels and operators
- Spectral Methdods (Chebychev/Fourrier)
- Backward FDM
- Some (several) optimizations
- Plug Tensorium_MLIR and externalize Compiler plugins (subdependencies)
- ARM support 


### Exemple using in C++
```cpp
#include "Tensorium.hpp"

int main() {

	#pragma tensorium dispatch

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


