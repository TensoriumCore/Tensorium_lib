![Nouveau projet](https://github.com/user-attachments/assets/5f75f1f9-999d-410b-971e-ba3bd5e8b5e9)
# Tensorium_lib

Tensorium_lib is a C++17 scientific computing library focused on high-performance CPU kernels and numerical relativity workflows.

## Current scope

- Core dense linear algebra and tensor containers (`Vector`, `Matrix`, `Tensor`) with aligned storage and SIMD paths.
- Numerical relativity stack centered on `BSSN_Grid` (single-grid, vacuum BSSN/Z4c style evolution).
- Test and visualization tooling for long moving-puncture runs.
- Optional R&D compiler plugins (LLVM/Clang/MLIR).

## Domain coherence (what is production vs experimental)

| Area | Status | Notes |
| --- | --- | --- |
| `Core/Vector.hpp` | usable | SIMD + aligned storage, covered by `Tests/core/VectorTests.cpp`. |
| `Core/Matrix.hpp` | usable | Dense ops + GEMM path + inverse/solve utilities, covered by `Tests/core/MatrixTests.cpp`. |
| `Core/Tensor.hpp` | usable with caution | Generic rank support, contraction/product utilities; still evolving. |
| `Physics/DiffGeometry/BSSN_Grid` | active core | Main NR implementation used by current tests and runs. |
| `Physics/DiffGeometry/BSSN` | legacy/experimental | Historical pointwise layer, not the main runtime path. |
| `Backend/MPI` | foundational | Domain decomposition and halo exchange utilities exist, not yet integrated end-to-end in `BSSN_Grid` runtime. |
| `Backend/CUDA` | scaffold | CUDA structures exist; NR kernels are currently CPU/OpenMP. |
| `Plugins/` (LLVM/MLIR) | R&D | Optional compiler tooling, not required for physics runs. |

## Repository map

```text
includes/Tensorium/Core/                          # Vector/Matrix/Tensor + solvers + derivatives
includes/Tensorium/Backend/SIMD/                 # ISA traits (SSE/AVX/AVX2/AVX512/NEON)
includes/Tensorium/Backend/OpenMP/               # OpenMP compatibility helpers
includes/Tensorium/Backend/MPI/                  # Cartesian decomposition + halo exchange building blocks
includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/  # Main numerical relativity implementation
Tests/core/                                      # Core numerical tests
Tests/bssn/                                      # NR validation/stability tests and CSV exporters
Plugins/                                         # Optional LLVM/Clang/MLIR experiments
```

## Build

### Requirements

- CMake >= 3.22
- C++17 compiler (Clang recommended)
- OpenMP runtime
- Optional: MPI, CUDA toolkit, OpenBLAS/BLAS, pybind11

### Configure and compile

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_PYBIND=OFF \
  -DBUILD_PLUGINS=OFF
cmake --build build -j
```

### Useful options

| Option | Default | Description |
| --- | --- | --- |
| `BUILD_TESTS` | `ON` | Build test executables in `Tests/`. |
| `BUILD_PYBIND` | `OFF` | Build Python bindings. |
| `BUILD_PLUGINS` | `ON` | Build LLVM/Clang plugins. Disable if you only need runtime/library code. |
| `USE_MPI` | `OFF` | Link MPI and enable MPI compile definitions. |
| `USE_CUDA` | `OFF` | Enable CUDA language/backend pieces. |
| `AVX2` / `AVX512` | `OFF` | Force stronger x86 codegen flags when desired. |
| `TENSORIUM_BSSN_PROFILE_KERNELS` | `OFF` | BSSN kernel timers. |
| `TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS` | `OFF` | Extra BSSN consistency validation. |

## Running tests

```bash
ctest --test-dir build --output-on-failure
```

Common filters:

```bash
ctest --test-dir build -R core
ctest --test-dir build -R bssn
./build/Tests/TensoriumTests --list
```

## Numerical relativity notes

- The active physics path is `includes/Tensorium/Physics/DiffGeometry/BSSN_Grid`.
- The moving puncture visualization/stability driver is in `Tests/bssn/bowen_york_boost_stability.cpp`.
- Interpolated Two-Punctures initialization requires the external TwoPunctures codebase from:
  `https://github.com/GRTLCollaboration/TwoPunctures.git`
- In this repository, that external solver is expected through the local `../TwoPuncturesC` integration (with GSL) at build time.

## LLVM/MLIR plugins

`Plugins/` contains optional compiler-side tooling (diagnostic/alignment pass, Clang pragma plugin, MLIR conversion experiments).

These plugins are not required for:

- `Vector/Matrix/Tensor` runtime usage
- BSSN/Z4c simulations
- Standard test execution

If your goal is NR runs, keep `BUILD_PLUGINS=OFF`.

## Documentation

- Doxygen config: `Doxyfile`
- Generated docs: `docs/html/index.html`
- BSSN_Grid technical guide:
  [includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md](includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md)
