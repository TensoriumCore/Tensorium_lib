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
| `Tensorium/MPI` | active experimental | Current MPI domain decomposition, halo exchange, and BSSN stepping live here. |
| `Backend/CUDA` | scaffold | Runtime/device buffers plus `VectorCUDA`/`MatrixCUDA` exist; NR kernels are still CPU/OpenMP. |
| `Plugins/` (LLVM/MLIR) | R&D | Optional compiler tooling, not required for physics runs. |

## Repository map

```text
includes/Tensorium/Core/                          # Vector/Matrix/Tensor + solvers + derivatives
includes/Tensorium/Backend/SIMD/                 # ISA traits (SSE/AVX/AVX2/AVX512/NEON)
includes/Tensorium/Backend/OpenMP/               # OpenMP compatibility helpers
includes/Tensorium/MPI/                          # Active MPI decomposition, halo exchange, RK stepping
includes/Tensorium/Backend/CUDA/                # CUDA runtime + device buffer scaffolding
includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/  # Main numerical relativity implementation
Tests/core/                                      # Core numerical tests
Tests/z4c/                                      # NR validation/stability tests and CSV exporters
Plugins/                                         # Optional LLVM/Clang/MLIR experiments
```

## Build

### Requirements

- CMake >= 3.22
- C++17 compiler (Clang recommended)
- OpenMP runtime
- Optional: MPI, CUDA toolkit, OpenBLAS/BLAS (or Accelerate on macOS), pybind11, NumPy

### Configure and compile

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_PYBIND=OFF \
  -DBUILD_PLUGINS=OFF
cmake --build build -j
```

If you need the Python extension from CMake:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYBIND=ON \
  -DBUILD_PLUGINS=OFF
cmake --build build -j --target tensorium
```

### Useful options

| Option | Default | Description |
| --- | --- | --- |
| `BUILD_TESTS` | `ON` | Build test executables in `Tests/`. |
| `BUILD_PYBIND` | `OFF` | Build Python bindings. |
| `BUILD_PLUGINS` | `OFF` | Build LLVM/Clang plugins. Disable if you only need runtime/library code. |
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
ctest --test-dir build -R z4c
./build/Tests/TensoriumTests --list
```

## Python bindings

Python bindings expose core vector/matrix/tensor algebra and selected differential-geometry helpers.

### Prerequisites

- `python3`
- `python3 -m pip`
- Python packages: `setuptools`, `wheel`, `cmake`, `numpy`, `pybind11`

Install prerequisites:

```bash
python3 -m pip install --upgrade pip setuptools wheel cmake numpy pybind11
```

### Build and install (editable)

```bash
python3 -m pip install -e .
```

### Alternative: build extension only (no install)

```bash
python3 setup.py build_ext --inplace
```

### Quick check

```bash
python3 - <<'PY'
import tensorium
v = tensorium.Vectord([1.0, 2.0, 3.0])
print(v)
PY
```

NumPy interop example:

```bash
python3 - <<'PY'
import numpy as np
import tensorium
A = tensorium.Matrixd.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
print(tensorium.matrices.det(A))
PY
```

Current scope of Python API:

- `Vector`/`Vectord`, `Matrix`/`Matrixd`, `Tensor2d`, `Tensor4d`
- `tns.*` algebra helpers (`add/sub/mul`, norms, solvers, tensor contractions, etc.)
- metric/curvature helpers (`Metric`, `compute_christoffel`, `compute_riemann_tensor`, Ricci utilities)
- `z4c.Z4cGrid` with Z4c initial-data builders (`minkowski`, `schwarzschild_isotropic`,
  `kerr_schild_single`, Bowen-York puncture initializers)
- explicit `TwoPuncturesC` init path via
  `tensorium.z4c.binary_bowen_york_puncture_twopunctures_c_init(...)`

`z4c` quick start:

```bash
python3 - <<'PY'
import tensorium
g = tensorium.z4c.Z4cGrid(24, 24, 24, 4, 0.25, 0.25, 0.25)
tensorium.z4c.minkowski(g)
alpha = g.alpha()
print(alpha.shape, alpha.min(), alpha.max())
PY
```

Field access examples:

- `g.alpha()` interior values
- `g.alpha(include_halo=True)` including ghost zones
- `g.beta(0)` for `beta^x`
- `g.gamma_tilde(tensorium.z4c.XX)` for `\\tilde{\\gamma}_{xx}`

TwoPuncturesC availability check:

```bash
python3 - <<'PY'
import tensorium
print(tensorium.z4c.has_twopunctures_c())
PY
```

Not yet exposed in Python:

- full `BSSN_Grid` evolution runtime (RK stepper/gauge controls/diagnostics callbacks)
- MPI backend / CUDA device scaffolding

## Numerical relativity notes

- The active physics path is `includes/Tensorium/Physics/DiffGeometry/BSSN_Grid`.
- The moving puncture visualization/stability driver is in `Tests/z4c/bowen_york_boost_stability.cpp`.
- A standalone moving-puncture executable is available at:
  `./build/Tests/TensoriumMovingPuncture`
- It consumes the same environment variables as the test driver
  (`TENSORIUM_MOVING_PUNCTURE_*` + `OMP_*`) without `--test`.
- Interpolated Two-Punctures initialization requires the external TwoPunctures codebase from:
  `https://github.com/GRTLCollaboration/TwoPunctures.git`
- In this repository, that external solver is expected through the local `../TwoPuncturesC` integration (with GSL) at build time.

Standalone run example:

```bash
OMP_NUM_THREADS=24 \
OMP_PROC_BIND=spread \
OMP_PLACES=cores \
OMP_DYNAMIC=FALSE \
TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT=1 \
TENSORIUM_MOVING_PUNCTURE_GRID_N=212 \
TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH=24.0 \
TENSORIUM_MOVING_PUNCTURE_SPATIAL_ORDER=6 \
TENSORIUM_MOVING_PUNCTURE_CFL=0.11 \
TENSORIUM_MOVING_PUNCTURE_STEPS=12000 \
./build/Tests/TensoriumMovingPuncture
```

For nested moving-puncture FMR runs, the most useful controls are:

- `TENSORIUM_MOVING_PUNCTURE_FMR_LEVELS`: number of refined levels nested inside the root grid
- `TENSORIUM_MOVING_PUNCTURE_FMR_REFINEMENT_RATIO`: refinement ratio between successive levels; `2` is the intended production setting
- `TENSORIUM_MOVING_PUNCTURE_FMR_OUTER_BOX_HALF_WIDTH`: physical half-width of the outermost refined box
- `TENSORIUM_MOVING_PUNCTURE_FMR_FINEST_BOX_HALF_WIDTH`: physical half-width of the finest box when sizing from the center outward
- `TENSORIUM_MOVING_PUNCTURE_FMR_FINE_LEVELS_USE_PARENT_INIT=1`: initialize refined levels from parent prolongation instead of a second local TwoPunctures solve
- `TENSORIUM_MOVING_PUNCTURE_FMR_MOVE_WITH_PUNCTURES=1`: re-center the refined hierarchy on the tracked puncture midpoint during evolution
- `TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_INTERVAL`: root-step interval between regrid checks when moving boxes are enabled
- `TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_THRESHOLD_CELLS`: minimum midpoint drift, in finest-grid cells, before rebuilding the hierarchy
- `TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_ON_DRIFT=1`: snap the shift-integrated puncture tracker back to the slice minima if it drifts too far
- `TENSORIUM_MOVING_PUNCTURE_TRACKER_RECENTER_CELLS`: tracker drift threshold, in finest-grid cells, for that recenter

When `TENSORIUM_MOVING_PUNCTURE_FMR_OUTER_BOX_HALF_WIDTH` is set, the refined hierarchy is built geometrically toward the center. For example, `LEVELS=4`, `RATIO=2`, `OUTER_BOX_HALF_WIDTH=64` produces refined half-widths close to `64 -> 32 -> 16 -> 8`.

Example multilevel run:

```bash
OMP_NUM_THREADS=24 \
OMP_DYNAMIC=FALSE \
TENSORIUM_MOVING_PUNCTURE_GRID_N=192 \
TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH=256.0 \
TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT=1 \
TENSORIUM_MOVING_PUNCTURE_ENABLE_FMR=1 \
TENSORIUM_MOVING_PUNCTURE_FMR_LEVELS=4 \
TENSORIUM_MOVING_PUNCTURE_FMR_REFINEMENT_RATIO=2 \
TENSORIUM_MOVING_PUNCTURE_FMR_OUTER_BOX_HALF_WIDTH=64.0 \
TENSORIUM_MOVING_PUNCTURE_FMR_FINE_LEVELS_USE_PARENT_INIT=1 \
./build/Tests/TensoriumMovingPuncture
```

Example moving-box run:

```bash
OMP_NUM_THREADS=24 \
OMP_DYNAMIC=FALSE \
TENSORIUM_MOVING_PUNCTURE_GRID_N=160 \
TENSORIUM_MOVING_PUNCTURE_BOX_LENGTH=64.0 \
TENSORIUM_MOVING_PUNCTURE_USE_INTERPOLATED_INIT=1 \
TENSORIUM_MOVING_PUNCTURE_ENABLE_FMR=1 \
TENSORIUM_MOVING_PUNCTURE_FMR_LEVELS=2 \
TENSORIUM_MOVING_PUNCTURE_FMR_REFINEMENT_RATIO=2 \
TENSORIUM_MOVING_PUNCTURE_FMR_OUTER_BOX_HALF_WIDTH=20.0 \
TENSORIUM_MOVING_PUNCTURE_FMR_FINE_LEVELS_USE_PARENT_INIT=1 \
TENSORIUM_MOVING_PUNCTURE_FMR_MOVE_WITH_PUNCTURES=1 \
TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_INTERVAL=16 \
TENSORIUM_MOVING_PUNCTURE_FMR_REGRID_THRESHOLD_CELLS=2.0 \
./build/Tests/TensoriumMovingPuncture
```

The moving-puncture slice exports now also write companion geometry metadata
(`slice_boxes_*.csv` plus per-cell bounds in `slice_*.csv`) for external
post-processing tools. The stock `plot.py` keeps its default full-slice view.

## LLVM/MLIR plugins

`Plugins/` contains optional compiler-side tooling (diagnostic/alignment pass, Clang pragma plugin, MLIR conversion experiments).

These plugins are not required for:

- `Vector/Matrix/Tensor` runtime usage
- BSSN/Z4c simulations
- Standard test execution

If your goal is NR runs, keep `BUILD_PLUGINS=OFF`.

## Documentation

- Doxygen config: `Doxyfile`
- Generated docs: `docs/index.html`
- Python binding guide: `Pybind/README.md`
- BSSN_Grid technical guide:
  [includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md](includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/README.md)
