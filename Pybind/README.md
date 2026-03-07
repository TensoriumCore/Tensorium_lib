# Tensorium Python Binding

This folder contains the `pybind11` module exposed as `tensorium.tensorium`.

## Requirements

- `python3`
- `python3 -m pip`
- CMake >= 3.22
- Python packages: `setuptools`, `wheel`, `cmake`, `numpy`, `pybind11`

Install Python requirements:

```bash
python3 -m pip install --upgrade pip setuptools wheel cmake numpy pybind11
```

## Option 1: Build + install package (recommended)

From repo root:

```bash
python3 -m pip install -e .
```

This builds the extension and installs the `tensorium` package in editable mode.

## Option 2: Build extension only

From repo root:

```bash
python3 setup.py build_ext --inplace
```

This produces the extension in the package path (`Pysrc/tensorium/`).

## Option 3: Build via CMake directly

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYBIND=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PLUGINS=OFF
cmake --build build -j --target tensorium
```

## Verify

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, "Pysrc")
import tensorium
v = tensorium.Vectord([1.0, 2.0, 3.0])
print(v)
PY
```

## Current Binding Scope

- Core containers:
  - `Vector` (`float`), `Vectord` (`double`)
  - `Matrix` (`float`), `Matrixd` (`double`)
  - `tns.Tensor2d`, `tns.Tensor4d` (`double`)
- Algebra helpers through `tensorium.tns`:
  - vector ops, norms, interpolation, cross product
  - matrix ops, inversion, determinant/rank, linear solvers
  - tensor product and rank-4 contractions
- Differential geometry helpers:
  - `tns.Metric`
  - `tns.compute_christoffel`
  - `tns.compute_riemann_tensor`
  - Ricci contraction/scalar helpers
- NumPy conversion helpers are available on bound classes via
  - `Class.from_numpy(...)`
  - `instance.to_numpy()`

## Not Yet Exposed

- `BSSN_Grid` runtime objects and steppers
- MPI-distributed paths and CUDA backends
- High-level Python orchestration APIs for full NR runs
