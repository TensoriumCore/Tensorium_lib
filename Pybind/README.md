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
- BSSN grid initial-data bindings:
  - `bssn.BSSNGrid`
  - `bssn.minkowski`
  - `bssn.schwarzschild_isotropic`
  - `bssn.kerr_schild_single`
  - `bssn.binary_bowen_york_puncture_init`
  - `bssn.binary_bowen_york_puncture_interpolated_init`
  - `bssn.binary_bowen_york_puncture_twopunctures_c_init`
- NumPy conversion helpers are available on bound classes via
  - `Class.from_numpy(...)`
  - `instance.to_numpy()`

## BSSN quick start

```bash
python3 - <<'PY'
import tensorium
g = tensorium.bssn.BSSNGrid(24, 24, 24, 4, 0.25, 0.25, 0.25)
tensorium.bssn.minkowski(g)
alpha = g.alpha()
print(alpha.shape, alpha.min(), alpha.max())
PY
```

### TwoPuncturesC backend

The Python extension enables TwoPuncturesC automatically when:

- `../TwoPuncturesC/include/TwoPunctures.h` exists (relative to repo root)
- `gsl-config` is available in `PATH`

Runtime check:

```bash
python3 - <<'PY'
import tensorium
print("two_punctures_c:", tensorium.bssn.has_twopunctures_c())
PY
```

Explicit TwoPuncturesC init:

```bash
python3 - <<'PY'
import tensorium
g = tensorium.bssn.BSSNGrid(64, 64, 64, 4, 0.125, 0.125, 0.125)
tensorium.bssn.binary_bowen_york_puncture_twopunctures_c_init(
    g,
    0.5, -1.0, 0.0, 0.0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0],
    0.5,  1.0, 0.0, 0.0, [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0],
    npoints_A=30, npoints_B=30, npoints_phi=16, tp_threads=8
)
PY
```

### Grid field access

- Scalar fields:
  - `g.alpha(include_halo=False)`
  - `g.chi(include_halo=False)`
  - `g.K(include_halo=False)`
  - `g.Theta(include_halo=False)`
- Vector fields (`component` in `[0,1,2]`):
  - `g.beta(component, include_halo=False)`
  - `g.B(component, include_halo=False)`
  - `g.tildeGamma(component, include_halo=False)`
  - `g.Z(component, include_halo=False)`
- Symmetric tensor fields (`component` in `[0..5]`, constants `XX,XY,XZ,YY,YZ,ZZ`):
  - `g.gamma_tilde(component, include_halo=False)`
  - `g.gamma_tilde_inv(component, include_halo=False)`
  - `g.A_tilde(component, include_halo=False)`
  - `g.Ricci(component, include_halo=False)`
- Basic scalar setters:
  - `g.set_alpha(array3d, include_halo=False)`
  - `g.set_chi(array3d, include_halo=False)`
  - `g.set_K(array3d, include_halo=False)`

## Not Yet Exposed

- `BSSN_Grid` steppers and gauge runtime controls
- MPI-distributed paths and CUDA backends
- High-level Python orchestration APIs for full NR runs
