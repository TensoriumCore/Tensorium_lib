
# TODO.md - Morpheus SIMD Linear Algebra Library

## Implemented

### Vector<float>
- `add(const Vector&)`
- `sub(const Vector&)`
- `scl(float)`
- `dot(const Vector&) const`
- `norm_1() const`
- `norm_2() const`
- `norm_inf() const`
- `lerp(const Vector&, const Vector&, float)`
- `angle_cos(const Vector&, const Vector&)`
- `cross_product(const Vector&, const Vector&)`
- `linear_combination(const std::vector<Vector>&, const std::vector<float>&)`
- `print()`, `size()`, `operator[]`, `constructor from std::vector`

### Matrix<float>
- `add(const Matrix&)`
- `sub(const Matrix&)`
- `scl(float)`
- `mul_mat(const Matrix&)`
- `print()`, `operator()`, `constructor with size`

### SIMD Backend
- `SimdTraits<float>` (AVX2)
- `SimdTraits<double>` (AVX2)
- `reduce_sum(__m256)` and `reduce_sum(__m256d)`
- `aligned_vector<T>` with `AlignedAllocator<T, 32>`
- `dispatch_simd()` with AVX512/AVX2/SSE support

---

## High Priority

### Vector<T> extensions
- [ ] `normalize()` (unit vector)
- [ ] `max() const`, `min() const`
- [ ] `sum() const`, `mean() const`
- [ ] `argmax() const`, `argmin() const`
- [ ] `clamp(float min, float max)`
- [ ] `apply(std::function<T(T)>)` or a templated transform

### Matrix<T> extensions
- [ ] `transpose() const` (return a new transposed matrix)
- [ ] `matvec(const Vector&) const` (matrix-vector product)
- [ ] `eye(size_t n)` (identity matrix)
- [ ] `diag(const Vector&)` (diagonal matrix)
- [ ] `zero(size_t r, size_t c)`, `ones(...)`

### SIMD Backends
- [ ] SSE fallback in `SimdTraits<float>` and `SimdTraits<double>`
- [ ] AVX512 variant of `SimdTraits`
- [ ] Dynamic dispatch and runtime detection wrapper for `SimdTraits`
- [ ] Support for non-float types (e.g., `int32_t`, `uint32_t`, etc.)

---

## Python API (pybind11)

- [x] Bind `Vector<float>` (constructor, len, getitem, repr, print)
- [x] Bind `add`, `sub`, `scl` globally
- [ ] Bind all other vector methods: `dot`, `norm_*`, `angle_cos`, etc.
- [ ] Bind `Matrix<float>` and main methods
- [ ] Enable conversion to/from `numpy.ndarray`
- [ ] Package as editable module with `pyproject.toml` and `setup.py`

---

## Testing and Benchmarks

- [x] Python test for basic vector ops
- [ ] Benchmark Vector vs NumPy (add, scl, dot)
- [ ] C++ benchmarks: measure AVX2 speedup vs scalar
- [ ] Benchmark `mul_mat` for large sizes (512x512, 1024x1024)
- [ ] Add unit tests for all SIMD paths

---

## Build System

- [x] `CMakeLists.txt` for native + bindings
- [ ] `setup.py` + `pyproject.toml` for Python install
- [ ] CMake option to build Python bindings only
- [ ] Add install target for headers and shared libs

---

## Extras / Long Term

- [ ] Add support for `Vector<double>`, `Matrix<double>`
- [ ] Add CPU cache blocking optimization to `mul_mat`
- [ ] Add OpenMP offload / multithreaded Vector ops
- [ ] Expose aligned allocator as standalone utility
- [ ] Documentation website (Doxygen + Sphinx + GitHub Pages)

