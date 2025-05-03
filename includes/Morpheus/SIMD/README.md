# Morpheus — SIMD Module

This directory contains the low-level SIMD backend used throughout Morpheus to accelerate vector, matrix, and tensor operations. It provides abstraction layers over AVX, AVX2, AVX512, and SSE instruction sets, with automatic runtime detection and alignment-aware memory allocation. Writing separate code for each ISA is unnecessary — all operations are unified through `SimdTraits` with graceful fallback to the best supported instruction set.

## Purpose

The module enables:
- High-performance vectorized operations on aligned data
- Support for AVX, SSE, and AVX512 with fallback logic
- Architecture-specific runtime dispatching
- Cache hierarchy introspection and memory-aware optimization

## Structure

```
SIMD/
├── SIMD.hpp         // Abstractions for AVX, AVX512, and SSE intrinsics
├── Allocator.hpp    // Aligned memory allocation (posix_memalign / hbwmalloc)
├── CPU_id.hpp       // CPU feature detection (AVX, AVX2, AVX512F, SSE4.2, etc.)
└── CacheInfo.hpp    // L1/L2/L3 cache size detection and prefetch tuning
```

## Features

### SIMD.hpp
- `SimdTraits<T, ISA>` specialization for scalar types (`float`, `double`, `size_t`)
- Unified interface for:
  - vector loads/stores (`load`, `store`, `setzero`, `broadcast`)
  - arithmetic (`add`, `sub`, `fmadd`, `mul`)
  - reductions (`horizontal_add`, etc.)
  - Complex numbers are supported
- Auto-selection of best ISA at runtime or compile-time

### aarch64 NEON are incomming !

### Allocator.hpp
- Provides `aligned_allocator<T>` for STL compatibility
- Uses `posix_memalign` or `hbwmalloc` (on Xeon Phi)
- Ensures 32-byte or 64-byte alignment depending on ISA

### CPU_id.hpp
- Detects CPU SIMD capabilities via CPUID
- Flags: SSE4.2, AVX, AVX2, AVX512F, AVX512DQ, etc.
- Used for runtime dispatching in templated kernels

### CacheInfo.hpp
- Detects L1, L2, and L3 cache sizes per core
- Used to tune matrix blocking and tiling strategies
- Helps choose loop unrolling depth based on available cache

## Example Usage

```cpp
using namespace morpheus::simd;
using T    = float;
using Simd = SimdTraits<T, DefaultISA>;
using reg  = typename Simd::reg;

alignas(64) T a[Simd::width] = {1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f};
alignas(64) T b[Simd::width] = {8.0f, 7.0f, 6.0f, 5.0f,
                                4.0f, 3.0f, 2.0f, 1.0f};
alignas(64) T result[Simd::width];

reg va = Simd::load(a);
reg vb = Simd::load(b);
reg vr = Simd::add(va, vb);
Simd::store(result, vr);

for (std::size_t i = 0; i < Simd::width; ++i)
    std::cout << "result[" << i << "] = " << result[i] << "\n";
```

## Status

Fully functional and integrated into all math kernels of Morpheus, including `Vector`, `Matrix`, `Tensor`, and numerical methods. Portability fallback to SSE is provided for older CPUs. AVX512 path is fully optimized.

## Supported Architectures

- SSE4.2 (128-bit)
- AVX / AVX2 (256-bit)
- AVX512F / AVX512DQ (512-bit)
- Optional hbw (High Bandwidth Memory) detection on Xeon Phi (KNL)


