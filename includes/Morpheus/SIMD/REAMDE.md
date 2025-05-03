# Morpheus — SIMD Module

This directory contains the low-level SIMD backend used throughout Morpheus for accelerating vector, matrix, and tensor operations. It provides abstraction layers over AVX, SSE, and AVX512 instructions, as well as runtime detection and alignment-aware memory allocation.

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
- Auto-selection of best ISA at runtime or compile-time

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

SimdTraits<float, avx512_t>::vec a = set1<float, avx512_t>(3.0f);
SimdTraits<float, avx512_t>::vec b = set1<float, avx512_t>(2.0f);
auto c = add<float, avx512_t>(a, b); // SIMD add: c = a + b
```

## Status

Fully functional and integrated into all math kernels of Morpheus, including `Vector`, `Matrix`, `Tensor`, and numerical methods. Portability fallback to SSE is provided for older CPUs. AVX512 path is fully optimized.

## Supported Architectures

- SSE4.2 (128-bit)
- AVX / AVX2 (256-bit)
- AVX512F / AVX512DQ (512-bit)
- Optional hbw (High Bandwidth Memory) detection on Xeon Phi (KNL)

## References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- A. Fog, *Optimizing Software in C++*
- Intel 64 and IA-32 Architectures Software Developer Manuals
