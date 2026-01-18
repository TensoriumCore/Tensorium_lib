#pragma once

#include <algorithm>

#include "../Fields/BSSNGridSoA.hpp"

/**
 * @file BSSNEvolutionCommon.hpp
 * @brief Shared utilities (interior clamping helpers) for all RHS kernels.
 * @details Kernels call `clamped_lower/upper` to skip guard cells that halo/B.C. handlers may still
 * be updating.  Default padding of 4 ensures the ±2 (and ±3 KO6) stencils remain interior.
 */

namespace tensorium_RG::bssn {

#ifndef TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
#define TENSORIUM_BSSN_EVOLUTION_CLAMP_HELPERS_DEFINED
inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}
#endif

struct InteriorRegion {
    size_t i0 = 0, i1 = 0;
    size_t j0 = 0, j1 = 0;
    size_t k0 = 0, k1 = 0;

    [[nodiscard]] bool empty() const noexcept { return i0 >= i1 || j0 >= j1 || k0 >= k1; }
};

template <typename T>
inline InteriorRegion interior_bounds(const BSSNGridSoA<T> &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    InteriorRegion region;
    region.i0 = clamped_lower(I0, padding, I1);
    region.j0 = clamped_lower(J0, padding, J1);
    region.k0 = clamped_lower(K0, padding, K1);
    region.i1 = clamped_upper(I1, padding, I0);
    region.j1 = clamped_upper(J1, padding, J0);
    region.k1 = clamped_upper(K1, padding, K0);
    return region;
}

template <typename T, typename Fn>
inline void for_each_interior_index(const BSSNGridSoA<T> &grid, size_t padding, Fn &&fn) {
    const auto region = interior_bounds(grid, padding);
    if (region.empty())
        return;
    for (size_t i = region.i0; i < region.i1; ++i)
        for (size_t j = region.j0; j < region.j1; ++j)
            for (size_t k = region.k0; k < region.k1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                fn(i, j, k, idx);
            }
}

} // namespace tensorium_RG::bssn
