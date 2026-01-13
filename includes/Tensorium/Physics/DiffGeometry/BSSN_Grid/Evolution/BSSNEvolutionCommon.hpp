#pragma once

#include <algorithm>

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

} // namespace tensorium_RG::bssn
