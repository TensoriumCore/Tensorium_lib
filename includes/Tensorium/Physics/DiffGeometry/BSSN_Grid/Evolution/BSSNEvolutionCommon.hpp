#pragma once

#include <algorithm>

namespace tensorium_RG::bssn {

inline size_t clamped_lower(size_t lower, size_t guard, size_t upper) {
    return std::min(lower + guard, upper);
}

inline size_t clamped_upper(size_t upper, size_t guard, size_t lower) {
    return (upper > guard) ? upper - guard : lower;
}

} // namespace tensorium_RG::bssn

