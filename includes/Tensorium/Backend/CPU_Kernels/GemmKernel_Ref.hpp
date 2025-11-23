// #pragma once

#include "../../Core/Matrix.hpp"

namespace tensorium {
template <typename T> class GemmKernelBig {
  public:
    using Simd = simd::SimdTraits<T, DefaultISA>;
    using reg = typename Simd::reg;
    static constexpr int SimdWidth = Simd::width;
};
} // namespace tensorium
