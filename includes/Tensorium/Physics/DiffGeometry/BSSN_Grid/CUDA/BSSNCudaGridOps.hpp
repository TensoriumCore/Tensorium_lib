#pragma once

#include "../Fields/BSSNGridViews.hpp"

#include <cstddef>
#include <stdexcept>

namespace tensorium_RG::bssn::cuda {

#ifdef TENSORIUM_CUDA

template <typename T>
void copy_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, std::size_t padding);

template <typename T>
void accumulate_stage_reference(BSSNGridView<const T> src, BSSNGridView<T> dst, T delta,
                                std::size_t padding);

template <typename T> void apply_alpha_floor(BSSNGridView<T> grid, T floor);

template <typename T> void apply_chi_floor(BSSNGridView<T> grid, T floor);

#else

template <typename T>
inline void copy_stage_reference(BSSNGridView<const T>, BSSNGridView<T>, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T>
inline void accumulate_stage_reference(BSSNGridView<const T>, BSSNGridView<T>, T, std::size_t) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T> inline void apply_alpha_floor(BSSNGridView<T>, T) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

template <typename T> inline void apply_chi_floor(BSSNGridView<T>, T) {
    throw std::runtime_error("Tensorium was built without CUDA support");
}

#endif

} // namespace tensorium_RG::bssn::cuda
