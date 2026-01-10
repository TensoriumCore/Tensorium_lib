#pragma once

#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"

namespace tensorium_RG::bssn {

struct EvolutionConfig {
    size_t padding = 4;
};

inline size_t idx_guard_lower(size_t begin, size_t guard, size_t end) {
    return (begin + guard < end) ? begin + guard : end;
}
inline size_t idx_guard_upper(size_t end, size_t guard, size_t begin) {
    return (end > guard) ? end - guard : begin;
}

template <typename T>
inline void compute_rhs_chi(const BSSNGridSoA<T> &G, Field3D<T> &rhs_chi,
                            const EvolutionConfig &cfg = {}) {
    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k)
                rhs_chi.ptr()[G.chi.idx(i, j, k)] = T(0);

    const size_t guard = std::max<size_t>(cfg.padding, size_t(0));
    const size_t i0 = idx_guard_lower(I0, guard, I1);
    const size_t j0 = idx_guard_lower(J0, guard, J1);
    const size_t k0 = idx_guard_lower(K0, guard, K1);
    const size_t i1 = idx_guard_upper(I1, guard, I0);
    const size_t j1 = idx_guard_upper(J1, guard, J0);
    const size_t k1 = idx_guard_upper(K1, guard, K0);

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return;

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i) {
        for (size_t j = j0; j < j1; ++j) {
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = G.chi.idx(i, j, k);
                rhs_chi.ptr()[id] = (T)((2.0 / 3.0) * G.chi.ptr()[id] * G.alpha.ptr()[id] *
                                        G.K.ptr()[id]);
            }
        }
    }
}

} // namespace tensorium_RG::bssn
