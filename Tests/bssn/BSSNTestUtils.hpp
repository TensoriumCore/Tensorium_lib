#pragma once

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNInvariants.hpp"
#include "../../includes/Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tensorium::tests {

struct BSSNInvariantStats : public tensorium_RG::bssn::InvariantStats {
    double max_ricci = 0.0;
    double max_H = 0.0;
    double max_M = 0.0;
    double max_C = 0.0;
};

inline double max_tensor_component(const tensorium_RG::Field3D<double> *fields, size_t idx) {
    double vals[6];
    vals[0] = std::abs(fields[tensorium_RG::XX].ptr()[idx]);
    vals[1] = std::abs(fields[tensorium_RG::XY].ptr()[idx]);
    vals[2] = std::abs(fields[tensorium_RG::XZ].ptr()[idx]);
    vals[3] = std::abs(fields[tensorium_RG::YY].ptr()[idx]);
    vals[4] = std::abs(fields[tensorium_RG::YZ].ptr()[idx]);
    vals[5] = std::abs(fields[tensorium_RG::ZZ].ptr()[idx]);
    return *std::max_element(std::begin(vals), std::end(vals));
}

inline double max_vector_component(const tensorium_RG::Field3D<double> *fields, size_t idx) {
    double vals[3];
    vals[0] = std::abs(fields[0].ptr()[idx]);
    vals[1] = std::abs(fields[1].ptr()[idx]);
    vals[2] = std::abs(fields[2].ptr()[idx]);
    return std::max({vals[0], vals[1], vals[2]});
}

inline BSSNInvariantStats compute_invariants(const tensorium_RG::BSSNGridSoA<double> &grid,
                                             size_t padding, double r_min = 0.0,
                                             double r_max = std::numeric_limits<double>::max(),
                                             double xc = 0.0, double yc = 0.0, double zc = 0.0) {
    BSSNInvariantStats stats;

    const auto inv_stats = tensorium_RG::bssn::compute_invariant_stats(grid, padding, r_min, r_max,
                                                                      xc, yc, zc);
    static_cast<tensorium_RG::bssn::InvariantStats &>(stats) = inv_stats;

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return stats;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const double X = x - xc;
                const double Y = y - yc;
                const double Z = z - zc;
                const double r = std::sqrt(X * X + Y * Y + Z * Z);
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = grid.alpha.idx(i, j, k);
                stats.max_ricci = std::max(stats.max_ricci, max_tensor_component(grid.Ricci, id));
                stats.max_H = std::max(stats.max_H, std::abs(grid.Hc.ptr()[id]));
                stats.max_M = std::max(stats.max_M, max_vector_component(grid.Mc, id));
                stats.max_C = std::max(stats.max_C, max_vector_component(grid.Cc, id));
            }

    return stats;
}

} // namespace tensorium::tests
