#pragma once

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNGamma.hpp"
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

    auto                      H_tmp = tensorium_RG::make_field(grid.alpha.st);
    tensorium_RG::Field3D<double> M_tmp[3];
    tensorium_RG::Field3D<double> C_tmp[3];
    for (int q = 0; q < 3; ++q) {
        M_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
        C_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
    }
    auto &mutable_grid = const_cast<tensorium_RG::BSSNGridSoA<double> &>(grid);
    tensorium_RG::bssn::compute_bssn_constraints(mutable_grid, grid.Ricci, H_tmp, M_tmp, C_tmp,
                                                 r_min, r_max, xc, yc, zc, false);

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
                stats.max_H = std::max(stats.max_H, std::abs(H_tmp.ptr()[id]));
                stats.max_M = std::max(stats.max_M, max_vector_component(M_tmp, id));
                stats.max_C = std::max(stats.max_C, max_vector_component(C_tmp, id));
            }

    return stats;
}

inline double max_gamma_violation_z4c(const tensorium_RG::BSSNGridSoA<double> &grid,
                                      size_t padding, double r_min = 0.0,
                                      double r_max = std::numeric_limits<double>::max(),
                                      double xc = 0.0, double yc = 0.0, double zc = 0.0) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    double max_violation = 0.0;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                x -= xc;
                y -= yc;
                z -= zc;
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = grid.alpha.idx(i, j, k);

                double div[3];
                tensorium_RG::bssn::metric_inverse_divergence(grid, i, j, k, div);

                const double gixx = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 0, 0);
                const double gixy = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 0, 1);
                const double gixz = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 0, 2);
                const double giyy = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 1, 1);
                const double giyz = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 1, 2);
                const double gizz = tensorium_RG::sym6_get(grid.gamma_tilde_inv, id, 2, 2);

                const double Zx = static_cast<double>(grid.Z[0].ptr()[id]);
                const double Zy = static_cast<double>(grid.Z[1].ptr()[id]);
                const double Zz = static_cast<double>(grid.Z[2].ptr()[id]);

                const double raise0 = gixx * Zx + gixy * Zy + gixz * Zz;
                const double raise1 = gixy * Zx + giyy * Zy + giyz * Zz;
                const double raise2 = gixz * Zx + giyz * Zy + gizz * Zz;

                const double target0 = -div[0] + 2.0 * raise0;
                const double target1 = -div[1] + 2.0 * raise1;
                const double target2 = -div[2] + 2.0 * raise2;

                const double stored0 = static_cast<double>(grid.tildeGamma[0].ptr()[id]);
                const double stored1 = static_cast<double>(grid.tildeGamma[1].ptr()[id]);
                const double stored2 = static_cast<double>(grid.tildeGamma[2].ptr()[id]);

                max_violation = std::max(max_violation, std::abs(stored0 - target0));
                max_violation = std::max(max_violation, std::abs(stored1 - target1));
                max_violation = std::max(max_violation, std::abs(stored2 - target2));
            }

    return max_violation;
}

} // namespace tensorium::tests
