#pragma once

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium_Grid/Grid/GridLayout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tensorium::tests {

struct BSSNInvariantStats {
    double max_det_deviation = 0.0;
    double max_trace_A = 0.0;
    double max_ricci = 0.0;
    double max_H = 0.0;
    double max_M = 0.0;
    double max_C = 0.0;
    size_t samples = 0;
};

inline void load_sym_matrix(const tensorium_RG::Field3D<double> *fields, size_t idx,
                             double out[3][3]) {
    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
            const double val = tensorium_RG::sym6_get(fields, idx, a, b);
            out[a][b] = val;
            out[b][a] = val;
        }
}

inline double det3(const double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[1][2]) -
           m[0][1] * (m[0][1] * m[2][2] - m[0][2] * m[1][2]) +
           m[0][2] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
}

inline double trace_weighted(const double ginv[3][3], const double tensor[3][3]) {
    double trace = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            trace += ginv[a][b] * tensor[a][b];
    return trace;
}

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

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t i0 = std::min(I0 + padding, I1);
    const size_t j0 = std::min(J0 + padding, J1);
    const size_t k0 = std::min(K0 + padding, K1);
    const size_t i1 = (I1 > padding) ? I1 - padding : I1;
    const size_t j1 = (J1 > padding) ? J1 - padding : J1;
    const size_t k1 = (K1 > padding) ? K1 - padding : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return stats;

    double g[3][3], ginv[3][3], A[3][3];

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
                load_sym_matrix(grid.gamma_tilde, id, g);
                load_sym_matrix(grid.gamma_tilde_inv, id, ginv);
                load_sym_matrix(grid.A_tilde, id, A);

                stats.max_det_deviation =
                    std::max(stats.max_det_deviation, std::abs(det3(g) - 1.0));
                stats.max_trace_A =
                    std::max(stats.max_trace_A, std::abs(trace_weighted(ginv, A)));
                stats.max_ricci = std::max(stats.max_ricci, max_tensor_component(grid.Ricci, id));
                stats.max_H = std::max(stats.max_H, std::abs(grid.Hc.ptr()[id]));
                stats.max_M = std::max(stats.max_M, max_vector_component(grid.Mc, id));
                stats.max_C = std::max(stats.max_C, max_vector_component(grid.Cc, id));
                ++stats.samples;
            }

    return stats;
}

} // namespace tensorium::tests
