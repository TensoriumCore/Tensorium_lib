#pragma once

#include "../Fields/BSSNGridSoA.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace tensorium_RG::bssn {

struct ConstraintMonitorStats {
    double max_H = 0.0;
    double l2_H = 0.0;
    double max_trace_A = 0.0;
    double max_det_drift = 0.0;
    size_t samples = 0;
};

namespace constraint_monitor_detail {
inline double det3(double gxx, double gxy, double gxz, double gyy, double gyz, double gzz) {
    return gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
           gxz * (gxy * gyz - gxz * gyy);
}
}

template <typename T>
inline ConstraintMonitorStats compute_constraint_monitor(const BSSNGridSoA<T> &G,
                                                         const Field3D<T> &H, size_t padding = 4) {
    ConstraintMonitorStats stats;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return stats;

    double accum_H = 0.0;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = H.idx(i, j, k);
                const double h = static_cast<double>(H.ptr()[id]);
                stats.max_H = std::max(stats.max_H, std::abs(h));
                accum_H += h * h;
                stats.samples += 1;

                const double gxx = tensorium_RG::sym6_get(G.gamma_tilde, id, 0, 0);
                const double gxy = tensorium_RG::sym6_get(G.gamma_tilde, id, 0, 1);
                const double gxz = tensorium_RG::sym6_get(G.gamma_tilde, id, 0, 2);
                const double gyy = tensorium_RG::sym6_get(G.gamma_tilde, id, 1, 1);
                const double gyz = tensorium_RG::sym6_get(G.gamma_tilde, id, 1, 2);
                const double gzz = tensorium_RG::sym6_get(G.gamma_tilde, id, 2, 2);
                const double det_val = constraint_monitor_detail::det3(gxx, gxy, gxz, gyy, gyz, gzz);
                stats.max_det_drift = std::max(stats.max_det_drift, std::abs(det_val - 1.0));

                const double Axx = tensorium_RG::sym6_get(G.A_tilde, id, 0, 0);
                const double Axy = tensorium_RG::sym6_get(G.A_tilde, id, 0, 1);
                const double Axz = tensorium_RG::sym6_get(G.A_tilde, id, 0, 2);
                const double Ayy = tensorium_RG::sym6_get(G.A_tilde, id, 1, 1);
                const double Ayz = tensorium_RG::sym6_get(G.A_tilde, id, 1, 2);
                const double Azz = tensorium_RG::sym6_get(G.A_tilde, id, 2, 2);

                const double gixx = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 0, 0);
                const double gixy = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 0, 1);
                const double gixz = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 0, 2);
                const double giyy = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 1, 1);
                const double giyz = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 1, 2);
                const double gizz = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, 2, 2);
                const double trace = gixx * Axx + giyy * Ayy + gizz * Azz +
                                     2.0 * (gixy * Axy + gixz * Axz + giyz * Ayz);
                stats.max_trace_A = std::max(stats.max_trace_A, std::abs(trace));
            }

    if (stats.samples > 0)
        stats.l2_H = std::sqrt(accum_H / static_cast<double>(stats.samples));
    else
        stats.l2_H = 0.0;

    return stats;
}

inline void print_constraint_monitor(const ConstraintMonitorStats &stats,
                                     const char *label = "constraints") {
    std::printf("[%s] maxH=%.3e L2H=%.3e max|TrA|=%.3e max|det-1|=%.3e samples=%zu\n", label,
                stats.max_H, stats.l2_H, stats.max_trace_A, stats.max_det_drift, stats.samples);
}

} // namespace tensorium_RG::bssn
