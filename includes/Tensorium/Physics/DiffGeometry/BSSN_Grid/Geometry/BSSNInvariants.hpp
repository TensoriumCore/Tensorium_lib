#pragma once

#include "BSSNGamma.hpp"
#include "../Fields/BSSNGridSoA.hpp"
#include "Tensorium_Grid/Grid/GridLayout.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tensorium_RG::bssn {

struct SampleLocation {
    size_t i = 0, j = 0, k = 0;
    double x = 0.0, y = 0.0, z = 0.0;
};

struct InvariantStats {
    double         max_det_deviation = 0.0;
    double         l2_det_deviation = 0.0;
    double         mean_det_deviation = 0.0;
    SampleLocation det_location;

    double         max_trace_A = 0.0;
    double         l2_trace_A = 0.0;
    double         mean_trace_A = 0.0;
    SampleLocation trace_location;

    double         max_metric_identity = 0.0;
    double         l2_metric_identity = 0.0;
    double         mean_metric_identity = 0.0;
    SampleLocation metric_location;

    double         max_gamma_constraint = 0.0;
    double         l2_gamma_constraint = 0.0;
    double         mean_gamma_constraint = 0.0;
    SampleLocation gamma_location;

    double         min_chi = std::numeric_limits<double>::infinity();
    SampleLocation chi_location;
    size_t samples = 0;
};

struct InvariantTolerances {
    double det_tol = 0.0;
    double trace_tol = 0.0;
    double metric_tol = 0.0;
    double gamma_tol = 0.0;
    double chi_tol = 0.0;
};

namespace detail {

inline double det3(const double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[1][2]) -
           m[0][1] * (m[0][1] * m[2][2] - m[0][2] * m[1][2]) +
           m[0][2] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
}

} // namespace detail

template <typename T>
inline InvariantStats compute_invariant_stats(const BSSNGridSoA<T> &G, size_t padding = 4,
                                             double r_min = 0.0,
                                             double r_max = std::numeric_limits<double>::max(),
                                             double xc = 0.0, double yc = 0.0, double zc = 0.0) {
    InvariantStats stats;

    size_t I0, I1, J0, J1, K0, K1;
    G.domain_bounds(I0, I1, J0, J1, K0, K1);

    // Samples are gathered strictly inside the physical domain.  We skip
    // "padding" cells from the computational boundary to (1) avoid touching
    // halo points (domain_bounds already drops ng cells) and (2) guarantee
    // that the 4th-order stencils (which require ±2 points) operate on valid
    // data.  The default padding value of 4 matches the standard halo width.
    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return stats;

    double det_sum = 0.0, det_sq = 0.0;
    double tr_sum = 0.0, tr_sq = 0.0;
    double metric_sum = 0.0, metric_sq = 0.0;
    double gamma_sum = 0.0, gamma_sq = 0.0;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                double x, y, z;
                G.coords(i, j, k, x, y, z);
                const double X = x - xc;
                const double Y = y - yc;
                const double Z = z - zc;
                const double r = std::sqrt(X * X + Y * Y + Z * Z);
                if (r <= r_min || r >= r_max)
                    continue;

                const size_t id = G.gamma_tilde[XX].idx(i, j, k);

                double g[3][3];
                double ginv[3][3];
                double A[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b) {
                        const double gval = tensorium_RG::sym6_get(G.gamma_tilde, id, a, b);
                        g[a][b] = gval;
                        g[b][a] = gval;
                        const double gIval = tensorium_RG::sym6_get(G.gamma_tilde_inv, id, a, b);
                        ginv[a][b] = gIval;
                        ginv[b][a] = gIval;
                        const double Aval = tensorium_RG::sym6_get(G.A_tilde, id, a, b);
                        A[a][b] = Aval;
                        A[b][a] = Aval;
                    }

                const double det_dev = std::abs(detail::det3(g) - 1.0);
                if (det_dev > stats.max_det_deviation) {
                    stats.max_det_deviation = det_dev;
                    stats.det_location = SampleLocation{i, j, k, x, y, z};
                }
                det_sum += det_dev;
                det_sq += det_dev * det_dev;

                double traceA = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        traceA += ginv[a][b] * A[a][b];
                const double abs_trace = std::abs(traceA);
                if (abs_trace > stats.max_trace_A) {
                    stats.max_trace_A = abs_trace;
                    stats.trace_location = SampleLocation{i, j, k, x, y, z};
                }
                tr_sum += abs_trace;
                tr_sq += abs_trace * abs_trace;

                double metric_err = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double sum = 0.0;
                        for (int c = 0; c < 3; ++c)
                            sum += g[a][c] * ginv[c][b];
                        const double delta = (a == b) ? 1.0 : 0.0;
                        metric_err = std::max(metric_err, std::abs(sum - delta));
                    }
                if (metric_err > stats.max_metric_identity) {
                    stats.max_metric_identity = metric_err;
                    stats.metric_location = SampleLocation{i, j, k, x, y, z};
                }
                metric_sum += metric_err;
                metric_sq += metric_err * metric_err;

                T div_gamma[3];
                metric_inverse_divergence(G, i, j, k, div_gamma);
                for (int a = 0; a < 3; ++a) {
                    const double gamma_constraint =
                        (double)G.tildeGamma[a].ptr()[id] + (double)div_gamma[a];
                    const double abs_gamma = std::abs(gamma_constraint);
                    if (abs_gamma > stats.max_gamma_constraint) {
                        stats.max_gamma_constraint = abs_gamma;
                        stats.gamma_location = SampleLocation{i, j, k, x, y, z};
                    }
                    gamma_sum += abs_gamma;
                    gamma_sq += abs_gamma * abs_gamma;
                }

                const double chi = (double)G.chi.ptr()[id];
                if (chi < stats.min_chi) {
                    stats.min_chi = chi;
                    stats.chi_location = SampleLocation{i, j, k, x, y, z};
                }

                ++stats.samples;
            }

    if (stats.samples > 0) {
        const double inv_n = 1.0 / (double)stats.samples;
        stats.mean_det_deviation = det_sum * inv_n;
        stats.mean_trace_A = tr_sum * inv_n;
        stats.mean_metric_identity = metric_sum * inv_n;
        stats.mean_gamma_constraint = gamma_sum * inv_n;

        stats.l2_det_deviation = std::sqrt(det_sq * inv_n);
        stats.l2_trace_A = std::sqrt(tr_sq * inv_n);
        stats.l2_metric_identity = std::sqrt(metric_sq * inv_n);
        stats.l2_gamma_constraint = std::sqrt(gamma_sq * inv_n);
    }

    return stats;
}

// Resolution-aware tolerance model: the leading truncation error of the 4th
// order stencils scales like (h / L)^p with p=4, where h is the largest grid
// spacing and L the characteristic domain size.  We blend that error estimate
// with sqrt(epsilon) to avoid demanding more than double precision can offer.
template <typename T>
inline InvariantTolerances compute_invariant_tolerances(const BSSNGridSoA<T> &G,
                                                        double scheme_order = 4.0,
                                                        double safety_factor = 4.0) {
    const double hx = (double)G.dx;
    const double hy = (double)G.dy;
    const double hz = (double)G.dz;
    const double h = std::max({hx, hy, hz});
    const double Lx = G.dims.nx * hx;
    const double Ly = G.dims.ny * hy;
    const double Lz = G.dims.nz * hz;
    const double L = std::max({Lx, Ly, Lz, h});
    const double ratio = (L > 0.0) ? h / L : 1.0;
    const double scheme_error = std::pow(ratio, scheme_order);
    const double machine = std::sqrt(std::numeric_limits<double>::epsilon());

    auto make_tol = [&](double weight) {
        const double scaled = safety_factor * scheme_error * weight;
        return std::max(machine * weight, scaled);
    };

    InvariantTolerances tol;
    tol.det_tol = make_tol(1.0);
    tol.trace_tol = make_tol(1.0);
    tol.metric_tol = make_tol(1.5);
    tol.gamma_tol = make_tol(2.0);
    tol.chi_tol = machine;
    return tol;
}

template <typename T>
inline void assert_invariants(const BSSNGridSoA<T> &G, const char *stage, size_t padding,
                              const InvariantTolerances &tol) {
    const auto stats = compute_invariant_stats(G, padding);
    if (stats.samples == 0)
        return;

    auto describe = [](const SampleLocation &p) {
        std::ostringstream oss;
        oss << "(i=" << p.i << ", j=" << p.j << ", k=" << p.k << ")"
            << " @ (" << p.x << ", " << p.y << ", " << p.z << ')';
        return oss.str();
    };

    auto raise = [&](const std::string &quantity, double value, double bound, double l2,
                     double mean, const SampleLocation &loc) {
        std::ostringstream oss;
        oss << "[BSSN::" << (stage ? stage : "unknown") << "] " << quantity
            << " violation: max=" << value << " (tol=" << bound << ")"
            << " L2=" << l2 << " mean=" << mean << " at " << describe(loc);
        throw std::runtime_error(oss.str());
    };

    if (stats.max_det_deviation > tol.det_tol)
        raise("det(gamma_tilde)", stats.max_det_deviation, tol.det_tol,
              stats.l2_det_deviation, stats.mean_det_deviation, stats.det_location);
    if (stats.max_trace_A > tol.trace_tol)
        raise("tr(A_tilde)", stats.max_trace_A, tol.trace_tol, stats.l2_trace_A,
              stats.mean_trace_A, stats.trace_location);
    if (stats.max_metric_identity > tol.metric_tol)
        raise("gamma * gamma^{-1}", stats.max_metric_identity, tol.metric_tol,
              stats.l2_metric_identity, stats.mean_metric_identity, stats.metric_location);
    if (stats.max_gamma_constraint > tol.gamma_tol)
        raise("Gamma coherence (tildeGamma + ∂_j g^{ij})", stats.max_gamma_constraint,
              tol.gamma_tol, stats.l2_gamma_constraint, stats.mean_gamma_constraint,
              stats.gamma_location);
    if (!(stats.min_chi > tol.chi_tol))
        raise("chi", stats.min_chi, tol.chi_tol, 0.0, stats.min_chi, stats.chi_location);
}

template <typename T>
inline void assert_invariants(const BSSNGridSoA<T> &G, const char *stage, size_t padding = 4) {
    assert_invariants(G, stage, padding, compute_invariant_tolerances(G));
}

} // namespace tensorium_RG::bssn
