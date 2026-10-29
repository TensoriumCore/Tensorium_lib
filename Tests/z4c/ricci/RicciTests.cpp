#include "../../framework/TestRegistry.hpp"
#include "../../framework/Assertions.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjection.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

namespace {
double conformal_ricci_error(double h) {
    using namespace tensorium_RG;
    constexpr size_t n = 12, ng = 4, point = ng + 5;
    constexpr double a = 0.7;
    Z4cGridSoA<double> grid(n, n, n, ng, h, h, h);
    for (size_t i = 0; i < grid.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                const size_t id = grid.chi.idx(i, j, k);
                const double x = 0.3 + (double(i) - double(point)) * h;
                grid.chi.ptr()[id] = std::exp(-2 * a * x);
                for (int s = 0; s < 6; ++s) {
                    const double value = (s == XX || s == YY || s == ZZ) ? 1.0 : 0.0;
                    grid.gamma_tilde[s].ptr()[id] = grid.gamma_tilde_inv[s].ptr()[id] = value;
                }
                for (int c = 0; c < 3; ++c)
                    grid.tildeGamma[c].ptr()[id] = 0.0;
            }
    bssn::compute_ricci_bssn(grid, grid.Ricci, false);
    const auto id = grid.chi.idx(point, point, point);
    const double exact[6] = {0.0, 0.0, 0.0, -a*a, 0.0, -a*a};
    double max_error = 0.0;
    for (int s = 0; s < 6; ++s) {
        const double value = grid.Ricci[s].ptr()[id];
        TENSORIUM_TEST_ASSERT(std::isfinite(value));
        max_error = std::max(max_error, std::abs(value - exact[s]));
    }
    return max_error;
}
} // namespace

REGISTER_TEST("z4c.ricci.conformal_factor_convergence",
              "Ricci converges to a non-flat conformally flat analytic metric", []() {
    const int saved = tensorium_RG::fd::max_spatial_derivative_order();
    struct Restore { int order; ~Restore() { tensorium_RG::fd::set_max_spatial_derivative_order(order); } } restore{saved};
    for (int order : {4, 6}) {
        tensorium_RG::fd::set_max_spatial_derivative_order(order);
        const double coarse = conformal_ricci_error(0.4);
        const double medium = conformal_ricci_error(0.2);
        const double fine = conformal_ricci_error(0.1);
        const double ratio = order == 4 ? 12.0 : 40.0;
        TENSORIUM_TEST_ASSERT(coarse > ratio * medium);
        TENSORIUM_TEST_ASSERT(medium > ratio * fine);
    }
});

REGISTER_TEST("z4c.ricci.anisotropic_analytic",
              "Ricci agrees with an anisotropic analytic metric and its rotation", []() {
    using namespace tensorium_RG;
    const int old_order = fd::max_spatial_derivative_order();
    struct RestoreOrder {
        int order;
        ~RestoreOrder() { fd::set_max_spatial_derivative_order(order); }
    } restore{old_order};

    constexpr double a = 0.4;
    constexpr double h = 0.05;
    constexpr size_t ng = 4;
    constexpr int row[6] = {0, 0, 0, 1, 1, 2};
    constexpr int col[6] = {0, 1, 2, 1, 2, 2};
    for (int order : {4, 6}) {
        fd::set_max_spatial_derivative_order(order);
        for (double angle : {0.0, 0.6}) {
            Z4cGridSoA<double> grid(24, 24, 24, ng, h, h, h);
            const double n[3] = {std::cos(angle), std::sin(angle), 0.0};
            const double m[3] = {-n[1], n[0], 0.0};
            for (size_t i = 0; i < grid.chi.st.nx_tot; ++i)
                for (size_t j = 0; j < grid.chi.st.ny_tot; ++j)
                    for (size_t k = 0; k < grid.chi.st.nz_tot; ++k) {
                        const auto id = grid.chi.idx(i, j, k);
                        const double x = (double(i) - double(ng)) * h;
                        const double y = (double(j) - double(ng)) * h;
                        const double e = std::exp(2.0 * a * (n[0] * x + n[1] * y));
                        grid.chi.ptr()[id] = 1.0;
                        for (int s = 0; s < 6; ++s) {
                            const int p = row[s], q = col[s];
                            const double zz = (p == 2 && q == 2) ? 1.0 : 0.0;
                            grid.gamma_tilde[s].ptr()[id] = e * n[p] * n[q] + m[p] * m[q] / e + zz;
                            grid.gamma_tilde_inv[s].ptr()[id] = n[p] * n[q] / e + e * m[p] * m[q] + zz;
                        }
                        for (int p = 0; p < 3; ++p)
                            grid.tildeGamma[p].ptr()[id] = 2.0 * a * n[p] / e;
                    }

            bssn::compute_ricci_bssn(grid, grid.Ricci, false);
            const size_t i = ng + 10, j = ng + 11, k = ng + 12;
            const auto id = grid.chi.idx(i, j, k);
            const double u = h * (10.0 * n[0] + 11.0 * n[1]);
            double baseline[6];
            for (int s = 0; s < 6; ++s) {
                const int p = row[s], q = col[s];
                const double exact = -2.0 * a * a *
                    (n[p] * n[q] + std::exp(-4.0 * a * u) * m[p] * m[q]);
                const double actual = grid.Ricci[s].ptr()[id];
                baseline[s] = actual;
                TENSORIUM_TEST_ASSERT(std::isfinite(actual));
                tensorium::tests::expect_near(actual, exact, 2e-6, "analytic Ricci component " + std::to_string(s));
            }

            // The evolved connection is independent off the constraint surface. A constant
            // perturbation has zero derivative, leaving only Gamma^k Gamma_(ij)k.
            constexpr double delta = 0.07;
            for (size_t idx = 0; idx < grid.chi.st.nx_tot * grid.chi.st.ny_tot * grid.chi.st.nz_tot; ++idx)
                for (int p = 0; p < 3; ++p)
                    grid.tildeGamma[p].ptr()[idx] += delta * n[p];
            bssn::compute_ricci_bssn(grid, grid.Ricci, false);
            const double e = std::exp(2.0 * a * u);
            for (int s = 0; s < 6; ++s) {
                const int p = row[s], q = col[s];
                const double expected_delta = delta * a *
                    (e * n[p] * n[q] - m[p] * m[q] / e);
                const double actual = grid.Ricci[s].ptr()[id];
                TENSORIUM_TEST_ASSERT(std::isfinite(actual));
                tensorium::tests::expect_near(actual - baseline[s], expected_delta, 2e-6,
                                               "Ricci connection perturbation " + std::to_string(s));
            }
        }
    }
});

REGISTER_TEST("z4c.ricci.flat_sheared_coordinates",
              "Ricci vanishes in nonlinear volume-preserving coordinates", []() {
    using namespace tensorium_RG;
    constexpr size_t ng = 4;
    constexpr double h = 0.05, a = 0.4, b = 0.3;
    Z4cGridSoA<double> grid(24, 24, 24, ng, h, h, h);
    // Pull back flat space by X=x-a*cos(y)-b*cos(z), Y=y, Z=z.
    // Its determinant is one but the metric and its contracted connection vary.
    for (size_t i = 0; i < grid.chi.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.chi.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.chi.st.nz_tot; ++k) {
                const auto id = grid.chi.idx(i, j, k);
                const double y = (double(j) - double(ng)) * h;
                const double z = (double(k) - double(ng)) * h;
                const double p = a * std::sin(y), q = b * std::sin(z);
                const double metric[6] = {1.0, p, q, 1.0 + p * p, p * q, 1.0 + q * q};
                const double inverse[6] = {1.0 + p * p + q * q, -p, -q, 1.0, 0.0, 1.0};
                grid.chi.ptr()[id] = 1.0;
                for (int s = 0; s < 6; ++s) {
                    grid.gamma_tilde[s].ptr()[id] = metric[s];
                    grid.gamma_tilde_inv[s].ptr()[id] = inverse[s];
                }
                grid.tildeGamma[0].ptr()[id] = a * std::cos(y) + b * std::cos(z);
                grid.tildeGamma[1].ptr()[id] = 0.0;
                grid.tildeGamma[2].ptr()[id] = 0.0;
            }
    bssn::compute_ricci_bssn(grid, grid.Ricci, false);
    const auto id = grid.chi.idx(ng + 10, ng + 11, ng + 12);
    for (int s = 0; s < 6; ++s) {
        const double actual = grid.Ricci[s].ptr()[id];
        TENSORIUM_TEST_ASSERT(std::isfinite(actual));
        tensorium::tests::expect_near(actual, 0.0, 2e-6, "flat sheared Ricci component " + std::to_string(s));
    }
});

REGISTER_TEST("z4c.ricci.flat", "Ricci tensor close to zero for flat data", []() {
    tensorium_RG::Z4cGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);
    tensorium_RG::z4c::project_z4c_state(grid);
    auto stats = tensorium::tests::compute_invariants(grid, 4);
    const auto tol = tensorium_RG::z4c::compute_invariant_tolerances(grid);
    tensorium::tests::expect_le(stats.max_ricci, 5.0 * tol.metric_tol, "Ricci tensor");
    tensorium::tests::expect_le(stats.max_det_deviation, tol.det_tol, "det gamma");
    tensorium::tests::expect_le(stats.max_trace_A, tol.trace_tol, "trace A");
    tensorium::tests::expect_le(stats.max_metric_identity, tol.metric_tol, "metric identity");
    const double gamma_violation = tensorium::tests::max_gamma_violation_z4c(grid, 4);
    tensorium::tests::expect_le(gamma_violation, tol.gamma_tol, "gamma coherence (Z4c)");
    tensorium::tests::expect_le(stats.max_H, 5.0 * tol.metric_tol, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 5.0 * tol.metric_tol, "Momentum constraint");
    tensorium::tests::expect_le(gamma_violation, 5.0 * tol.gamma_tol,
                                "Contracted Gamma constraint (Z4c)");
    TENSORIUM_TEST_ASSERT(stats.min_chi > 0.0);
});

REGISTER_TEST("z4c.ricci.vacuum",
              "Ricci remains small for Schwarzschild away from puncture", []() {
    tensorium_RG::Z4cGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);
    tensorium_RG::z4c::project_z4c_state(grid);
    const double domain_extent = grid.dims.nx * grid.dx;
    auto stats = tensorium::tests::compute_invariants(grid, 4, 2.0, 0.35 * domain_extent);
    const auto tol = tensorium_RG::z4c::compute_invariant_tolerances(grid);
    tensorium::tests::expect_le(stats.max_ricci, 2.0e4 * tol.metric_tol, "Ricci tensor");
    tensorium::tests::expect_le(stats.max_det_deviation, 10.0 * tol.det_tol, "det gamma");
    tensorium::tests::expect_le(stats.max_trace_A, 10.0 * tol.trace_tol, "trace A");
    tensorium::tests::expect_le(stats.max_metric_identity, 10.0 * tol.metric_tol, "metric identity");
    const double gamma_violation =
        tensorium::tests::max_gamma_violation_z4c(grid, 4, 2.0, 0.35 * domain_extent);
    tensorium::tests::expect_le(gamma_violation, 10.0 * tol.gamma_tol,
                                "gamma coherence (Z4c)");
    tensorium::tests::expect_le(stats.max_H, 80.0 * tol.metric_tol, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 80.0 * tol.metric_tol, "Momentum constraint");
    tensorium::tests::expect_le(gamma_violation, 80.0 * tol.gamma_tol,
                                "Contracted Gamma constraint (Z4c)");
    TENSORIUM_TEST_ASSERT(stats.min_chi > 0.0);
});
