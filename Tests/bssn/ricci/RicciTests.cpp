#include "../../framework/TestRegistry.hpp"
#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjection.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("bssn.ricci.flat", "Ricci tensor close to zero for flat data", []() {
    tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);
    tensorium_RG::bssn::project_bssn_state(grid);
    auto stats = tensorium::tests::compute_invariants(grid, 4);
    const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
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

REGISTER_TEST("bssn.ricci.vacuum",
              "Ricci remains small for Schwarzschild away from puncture", []() {
    tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);
    tensorium_RG::bssn::project_bssn_state(grid);
    const double domain_extent = grid.dims.nx * grid.dx;
    auto stats = tensorium::tests::compute_invariants(grid, 4, 2.0, 0.35 * domain_extent);
    const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
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
