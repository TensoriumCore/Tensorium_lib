#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("bssn.initial_data.minkowski", "Check Minkowski invariants", []() {
    tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);
    auto stats = tensorium::tests::compute_invariants(grid, 4);
    tensorium::tests::expect_le(stats.max_det_deviation, 1e-12, "det(gamma~) deviation");
    tensorium::tests::expect_le(stats.max_trace_A, 1e-12, "tr(A~)");
    tensorium::tests::expect_le(stats.max_ricci, 1e-12, "Ricci tensor");
    tensorium::tests::expect_le(stats.max_H, 1e-12, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 1e-12, "Momentum constraint");
    tensorium::tests::expect_le(stats.max_C, 1e-12, "Gamma constraint");
});

REGISTER_TEST("bssn.initial_data.schwarzschild",
              "Schwarzschild isotropic slice produces flat Ricci far from puncture", []() {
    tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

    const double domain_extent = grid.dims.nx * grid.dx;
    const double r_max = 0.4 * domain_extent;
    auto stats = tensorium::tests::compute_invariants(grid, 4, 2.0, r_max);

    tensorium::tests::expect_le(stats.max_det_deviation, 1e-2, "det(gamma~) deviation");
    tensorium::tests::expect_le(stats.max_trace_A, 1e-2, "tr(A~)");
    tensorium::tests::expect_le(stats.max_ricci, 1e-1, "Ricci tensor");
    tensorium::tests::expect_le(stats.max_H, 2e-4, "Hamiltonian constraint");
});

#include "../invariants/ProjectionTests.cpp"
#include "BowenYorkExportTest.cpp"
