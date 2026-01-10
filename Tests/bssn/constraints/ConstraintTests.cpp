#include "../../framework/TestRegistry.hpp"
#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("bssn.constraints.minkowski", "Constraints vanish for Minkowski data", []() {
    tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);
    auto stats = tensorium::tests::compute_invariants(grid, 4);
    tensorium::tests::expect_le(stats.max_H, 1e-12, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 1e-12, "Momentum constraint");
    tensorium::tests::expect_le(stats.max_C, 1e-12, "Gamma constraint");
});

REGISTER_TEST("bssn.constraints.schwarzschild",
              "Constraints bounded away from puncture", []() {
    tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);
    const double domain_extent = grid.dims.nx * grid.dx;
    auto stats = tensorium::tests::compute_invariants(grid, 4, 2.0, 0.4 * domain_extent);
    tensorium::tests::expect_le(stats.max_H, 2e-4, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 2e-4, "Momentum constraint");
    tensorium::tests::expect_le(stats.max_C, 2e-4, "Gamma constraint");
});
