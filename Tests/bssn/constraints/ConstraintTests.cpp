#include "../../framework/TestRegistry.hpp"
#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

REGISTER_TEST("bssn.constraints.minkowski", "Constraints vanish for Minkowski data", []() {
    tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);
    auto stats = tensorium::tests::compute_invariants(grid, 4);
    const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
    tensorium::tests::expect_le(stats.max_H, 5.0 * tol.metric_tol, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 5.0 * tol.metric_tol, "Momentum constraint");
    const double gamma_violation = tensorium::tests::max_gamma_violation_z4c(grid, 4);
    tensorium::tests::expect_le(gamma_violation, 5.0 * tol.gamma_tol,
                                "Gamma coherence (Z4c)");
});

REGISTER_TEST("bssn.constraints.schwarzschild",
              "Constraints bounded away from puncture", []() {
    tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);
    const double domain_extent = grid.dims.nx * grid.dx;
    auto stats = tensorium::tests::compute_invariants(grid, 4, 2.0, 0.4 * domain_extent);
    const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
    tensorium::tests::expect_le(stats.max_H, 80.0 * tol.metric_tol, "Hamiltonian constraint");
    tensorium::tests::expect_le(stats.max_M, 80.0 * tol.metric_tol, "Momentum constraint");
    const double gamma_violation =
        tensorium::tests::max_gamma_violation_z4c(grid, 4, 2.0, 0.4 * domain_extent);
    tensorium::tests::expect_le(gamma_violation, 80.0 * tol.gamma_tol,
                                "Gamma coherence (Z4c)");
});

REGISTER_TEST("bssn.constraints.gamma_coherence", "Gamma constraint fails on inconsistent data",
              []() {
                  tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  const size_t ic = I0 + grid.dims.nx / 2;
                  const size_t jc = J0 + grid.dims.ny / 2;
                  const size_t kc = K0 + grid.dims.nz / 2;
                  const size_t id = grid.tildeGamma[0].idx(ic, jc, kc);

                  grid.tildeGamma[0].ptr()[id] += 1e-2; // violate stored Γ̃^i only

                  auto stats = tensorium::tests::compute_invariants(grid, 4);
                  const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
                  const double gamma_violation = tensorium::tests::max_gamma_violation_z4c(grid, 4);
                  TENSORIUM_TEST_ASSERT(gamma_violation > 10.0 * tol.gamma_tol);

                  tensorium_RG::bssn::assert_invariants(grid, "gamma_test", 4, tol, false);
              });

REGISTER_TEST("bssn.constraints.gamma_coherence_stepper_resync",
              "Stepper resynchronizes auxiliary Z_i from Gamma even when evolve_Z is enabled",
              []() {
                  tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  for (size_t i = I0; i < I1; ++i)
                      for (size_t j = J0; j < J1; ++j)
                          for (size_t k = K0; k < K1; ++k) {
                              const size_t idx = grid.alpha.idx(i, j, k);
                              grid.Z[0].ptr()[idx] = 10.0 + 0.1 * double(i);
                              grid.Z[1].ptr()[idx] = -7.0 + 0.2 * double(j);
                              grid.Z[2].ptr()[idx] = 5.0 - 0.15 * double(k);
                          }

                  tensorium_RG::bssn::GaugeParameters<double> params;
                  params.evolve_Z = true;
                  params.apply_rhs_sommerfeld = false;

                  tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative>
                      stepper(grid, 4);
                  stepper.set_gauge_parameters(params);
                  stepper.set_state_log_stride(1000000);

                  stepper.step(grid, 0.01, 0);

                  const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
                  const double gamma_violation = tensorium::tests::max_gamma_violation_z4c(grid, 4);
                  tensorium::tests::expect_le(gamma_violation, 5.0 * tol.gamma_tol,
                                              "Gamma coherence restored by stepper resync");
              });

#include "GammaCoherenceNoResyncTests.cpp"
