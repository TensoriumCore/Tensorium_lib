#ifndef TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP
#define TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <limits>

REGISTER_TEST("z4c.constraints.gamma_coherence_no_resync",
              "Gamma constraint reacts when no resync is performed", []() {
                  tensorium_RG::Z4cGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  const size_t ic = I0 + grid.dims.nx / 2;
                  const size_t jc = J0 + grid.dims.ny / 2;
                  const size_t kc = K0 + grid.dims.nz / 2;
                  const size_t id = grid.tildeGamma[0].idx(ic, jc, kc);

                  grid.tildeGamma[0].ptr()[id] += 1e-2; // introduce incoherence

                  auto constraints = tensorium_RG::init::make_constraint_scratch(grid);
                  tensorium_RG::z4c::compute_z4c_constraints(
                      grid, grid.Ricci, constraints.H, constraints.M, constraints.C, 0.0,
                      std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);

                  const auto stats = tensorium::tests::compute_invariants(grid, 4);
                  const auto tol = tensorium_RG::z4c::compute_invariant_tolerances(grid);
                  const double gamma_violation = tensorium::tests::max_gamma_violation_z4c(grid, 4);
                  TENSORIUM_TEST_ASSERT(gamma_violation > 10.0 * tol.gamma_tol);
              });

#endif // TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP
