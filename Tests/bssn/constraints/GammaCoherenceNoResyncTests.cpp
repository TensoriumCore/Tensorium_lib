#ifndef TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP
#define TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <limits>

REGISTER_TEST("bssn.constraints.gamma_coherence_no_resync",
              "Gamma constraint reacts when no resync is performed", []() {
                  tensorium_RG::BSSNGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  const size_t ic = I0 + grid.dims.nx / 2;
                  const size_t jc = J0 + grid.dims.ny / 2;
                  const size_t kc = K0 + grid.dims.nz / 2;
                  const size_t id = grid.tildeGamma[0].idx(ic, jc, kc);

                  grid.tildeGamma[0].ptr()[id] += 1e-2; // introduce incoherence

                  bool threw = false;
                  try {
                      tensorium_RG::bssn::compute_bssn_constraints(
                          grid, grid.Ricci, grid.Hc, grid.Mc, grid.Cc, 0.0,
                          std::numeric_limits<double>::max(), 0.0, 0.0, 0.0);
                  } catch (const std::exception &) {
                      threw = true;
                  }
                  TENSORIUM_TEST_ASSERT(threw);

                  const auto stats = tensorium::tests::compute_invariants(grid, 4);
                  const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
                  TENSORIUM_TEST_ASSERT(stats.max_gamma_constraint > 10.0 * tol.gamma_tol);
              });

#endif // TENSORIUM_BSSN_GAMMA_COHERENCE_NORESYNC_TESTS_CPP
