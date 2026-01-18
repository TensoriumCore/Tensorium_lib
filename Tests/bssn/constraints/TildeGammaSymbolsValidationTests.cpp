#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNCHristoffelTilde.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("bssn.constraints.tilde_gamma_symbols.validate",
              "Validator passes on analytic Minkowski data", []() {
                  tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);

                  const size_t guard = 4;
                  const size_t i0 = I0 + guard;
                  const size_t j0 = J0 + guard;
                  const size_t k0 = K0 + guard;
                  const size_t i1 = I1 - guard;
                  const size_t j1 = J1 - guard;
                  const size_t k1 = K1 - guard;

                  size_t samples = 0;
                  for (size_t i = i0; i < i1; i += 4)
                      for (size_t j = j0; j < j1; j += 4)
                          for (size_t k = k0; k < k1; k += 4) {
                              double Gamma_vals[3][3][3];
                              tensorium_RG::bssn::compute_tildeGamma_symbols(grid, i, j, k,
                                                                              Gamma_vals);
                              tensorium_RG::bssn::validate_tildeGamma_symbols(grid, i, j, k,
                                                                               Gamma_vals);
                              ++samples;
                          }

                  TENSORIUM_TEST_ASSERT(samples > 0);
              });

#endif
