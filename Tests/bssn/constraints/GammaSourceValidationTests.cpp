#if defined(TENSORIUM_BSSN_VALIDATE_TILDE_GAMMA_SYMBOLS)

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGamma.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("bssn.constraints.gamma_source_validation",
              "Gamma-source validation on Schwarzschild initial data", []() {
                  tensorium_RG::BSSNGridSoA<double> grid(40, 40, 40, 4, 0.2, 0.2, 0.2);
                  tensorium_RG::init::schwarzschild_isotropic(grid, 0.5);

                  tensorium_RG::Field3D<double> rhs_gamma[3];
                  for (int q = 0; q < 3; ++q)
                      rhs_gamma[q] = tensorium_RG::make_field(grid.alpha.st);

                  tensorium_RG::bssn::GaugeParameters<double> params;
                  tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs_gamma, grid.Z, grid.Theta,
                                                       params, 4);
              });

#endif
