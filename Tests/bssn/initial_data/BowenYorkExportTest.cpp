#ifndef TENSORIUM_BSSN_BOWEN_YORK_EXPORT_TEST_CPP
#define TENSORIUM_BSSN_BOWEN_YORK_EXPORT_TEST_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjection.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <cmath>
#include <fstream>
#include <limits>

REGISTER_TEST("bssn.initial_data.bowen_york_export",
              "Exports Bowen-York binary slice/constraints to CSV", []() {
                  constexpr size_t NX = 128;
                  constexpr size_t NG = 4;
                  constexpr double DX = 0.15;
                  tensorium_RG::BSSNGridSoA<double> grid(NX, NX, NX, NG, DX, DX, DX);
                  const double half_extent = 0.5 * NX * DX;
                  grid.x0 = -half_extent;
                  grid.y0 = -half_extent;
                  grid.z0 = -half_extent;

                  const double P1[3] = {0.0, 0.05, 0.0};
                  const double P2[3] = {0.0, -0.05, 0.0};
                  const double S1[3] = {0.0, 0.0, 0.1};
                  const double S2[3] = {0.0, 0.0, -0.1};

                  tensorium_RG::init::binary_bowen_york_puncture_init(grid, 0.5, -2.0, 0.0, 0.0, P1,
                                                                      S1, 0.5, 2.0, 0.0, 0.0, P2, S2,
                                                                      1e-4);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);

                  tensorium_RG::bssn::compute_bssn_constraints(
                      grid, grid.Ricci, grid.Hc, grid.Mc, grid.Cc, 0.5,
                      std::numeric_limits<double>::max(), 0.0, 0.0, 0.0);

                  const auto stats = tensorium::tests::compute_invariants(grid, 4, 0.5,
                                                                          0.5 * NX * DX);
                  TENSORIUM_TEST_ASSERT(stats.min_chi > 0.0);

                  const size_t js = J0 + grid.dims.ny / 2;
                  std::ofstream slice("bowen_york_slice_xy.csv");
                  TENSORIUM_TEST_ASSERT(slice.good());
                  slice << "x,z,alpha,chi,K,Gamma_x,Gamma_y,Gamma_z\n";
                  for (size_t i = I0; i < I1; ++i) {
                      for (size_t k = K0; k < K1; ++k) {
                          const size_t id = grid.alpha.idx(i, js, k);
                          double x, y, z;
                          grid.coords(i, js, k, x, y, z);
                          (void)y;
                          const double Gx = grid.tildeGamma[0].ptr()[id];
                          const double Gy = grid.tildeGamma[1].ptr()[id];
                          const double Gz = grid.tildeGamma[2].ptr()[id];
                          slice << x << ',' << z << ',' << grid.alpha.ptr()[id] << ','
                                << grid.chi.ptr()[id] << ',' << grid.K.ptr()[id] << ',' << Gx
                                << ',' << Gy << ',' << Gz << '\n';
                      }
                  }
                  slice.close();
                  TENSORIUM_TEST_ASSERT(slice.good());

                  std::ofstream constr("bowen_york_constraints.csv");
                  TENSORIUM_TEST_ASSERT(constr.good());
                  constr << "x,y,z,H,M,C\n";
                  for (size_t i = I0; i < I1; i += 4) {
                      for (size_t j = J0; j < J1; j += 4) {
                          for (size_t k = K0; k < K1; k += 4) {
                              const size_t id = grid.alpha.idx(i, j, k);
                              double x, y, z;
                              grid.coords(i, j, k, x, y, z);
                              const double H = grid.Hc.ptr()[id];
                              const double Mx = grid.Mc[0].ptr()[id];
                              const double My = grid.Mc[1].ptr()[id];
                              const double Mz = grid.Mc[2].ptr()[id];
                              const double Cx = grid.Cc[0].ptr()[id];
                              const double Cy = grid.Cc[1].ptr()[id];
                              const double Cz = grid.Cc[2].ptr()[id];
                              const double Mmag = std::sqrt(Mx * Mx + My * My + Mz * Mz);
                              const double Cmag = std::sqrt(Cx * Cx + Cy * Cy + Cz * Cz);
                              constr << x << ',' << y << ',' << z << ',' << H << ',' << Mmag
                                     << ',' << Cmag << '\n';
                          }
                      }
                  }
                  constr.close();
                  TENSORIUM_TEST_ASSERT(constr.good());
              });

#endif // TENSORIUM_BSSN_BOWEN_YORK_EXPORT_TEST_CPP
