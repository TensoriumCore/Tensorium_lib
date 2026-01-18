#ifndef TENSORIUM_BSSN_SCHWARZSCHILD_CONSTRAINT_EXPORT_TEST_CPP
#define TENSORIUM_BSSN_SCHWARZSCHILD_CONSTRAINT_EXPORT_TEST_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjection.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

REGISTER_TEST("bssn.initial_data.schwarzschild_constraints",
              "Exports Schwarzschild isotropic constraints to CSV", []() {
                  constexpr size_t NX = 128;
                  constexpr size_t NG = 4;
                  constexpr double DX = 0.25;
                  tensorium_RG::BSSNGridSoA<double> grid(NX, NX, NX, NG, DX, DX, DX);

                  const double half_extent = 0.5 * NX * DX;
                  grid.x0 = -half_extent;
                  grid.y0 = -half_extent;
                  grid.z0 = -half_extent;

                  tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

                  auto constraints = tensorium_RG::init::make_constraint_scratch(grid);
                  tensorium_RG::bssn::compute_bssn_constraints(
                      grid, grid.Ricci, constraints.H, constraints.M, constraints.C, DX,
                      0.45 * NX * DX, 0.0, 0.0, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);

                  std::ofstream csv("schwarzschild_constraints.csv");
                  TENSORIUM_TEST_ASSERT(csv.good());
                  csv.setf(std::ios::scientific);
                  csv << std::setprecision(12);
                  csv << "x,y,z,r,H,Mx,My,Mz,M_mag,Cx,Cy,Cz,C_mag\n";

                  for (size_t i = I0; i < I1; i += 2) {
                      for (size_t j = J0; j < J1; j += 2) {
                          for (size_t k = K0; k < K1; k += 2) {
                              const size_t id = grid.alpha.idx(i, j, k);
                              double x, y, z;
                              grid.coords(i, j, k, x, y, z);
                              const double r = std::sqrt(x * x + y * y + z * z);
                              const double H = constraints.H.ptr()[id];
                              const double Mx = constraints.M[0].ptr()[id];
                              const double My = constraints.M[1].ptr()[id];
                              const double Mz = constraints.M[2].ptr()[id];
                              const double Mmag = std::sqrt(Mx * Mx + My * My + Mz * Mz);
                              const double Cx = constraints.C[0].ptr()[id];
                              const double Cy = constraints.C[1].ptr()[id];
                              const double Cz = constraints.C[2].ptr()[id];
                              const double Cmag = std::sqrt(Cx * Cx + Cy * Cy + Cz * Cz);
                              csv << x << ',' << y << ',' << z << ',' << r << ',' << H << ','
                                  << Mx << ',' << My << ',' << Mz << ',' << Mmag << ',' << Cx
                                  << ',' << Cy << ',' << Cz << ',' << Cmag << '\n';
                          }
                      }
                  }
                  csv.close();
                  TENSORIUM_TEST_ASSERT(csv.good());
              });

#endif // TENSORIUM_BSSN_SCHWARZSCHILD_CONSTRAINT_EXPORT_TEST_CPP
