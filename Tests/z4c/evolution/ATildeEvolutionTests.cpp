#ifndef TENSORIUM_BSSN_ATILDE_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_ATILDE_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionATilde.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>
#include <cmath>

namespace {

using Grid = tensorium_RG::Z4cGridSoA<double>;
using Field = tensorium_RG::Field3D<double>;

Field alloc_like(const Field &ref) {
    Field out;
    out.st = ref.st;
    const size_t total = ref.st.nx_tot * ref.st.ny_tot * ref.st.nz_tot;
    out.data = tensorium_RG::aligned_alloc_n<double>(total);
    return out;
}

void alloc_like(const Field ref[6], Field out[6]) {
    for (int s = 0; s < 6; ++s)
        out[s] = alloc_like(ref[s]);
}

double max_abs_interior(const Field &field, const Grid &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    double max_val = 0.0;
    for (size_t i = I0 + padding; i < I1 - padding; ++i)
        for (size_t j = J0 + padding; j < J1 - padding; ++j)
            for (size_t k = K0 + padding; k < K1 - padding; ++k)
                max_val =
                    std::max(max_val, std::abs(field.ptr()[field.idx(i, j, k)]));
    return max_val;
}

struct RegionStats {
    double max_far = 0.0;
    double l2_far = 0.0;
    size_t samples = 0;
};

template <typename Predicate>
RegionStats far_region_stats(const Field rhs[6], const Grid &grid, size_t padding,
                             Predicate is_far) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    RegionStats stats;
    for (size_t i = I0 + padding; i < I1 - padding; ++i)
        for (size_t j = J0 + padding; j < J1 - padding; ++j)
            for (size_t k = K0 + padding; k < K1 - padding; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                if (!is_far(x, y, z))
                    continue;
                double max_comp = 0.0;
                double sum_sq = 0.0;
                for (int s = 0; s < 6; ++s) {
                    const double val = rhs[s].ptr()[rhs[s].idx(i, j, k)];
                    const double abs_val = std::abs(val);
                    max_comp = std::max(max_comp, abs_val);
                    sum_sq += val * val;
                }
                stats.max_far = std::max(stats.max_far, max_comp);
                stats.l2_far += sum_sq;
                ++stats.samples;
            }

    if (stats.samples > 0)
        stats.l2_far = std::sqrt(stats.l2_far / (stats.samples * 6.0));
    return stats;
}

} // namespace

REGISTER_TEST("z4c.evolution.atilde.minkowski",
              "A_tilde RHS zero for Minkowski data", []() {
                  Grid grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  Field rhs[6];
                  alloc_like(grid.A_tilde, rhs);

                  tensorium_RG::z4c::compute_rhs_A_tilde(grid, rhs, 4);

                  for (int s = 0; s < 6; ++s)
                      tensorium::tests::expect_near(max_abs_interior(rhs[s], grid, 4), 0.0,
                                                   0.0, "minkowski A_tilde interior");
              });

REGISTER_TEST("z4c.evolution.atilde.schwarzschild",
              "A_tilde RHS bounded far from Schwarzschild BH", []() {
                  Grid grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
                  tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

                  Field rhs[6];
                  alloc_like(grid.A_tilde, rhs);
                  tensorium_RG::z4c::compute_rhs_A_tilde(grid, rhs, 4);

                  const double r_cut2 = std::pow(8.0 * grid.dx, 2);
                  const auto stats = far_region_stats(
                      rhs, grid, 4,
                      [=](double x, double y, double z) { return x * x + y * y + z * z > r_cut2; });

                  TENSORIUM_TEST_ASSERT(std::isfinite(stats.max_far));
                  TENSORIUM_TEST_ASSERT(std::isfinite(stats.l2_far));
                  tensorium::tests::expect_le(stats.max_far, 1e-1, "schwarzschild max far bound");
                  tensorium::tests::expect_le(stats.l2_far, 1e-2, "schwarzschild L2 far bound");
              });

REGISTER_TEST("z4c.evolution.atilde.bowenyork",
              "A_tilde RHS bounded away from Bowen-York punctures", []() {
                  Grid grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
                  const double P1[3] = {0.0, 0.05, 0.0};
                  const double P2[3] = {0.0, -0.05, 0.0};
                  const double S1[3] = {0.0, 0.0, 0.15};
                  const double S2[3] = {0.0, 0.0, -0.15};
                  tensorium_RG::init::binary_bowen_york_puncture_init(
                      grid, 0.5, -1.5, 0.0, 0.0, P1, S1, 0.5, 1.5, 0.0, 0.0, P2, S2, 1e-4);

                  Field rhs[6];
                  alloc_like(grid.A_tilde, rhs);
                  tensorium_RG::z4c::compute_rhs_A_tilde(grid, rhs, 4);

                  const double r_cut2 = std::pow(8.0 * grid.dx, 2);
                  const auto stats = far_region_stats(
                      rhs, grid, 4, [=](double x, double y, double z) {
                          const double r1 = (x + 1.5) * (x + 1.5) + y * y + z * z;
                          const double r2 = (x - 1.5) * (x - 1.5) + y * y + z * z;
                          return r1 > r_cut2 && r2 > r_cut2;
                      });

                  TENSORIUM_TEST_ASSERT(std::isfinite(stats.max_far));
                  TENSORIUM_TEST_ASSERT(std::isfinite(stats.l2_far));
                  tensorium::tests::expect_le(stats.max_far, 1.0, "bowen-york max far bound");
                  tensorium::tests::expect_le(stats.l2_far, 1e-1, "bowen-york L2 far bound");
              });

#endif
