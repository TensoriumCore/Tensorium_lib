#ifndef TENSORIUM_BSSN_GAMMA_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_GAMMA_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGamma.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionZ4C.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>

namespace {

void alloc_rhs(const tensorium_RG::BSSNGridSoA<double> &grid,
               tensorium_RG::Field3D<double>            rhs[3]) {
    const size_t total = grid.tildeGamma[0].st.nx_tot * grid.tildeGamma[0].st.ny_tot *
                         grid.tildeGamma[0].st.nz_tot;
    for (int c = 0; c < 3; ++c) {
        rhs[c].st = grid.tildeGamma[c].st;
        rhs[c].data = tensorium_RG::aligned_alloc_n<double>(total);
    }
}

void alloc_scalar_rhs(const tensorium_RG::Field3D<double> &src, tensorium_RG::Field3D<double> &dst) {
    dst.st = src.st;
    const size_t total = src.st.nx_tot * src.st.ny_tot * src.st.nz_tot;
    dst.data = tensorium_RG::aligned_alloc_n<double>(total);
}

struct FarStats {
    double max_far = 0.0;
    size_t samples = 0;
};

template <typename FarSelector>
FarStats far_region_stats(const tensorium_RG::BSSNGridSoA<double> &grid,
                          const tensorium_RG::Field3D<double> rhs[3], size_t padding,
                          FarSelector is_far) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    FarStats stats;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                if (!is_far(x, y, z))
                    continue;
                stats.samples += 1;
                for (int c = 0; c < 3; ++c) {
                    const double val = rhs[c].ptr()[rhs[c].idx(i, j, k)];
                    stats.max_far = std::max(stats.max_far, std::abs(val));
                }
            }
    return stats;
}

double interior_max(const tensorium_RG::BSSNGridSoA<double> &grid,
                    const tensorium_RG::Field3D<double>      rhs[3], size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    double max_val = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k)
                for (int c = 0; c < 3; ++c)
                    max_val =
                        std::max(max_val, std::abs(rhs[c].ptr()[rhs[c].idx(i, j, k)]));
    return max_val;
}

} // namespace

REGISTER_TEST("bssn.evolution.gamma_contracted", "Γ^i RHS behaves across datasets", []() {
    const size_t padding = 4;
    const double r_cut = 4.0;

    auto run_case = [&](auto init_fn, const char *label, double far_tol) {
        tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
        init_fn(grid);

        tensorium_RG::Field3D<double> rhs[3];
        alloc_rhs(grid, rhs);
        tensorium_RG::bssn::GaugeParameters<double> params;
        tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs, grid.Z, grid.Theta, params, padding);

        const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
        const double max_int = interior_max(grid, rhs, padding);

        if (far_tol == 0.0) {
            tensorium::tests::expect_le(max_int, 5.0 * tol.metric_tol,
                                        std::string("Γ rhs stationary (") + label + ")");
        } else {
            const auto stats = far_region_stats(
                grid, rhs, padding, [&](double x, double y, double z) {
                    return (x * x + y * y + z * z) > (r_cut * r_cut);
                });
            TENSORIUM_TEST_ASSERT(stats.samples > 0);
            tensorium::tests::expect_le(stats.max_far, far_tol,
                                        std::string("Γ rhs far bound (") + label + ")");
        }
    };

    run_case([](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, "minkowski", 0.0);
    run_case([](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
             "schwarzschild", 5e-3);

    run_case(
        [&](auto &g) {
            const double P1[3] = {0.0, 0.05, 0.0};
            const double P2[3] = {0.0, -0.05, 0.0};
            const double S1[3] = {0.0, 0.0, 0.15};
            const double S2[3] = {0.0, 0.0, -0.15};
            tensorium_RG::init::binary_bowen_york_puncture_init(
                g, 0.5, -1.5, 0.0, 0.0, P1, S1, 0.5, 1.5, 0.0, 0.0, P2, S2, 1e-4);
        },
        "bowen_york", 2e-2);
});

REGISTER_TEST("bssn.evolution.gamma_theta_ignore_auxiliary_z",
              "Gamma and Theta RHS derive Z from contracted Gamma, not the stored Z field", []() {
                  constexpr size_t padding = 4;
                  tensorium_RG::BSSNGridSoA<double> grid(24, 22, 20, 4, 0.25, 0.22, 0.20);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  for (size_t i = I0; i < I1; ++i)
                      for (size_t j = J0; j < J1; ++j)
                          for (size_t k = K0; k < K1; ++k) {
                              const size_t idx = grid.alpha.idx(i, j, k);
                              double       x, y, z;
                              grid.coords(i, j, k, x, y, z);
                              grid.alpha.ptr()[idx] = 1.0 + 5e-3 * x - 3e-3 * y;
                              grid.beta[0].ptr()[idx] = 0.03 * y;
                              grid.beta[1].ptr()[idx] = -0.02 * x;
                              grid.beta[2].ptr()[idx] = 0.01 * z;
                              grid.K.ptr()[idx] = 0.04 + 0.01 * x - 0.005 * z;
                              grid.Theta.ptr()[idx] = -0.015 * y + 0.01 * z;
                              grid.Z[0].ptr()[idx] = 0.1 + 0.02 * x;
                              grid.Z[1].ptr()[idx] = -0.05 + 0.03 * y;
                              grid.Z[2].ptr()[idx] = 0.08 - 0.01 * z;
                          }

                  tensorium_RG::Field3D<double> rhs_gamma_a[3];
                  tensorium_RG::Field3D<double> rhs_gamma_b[3];
                  tensorium_RG::Field3D<double> rhs_theta_a;
                  tensorium_RG::Field3D<double> rhs_theta_b;
                  alloc_rhs(grid, rhs_gamma_a);
                  alloc_rhs(grid, rhs_gamma_b);
                  alloc_scalar_rhs(grid.Theta, rhs_theta_a);
                  alloc_scalar_rhs(grid.Theta, rhs_theta_b);

                  tensorium_RG::bssn::GaugeParameters<double> params;
                  params.evolve_Z = true;
                  params.ko_sigma = 0.0;

                  tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs_gamma_a, grid.Z, grid.Theta,
                                                       params, padding);
                  tensorium_RG::bssn::compute_rhs_Theta(grid, rhs_theta_a, params, padding);

                  for (size_t i = I0; i < I1; ++i)
                      for (size_t j = J0; j < J1; ++j)
                          for (size_t k = K0; k < K1; ++k) {
                              const size_t idx = grid.alpha.idx(i, j, k);
                              double       x, y, z;
                              grid.coords(i, j, k, x, y, z);
                              grid.Z[0].ptr()[idx] = 9.0 - 0.5 * x + 0.25 * y;
                              grid.Z[1].ptr()[idx] = -7.0 + 0.4 * y - 0.2 * z;
                              grid.Z[2].ptr()[idx] = 5.0 + 0.3 * z + 0.1 * x;
                          }

                  tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs_gamma_b, grid.Z, grid.Theta,
                                                       params, padding);
                  tensorium_RG::bssn::compute_rhs_Theta(grid, rhs_theta_b, params, padding);

                  double max_gamma_diff = 0.0;
                  double max_theta_diff = 0.0;
                  for (size_t i = I0 + padding; i < I1 - padding; ++i)
                      for (size_t j = J0 + padding; j < J1 - padding; ++j)
                          for (size_t k = K0 + padding; k < K1 - padding; ++k) {
                              for (int c = 0; c < 3; ++c) {
                                  const size_t idx = rhs_gamma_a[c].idx(i, j, k);
                                  max_gamma_diff =
                                      std::max(max_gamma_diff, std::abs(rhs_gamma_a[c].ptr()[idx] -
                                                                        rhs_gamma_b[c].ptr()[idx]));
                              }
                              const size_t idx_theta = rhs_theta_a.idx(i, j, k);
                              max_theta_diff =
                                  std::max(max_theta_diff, std::abs(rhs_theta_a.ptr()[idx_theta] -
                                                                    rhs_theta_b.ptr()[idx_theta]));
                          }

                  tensorium::tests::expect_le(max_gamma_diff, 1e-13,
                                              "Gamma RHS is invariant under auxiliary Z changes");
                  tensorium::tests::expect_le(max_theta_diff, 1e-13,
                                              "Theta RHS is invariant under auxiliary Z changes");
              });

REGISTER_TEST("bssn.evolution.z_rhs_is_diagnostic_only",
              "Z_i RHS stays zero even when evolve_Z is enabled", []() {
                  constexpr size_t padding = 4;
                  tensorium_RG::BSSNGridSoA<double> grid(24, 22, 20, 4, 0.25, 0.22, 0.20);
                  tensorium_RG::init::minkowski(grid, 0.0);

                  size_t I0, I1, J0, J1, K0, K1;
                  grid.domain_bounds(I0, I1, J0, J1, K0, K1);
                  for (size_t i = I0; i < I1; ++i)
                      for (size_t j = J0; j < J1; ++j)
                          for (size_t k = K0; k < K1; ++k) {
                              const size_t idx = grid.alpha.idx(i, j, k);
                              double       x, y, z;
                              grid.coords(i, j, k, x, y, z);
                              grid.Z[0].ptr()[idx] = 0.1 + 0.02 * x;
                              grid.Z[1].ptr()[idx] = -0.05 + 0.03 * y;
                              grid.Z[2].ptr()[idx] = 0.08 - 0.01 * z;
                              grid.Theta.ptr()[idx] = 0.02 + 0.01 * x;
                              grid.K.ptr()[idx] = -0.03 + 0.02 * y;
                              grid.A_tilde[0].ptr()[idx] = 0.01 * z;
                          }

                  tensorium_RG::Field3D<double> rhs_Z[3];
                  alloc_rhs(grid, rhs_Z);

                  tensorium_RG::bssn::GaugeParameters<double> params;
                  params.evolve_Z = true;

                  tensorium_RG::bssn::compute_rhs_Z(grid, rhs_Z, params, padding);

                  double max_abs = 0.0;
                  for (size_t i = I0 + padding; i < I1 - padding; ++i)
                      for (size_t j = J0 + padding; j < J1 - padding; ++j)
                          for (size_t k = K0 + padding; k < K1 - padding; ++k)
                              for (int c = 0; c < 3; ++c) {
                                  const size_t idx = rhs_Z[c].idx(i, j, k);
                                  max_abs = std::max(max_abs, std::abs(rhs_Z[c].ptr()[idx]));
                              }

                  tensorium::tests::expect_le(max_abs, 1e-15, "Z_i RHS is zeroed");
              });

#endif
