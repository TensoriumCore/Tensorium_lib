#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNRicci.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRHSSweep.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"
#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include <array>
#include <cmath>

namespace {
using Grid = tensorium_RG::BSSNGridSoA<double>;
namespace bssn = tensorium_RG::bssn;

double gauge_wave_value(bssn::BoundaryField field, int component, double x, double t) {
    constexpr double amplitude = 0.1, wave = 0.7;
    const double     H = 1.0 - amplitude * std::sin(wave * (x - t));
    const double     Ht = amplitude * wave * std::cos(wave * (x - t));
    const double     chi = std::pow(H, -1.0 / 3.0), gxx = std::pow(H, 2.0 / 3.0);
    const double     K = -Ht / (2.0 * std::pow(H, 1.5));
    switch (field) {
    case bssn::BoundaryField::Alpha:
        return std::sqrt(H);
    case bssn::BoundaryField::Chi:
        return chi;
    case bssn::BoundaryField::K:
        return K;
    case bssn::BoundaryField::GammaTilde:
        return component == tensorium_RG::XX
                   ? gxx
                   : (component == tensorium_RG::YY || component == tensorium_RG::ZZ ? chi : 0.0);
    case bssn::BoundaryField::GammaTildeInverse:
        return component == tensorium_RG::XX
                   ? 1.0 / gxx
                   : (component == tensorium_RG::YY || component == tensorium_RG::ZZ ? 1.0 / chi
                                                                                     : 0.0);
    case bssn::BoundaryField::ATilde:
        return component == tensorium_RG::XX
                   ? 2.0 / 3.0 * K * gxx
                   : (component == tensorium_RG::YY || component == tensorium_RG::ZZ
                          ? -K * chi / 3.0
                          : 0.0);
    case bssn::BoundaryField::TildeGamma:
        return component == 0 ? -2.0 / 3.0 * Ht * std::pow(H, -5.0 / 3.0) : 0.0;
    default:
        return 0.0;
    }
}

struct GaugeWaveBoundary {
    static constexpr bool evolve_physical_cells = true;
    static constexpr bool radiative_rhs_collar_enabled = false;
    inline static double  step_start = 0.0, step_size = 0.0, stage_time = 0.0;
    static void           set_stage_fraction(double fraction) {
        stage_time = step_start + fraction * step_size;
    }
    static void apply_physical(tensorium_RG::Field3D<double> &, const Grid &, bssn::BoundaryField,
                               int) {}
    static void apply_halo(tensorium_RG::Field3D<double> &field, const Grid &grid,
                           bssn::BoundaryField which, int component) {
        const auto &d = grid.dims;
        for (size_t i = 0; i < grid.st.nx_tot; ++i)
            for (size_t j = 0; j < grid.st.ny_tot; ++j)
                for (size_t k = 0; k < grid.st.nz_tot; ++k)
                    if (i < d.ng || i >= d.ng + d.nx || j < d.ng || j >= d.ng + d.ny || k < d.ng ||
                        k >= d.ng + d.nz) {
                        const double x = grid.x0 + (double(i) - double(d.ng)) * grid.dx;
                        field.ptr()[field.idx(i, j, k)] =
                            gauge_wave_value(which, component, x, stage_time);
                    }
    }
};

double gauge_wave_step_error(double h) {
    using namespace tensorium_RG;
    Grid grid(12, 12, 12, 4, h, h, h);
    grid.x0 = 0.3;
    auto fill = [&](Field3D<double> &field, bssn::BoundaryField which, int component) {
        for (size_t i = 0; i < grid.st.nx_tot; ++i)
            for (size_t j = 0; j < grid.st.ny_tot; ++j)
                for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                    const double x = grid.x0 + (double(i) - double(grid.dims.ng)) * h;
                    field.ptr()[field.idx(i, j, k)] = gauge_wave_value(which, component, x, 0.0);
                }
    };
    fill(grid.alpha, bssn::BoundaryField::Alpha, 0);
    fill(grid.chi, bssn::BoundaryField::Chi, 0);
    fill(grid.K, bssn::BoundaryField::K, 0);
    fill(grid.Theta, bssn::BoundaryField::Theta, 0);
    for (int c = 0; c < 3; ++c) {
        fill(grid.beta[c], bssn::BoundaryField::Beta, c);
        fill(grid.B[c], bssn::BoundaryField::B, c);
        fill(grid.tildeGamma[c], bssn::BoundaryField::TildeGamma, c);
        fill(grid.Z[c], bssn::BoundaryField::Z, c);
    }
    for (int s = 0; s < 6; ++s) {
        fill(grid.gamma_tilde[s], bssn::BoundaryField::GammaTilde, s);
        fill(grid.gamma_tilde_inv[s], bssn::BoundaryField::GammaTildeInverse, s);
        fill(grid.A_tilde[s], bssn::BoundaryField::ATilde, s);
    }
    bssn::BSSNRKStepper<double, GaugeWaveBoundary> stepper(grid, 0);
    bssn::GaugeParameters<double>                  params;
    params.lapse_oplog = 0.0;
    params.lapse_harmonic = 1.0;
    params.use_direct_shift_rhs = true;
    params.shift_Gamma = 0.0;
    stepper.set_gauge_parameters(params);
    stepper.set_state_log_stride(10000);
    GaugeWaveBoundary::step_start = 0.0;
    GaugeWaveBoundary::step_size = 0.05 * h;
    stepper.step(grid, GaugeWaveBoundary::step_size, 0);
    double       error = 0.0;
    const size_t ng = grid.dims.ng;
    for (const auto yz : {std::array<size_t, 2>{ng + 6, ng + 6}, std::array<size_t, 2>{ng, ng + 6},
                          std::array<size_t, 2>{ng, ng}}) {
        const auto id = grid.alpha.idx(ng, yz[0], yz[1]);
        auto       check = [&](const Field3D<double> &f, bssn::BoundaryField which, int c) {
            const double value = f.ptr()[id];
            TENSORIUM_TEST_ASSERT(std::isfinite(value));
            error =
                std::max(error, std::abs(value - gauge_wave_value(which, c, grid.x0,
                                                                        GaugeWaveBoundary::step_size)));
        };
        check(grid.alpha, bssn::BoundaryField::Alpha, 0);
        check(grid.chi, bssn::BoundaryField::Chi, 0);
        check(grid.K, bssn::BoundaryField::K, 0);
        check(grid.Theta, bssn::BoundaryField::Theta, 0);
        for (int s = 0; s < 6; ++s) {
            check(grid.gamma_tilde[s], bssn::BoundaryField::GammaTilde, s);
            check(grid.A_tilde[s], bssn::BoundaryField::ATilde, s);
        }
        for (int c = 0; c < 3; ++c)
            check(grid.tildeGamma[c], bssn::BoundaryField::TildeGamma, c);
    }
    return error;
}

void fill_ccz4_sample(Grid &grid, double alpha_gradient, double gamma_hat_x) {
    using namespace tensorium_RG;
    for (size_t i = 0; i < grid.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                const double x = (double(i) - double(grid.dims.ng)) * grid.dx;
                grid.alpha.ptr()[id] = 1.0 + alpha_gradient * x;
                grid.chi.ptr()[id] = 1.0;
                grid.K.ptr()[id] = 0.4;
                grid.Theta.ptr()[id] = 0.1;
                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[id] = grid.B[c].ptr()[id] = grid.Z[c].ptr()[id] = 0.0;
                    grid.tildeGamma[c].ptr()[id] = c == 0 ? gamma_hat_x : 0.0;
                }
                for (int s = 0; s < 6; ++s) {
                    const double metric = (s == XX || s == YY || s == ZZ) ? 1.0 : 0.0;
                    grid.gamma_tilde[s].ptr()[id] = grid.gamma_tilde_inv[s].ptr()[id] = metric;
                    grid.Ricci[s].ptr()[id] = 0.0;
                    grid.A_tilde[s].ptr()[id] = s == XX ? 0.2 : (s == YY ? -0.2 : 0.0);
                }
            }
}

double gauge_wave_rhs_error(double h) {
    using namespace tensorium_RG;
    constexpr size_t ng = 4, point = ng;
    constexpr double amplitude = 0.1, wave = 0.7, sample_x = 0.3;
    Grid             grid(12, 12, 12, ng, h, h, h);
    grid.x0 = sample_x;
    for (size_t i = 0; i < grid.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                const auto   id = grid.alpha.idx(i, j, k);
                const double x = sample_x + (double(i) - double(ng)) * h;
                const double H = 1.0 - amplitude * std::sin(wave * x);
                const double Ht = amplitude * wave * std::cos(wave * x);
                const double Hx = -Ht;
                const double chi = std::pow(H, -1.0 / 3.0);
                const double gxx = std::pow(H, 2.0 / 3.0);
                const double K = -Ht / (2.0 * std::pow(H, 1.5));
                grid.alpha.ptr()[id] = std::sqrt(H);
                grid.chi.ptr()[id] = chi;
                grid.K.ptr()[id] = K;
                grid.Theta.ptr()[id] = 0.0;
                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[id] = grid.B[c].ptr()[id] = grid.Z[c].ptr()[id] = 0.0;
                    grid.tildeGamma[c].ptr()[id] =
                        c == 0 ? 2.0 / 3.0 * Hx * std::pow(H, -5.0 / 3.0) : 0.0;
                }
                for (int s = 0; s < 6; ++s) {
                    double metric = 0.0, inv = 0.0, A = 0.0;
                    if (s == XX) {
                        metric = gxx;
                        inv = 1.0 / gxx;
                        A = 2.0 / 3.0 * K * gxx;
                    } else if (s == YY || s == ZZ) {
                        metric = chi;
                        inv = 1.0 / chi;
                        A = -K * chi / 3.0;
                    }
                    grid.gamma_tilde[s].ptr()[id] = metric;
                    grid.gamma_tilde_inv[s].ptr()[id] = inv;
                    grid.A_tilde[s].ptr()[id] = A;
                }
            }
    bssn::compute_ricci_bssn(grid, grid.Ricci, false);
    auto            rhs_alpha = make_field(grid.st);
    auto            rhs_chi = make_field(grid.st);
    auto            rhs_K = make_field(grid.st);
    auto            rhs_theta = make_field(grid.st);
    Field3D<double> rhs_metric[6], rhs_A[6], rhs_gamma[3];
    for (auto &f : rhs_metric)
        f = make_field(grid.st);
    for (auto &f : rhs_A)
        f = make_field(grid.st);
    for (auto &f : rhs_gamma)
        f = make_field(grid.st);
    bssn::GaugeParameters<double> params;
    params.lapse_oplog = 0.0;
    params.lapse_harmonic = 1.0;
    bssn::compute_rhs_alpha(grid, rhs_alpha, 0, params);
    bssn::compute_rhs_chi(grid, rhs_chi, 0, params);
    bssn::compute_rhs_gamma_tilde(grid, rhs_metric, 0, params);
    bssn::compute_rhs_A_tilde(grid, rhs_A, 0, params);
    bssn::compute_rhs_K(grid, rhs_K, 0, params);
    bssn::compute_rhs_Theta(grid, rhs_theta, params, 0);
    bssn::compute_rhs_Gamma(grid, rhs_gamma, grid.Z, grid.Theta, params, 0);
    bssn::recompose_rhs_K_from_khat_core(grid, rhs_K, rhs_theta, 0);

    const double H = 1.0 - amplitude * std::sin(wave * sample_x);
    const double Ht = amplitude * wave * std::cos(wave * sample_x);
    const double Htt = amplitude * wave * wave * std::sin(wave * sample_x);
    const double Hx = -Ht, Hxt = -Htt;
    const double chi = std::pow(H, -1.0 / 3.0);
    const double chi_t = -Ht / 3.0 * std::pow(H, -4.0 / 3.0);
    const double gxx = std::pow(H, 2.0 / 3.0);
    const double gxx_t = 2.0 / 3.0 * Ht * std::pow(H, -1.0 / 3.0);
    const double K = -Ht / 2.0 * std::pow(H, -1.5);
    const double Kt = -0.5 * Htt * std::pow(H, -1.5) + 0.75 * Ht * Ht * std::pow(H, -2.5);
    const double gamma_t =
        2.0 / 3.0 * (Hxt * std::pow(H, -5.0 / 3.0) - 5.0 / 3.0 * Hx * Ht * std::pow(H, -8.0 / 3.0));
    const auto   id = grid.alpha.idx(point, ng + 6, ng + 6);
    const double actual[] = {
        rhs_alpha.ptr()[id], rhs_chi.ptr()[id],        rhs_K.ptr()[id],
        rhs_theta.ptr()[id], rhs_metric[XX].ptr()[id], rhs_metric[YY].ptr()[id],
        rhs_A[XX].ptr()[id], rhs_A[YY].ptr()[id],      rhs_gamma[0].ptr()[id]};
    const double exact[] = {Ht / (2.0 * std::sqrt(H)),
                            chi_t,
                            Kt,
                            0.0,
                            gxx_t,
                            chi_t,
                            2.0 / 3.0 * (Kt * gxx + K * gxx_t),
                            -(Kt * chi + K * chi_t) / 3.0,
                            gamma_t};
    double       error = 0.0;
    for (size_t q = 0; q < sizeof(actual) / sizeof(actual[0]); ++q) {
        TENSORIUM_TEST_ASSERT(std::isfinite(actual[q]));
        error = std::max(error, std::abs(actual[q] - exact[q]));
    }
    return error;
}
} // namespace

REGISTER_TEST("z4c.evolution.ccz4_reference.gauge_wave_step_with_analytic_boundary",
              "RK evolution converges at a physical face with stage-time analytic ghosts", []() {
                  const int saved = tensorium_RG::fd::max_spatial_derivative_order();
                  struct Restore {
                      int order;
                      ~Restore() { tensorium_RG::fd::set_max_spatial_derivative_order(order); }
                  } restore{saved};
                  tensorium_RG::fd::set_max_spatial_derivative_order(4);
                  const double coarse = gauge_wave_step_error(0.4);
                  const double medium = gauge_wave_step_error(0.2);
                  const double fine = gauge_wave_step_error(0.1);
                  std::printf("[CCZ4 gauge wave step] errors %.9g %.9g %.9g\n", coarse, medium,
                              fine);
                  TENSORIUM_TEST_ASSERT(coarse > 8.0 * medium);
                  TENSORIUM_TEST_ASSERT(medium > 8.0 * fine);
              });

REGISTER_TEST("z4c.evolution.ccz4_reference.gauge_wave_convergence",
              "CCZ4 RHS converges to a harmonic gauge wave with analytic ghost data", []() {
                  const int saved = tensorium_RG::fd::max_spatial_derivative_order();
                  struct Restore {
                      int order;
                      ~Restore() { tensorium_RG::fd::set_max_spatial_derivative_order(order); }
                  } restore{saved};
                  for (int order : {4, 6}) {
                      tensorium_RG::fd::set_max_spatial_derivative_order(order);
                      const double coarse = gauge_wave_rhs_error(0.4);
                      const double medium = gauge_wave_rhs_error(0.2);
                      const double fine = gauge_wave_rhs_error(0.1);
                      std::printf("[CCZ4 gauge wave RHS order %d] errors %.9g %.9g %.9g\n", order,
                                  coarse, medium, fine);
                      const double ratio = order == 4 ? 10.0 : 40.0;
                      TENSORIUM_TEST_ASSERT(coarse > ratio * medium);
                      TENSORIUM_TEST_ASSERT(medium > ratio * fine);
                  }
              });

REGISTER_TEST("z4c.evolution.ccz4_reference.atilde_theta_term",
              "CCZ4 tracefree-curvature RHS includes K minus 2 Theta", []() {
                  Grid grid(12, 12, 12, 4, 0.1, 0.1, 0.1);
                  fill_ccz4_sample(grid, 0.0, 0.0);
                  tensorium_RG::Field3D<double> rhs[6];
                  for (auto &field : rhs)
                      field = tensorium_RG::make_field(grid.st);
                  bssn::compute_rhs_A_tilde(grid, rhs, 4);
                  const size_t id = grid.alpha.idx(10, 10, 10);
                  const double expected_xx = (0.4 - 2.0 * 0.1) * 0.2 - 2.0 * 0.2 * 0.2;
                  const double expected_yy = (0.4 - 2.0 * 0.1) * -0.2 - 2.0 * 0.2 * 0.2;
                  TENSORIUM_TEST_ASSERT(std::isfinite(rhs[tensorium_RG::XX].ptr()[id]));
                  TENSORIUM_TEST_ASSERT(std::isfinite(rhs[tensorium_RG::YY].ptr()[id]));
                  tensorium::tests::expect_near(rhs[tensorium_RG::XX].ptr()[id], expected_xx, 1e-11,
                                                "CCZ4 A_xx RHS");
                  tensorium::tests::expect_near(rhs[tensorium_RG::YY].ptr()[id], expected_yy, 1e-11,
                                                "CCZ4 A_yy RHS");
              });

REGISTER_TEST(
    "z4c.evolution.ccz4_reference.spatial_z_divergence",
    "CCZ4 Ricci source contains the divergence and symmetrized gradient of Z", []() {
        using namespace tensorium_RG;
        constexpr double a[3] = {0.07, -0.03, 0.02};
        Grid             grid(12, 12, 12, 4, 0.1, 0.1, 0.1);
        fill_ccz4_sample(grid, 0.0, 0.0);
        for (size_t i = 0; i < grid.st.nx_tot; ++i)
            for (size_t j = 0; j < grid.st.ny_tot; ++j)
                for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                    const auto id = grid.alpha.idx(i, j, k);
                    grid.K.ptr()[id] = grid.Theta.ptr()[id] = 0.0;
                    for (int s = 0; s < 6; ++s)
                        grid.A_tilde[s].ptr()[id] = 0.0;
                    const double x[3] = {(double(i) - 4) * grid.dx, (double(j) - 4) * grid.dy,
                                         (double(k) - 4) * grid.dz};
                    for (int c = 0; c < 3; ++c)
                        grid.tildeGamma[c].ptr()[id] = 2.0 * a[c] * x[c];
                }
        bssn::compute_ricci_bssn(grid, grid.Ricci, false);
        auto            rhs_K = make_field(grid.st), rhs_theta = make_field(grid.st);
        Field3D<double> rhs_A[6];
        for (auto &field : rhs_A)
            field = make_field(grid.st);
        bssn::compute_rhs_K(grid, rhs_K, 4);
        bssn::compute_rhs_Theta(grid, rhs_theta, {}, 4);
        bssn::compute_rhs_A_tilde(grid, rhs_A, 4);
        bssn::recompose_rhs_K_from_khat_core(grid, rhs_K, rhs_theta, 4);
        const auto   id = grid.alpha.idx(10, 10, 10);
        const double div = a[0] + a[1] + a[2];
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs_theta.ptr()[id]) && std::isfinite(rhs_K.ptr()[id]));
        tensorium::tests::expect_near(rhs_theta.ptr()[id], div, 1e-11, "CCZ4 Theta divergence Z");
        tensorium::tests::expect_near(rhs_K.ptr()[id], 2.0 * div, 1e-11, "CCZ4 K divergence Z");
        constexpr int diag[3] = {XX, YY, ZZ};
        for (int c = 0; c < 3; ++c) {
            TENSORIUM_TEST_ASSERT(std::isfinite(rhs_A[diag[c]].ptr()[id]));
            tensorium::tests::expect_near(rhs_A[diag[c]].ptr()[id], 2.0 * (a[c] - div / 3.0), 1e-11,
                                          "CCZ4 A tracefree gradient Z");
        }
    });

REGISTER_TEST("z4c.evolution.ccz4_reference.physical_k_lapse_gradient",
              "Recomposed K RHS matches CCZ4 with nonzero Z and lapse gradient", []() {
                  Grid grid(12, 12, 12, 4, 0.1, 0.1, 0.1);
                  fill_ccz4_sample(grid, 0.1, 0.4); // Z^x=0.2, d_x alpha=0.1.
                  auto                          rhs_k = tensorium_RG::make_field(grid.st);
                  auto                          rhs_theta = tensorium_RG::make_field(grid.st);
                  bssn::GaugeParameters<double> params;
                  params.covariant_z4 = false;
                  params.kappa1 = 0.13;
                  params.kappa2 = 0.2;
                  bssn::compute_rhs_K(grid, rhs_k, 4, params);
                  bssn::compute_rhs_Theta(grid, rhs_theta, params, 4);
                  bssn::recompose_rhs_K_from_khat_core(grid, rhs_k, rhs_theta, 4);
                  const size_t id = grid.alpha.idx(10, 10, 10);
                  const double alpha = grid.alpha.ptr()[id];
                  const double expected =
                      alpha * (0.4 * 0.4 - 2.0 * 0.1 * 0.4 - 3.0 * 0.13 * (1.0 + 0.2) * 0.1);
                  TENSORIUM_TEST_ASSERT(std::isfinite(rhs_k.ptr()[id]));
                  tensorium::tests::expect_near(rhs_k.ptr()[id], expected, 1e-10,
                                                "CCZ4 physical K RHS");
              });

REGISTER_TEST(
    "z4c.evolution.ccz4_reference.theta_and_gamma_sources",
    "CCZ4 Theta and Gamma sources match a nonzero constraint sample", []() {
        Grid grid(12, 12, 12, 4, 0.1, 0.1, 0.1);
        fill_ccz4_sample(grid, 0.1, 0.4);
        auto                          rhs_theta = tensorium_RG::make_field(grid.st);
        tensorium_RG::Field3D<double> rhs_gamma[3];
        for (auto &field : rhs_gamma)
            field = tensorium_RG::make_field(grid.st);
        bssn::GaugeParameters<double> params;
        params.covariant_z4 = false;
        params.kappa1 = 0.13;
        params.kappa2 = 0.2;
        bssn::compute_rhs_Theta(grid, rhs_theta, params, 4);
        bssn::compute_rhs_Gamma(grid, rhs_gamma, grid.Z, grid.Theta, params, 4);
        const size_t id = grid.alpha.idx(10, 10, 10);
        const double alpha = grid.alpha.ptr()[id];
        const double theta_expected =
            alpha * 0.5 * (-0.08 + 2.0 / 3.0 * 0.4 * 0.4 - 2.0 * 0.1 * 0.4) - 0.2 * 0.1 -
            alpha * 0.13 * (2.0 + 0.2) * 0.1;
        const double gamma_x_expected = -2.0 * 0.2 * 0.1 - 2.0 * 0.1 * 0.1 -
                                        4.0 / 3.0 * alpha * 0.4 * 0.2 - 2.0 * alpha * 0.13 * 0.2;
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs_theta.ptr()[id]));
        for (const auto &field : rhs_gamma)
            TENSORIUM_TEST_ASSERT(std::isfinite(field.ptr()[id]));
        tensorium::tests::expect_near(rhs_theta.ptr()[id], theta_expected, 1e-10, "CCZ4 Theta RHS");
        tensorium::tests::expect_near(rhs_gamma[0].ptr()[id], gamma_x_expected, 1e-10,
                                      "CCZ4 Gamma^x RHS");
        tensorium::tests::expect_near(rhs_gamma[1].ptr()[id], 0.0, 1e-10, "CCZ4 Gamma^y RHS");
        tensorium::tests::expect_near(rhs_gamma[2].ptr()[id], 0.0, 1e-10, "CCZ4 Gamma^z RHS");
    });

REGISTER_TEST("z4c.evolution.ccz4_reference.shift_tensor_terms",
              "Conformal metric, factor and connection use the correct shift-index contractions",
              []() {
                  Grid grid(12, 12, 12, 4, 0.1, 0.1, 0.1);
                  fill_ccz4_sample(grid, 0.0, 0.0);
                  constexpr double d_beta[3][3] = {
                      {0.11, 0.04, -0.02}, {-0.03, 0.07, 0.05}, {0.02, -0.06, -0.08}};
                  constexpr double gamma[3] = {0.4, -0.2, 0.1};
                  constexpr double A[3][3] = {{0.2, 0.0, 0.0}, {0.0, -0.2, 0.0}, {0.0, 0.0, 0.0}};
                  for (size_t i = 0; i < grid.st.nx_tot; ++i)
                      for (size_t j = 0; j < grid.st.ny_tot; ++j)
                          for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                              const size_t id = grid.alpha.idx(i, j, k);
                              const double x[3] = {(double(i) - 4) * grid.dx,
                                                   (double(j) - 4) * grid.dy,
                                                   (double(k) - 4) * grid.dz};
                              grid.Theta.ptr()[id] = 0.0;
                              for (int a = 0; a < 3; ++a) {
                                  grid.tildeGamma[a].ptr()[id] = gamma[a];
                                  grid.beta[a].ptr()[id] = d_beta[a][0] * x[0] +
                                                           d_beta[a][1] * x[1] +
                                                           d_beta[a][2] * x[2];
                              }
                          }
                  auto                          rhs_chi = tensorium_RG::make_field(grid.st);
                  tensorium_RG::Field3D<double> rhs_metric[6], rhs_gamma[3];
                  for (auto &field : rhs_metric)
                      field = tensorium_RG::make_field(grid.st);
                  for (auto &field : rhs_gamma)
                      field = tensorium_RG::make_field(grid.st);
                  bssn::GaugeParameters<double> params;
                  bssn::compute_rhs_chi(grid, rhs_chi, 4, params);
                  bssn::compute_rhs_gamma_tilde(grid, rhs_metric, 4, params);
                  bssn::compute_rhs_Gamma(grid, rhs_gamma, grid.Z, grid.Theta, params, 4);
                  const size_t id = grid.alpha.idx(10, 10, 10);
                  const double div = d_beta[0][0] + d_beta[1][1] + d_beta[2][2];
                  TENSORIUM_TEST_ASSERT(std::isfinite(rhs_chi.ptr()[id]));
                  tensorium::tests::expect_near(rhs_chi.ptr()[id], 2.0 / 3.0 * (0.4 - div), 1e-11,
                                                "CCZ4 chi RHS");
                  constexpr int row[6] = {0, 0, 0, 1, 1, 2}, col[6] = {0, 1, 2, 1, 2, 2};
                  for (int s = 0; s < 6; ++s) {
                      const int    a = row[s], b = col[s];
                      const double expected = -2.0 * A[a][b] + d_beta[a][b] + d_beta[b][a] -
                                              (a == b ? 2.0 / 3.0 * div : 0.0);
                      TENSORIUM_TEST_ASSERT(std::isfinite(rhs_metric[s].ptr()[id]));
                      tensorium::tests::expect_near(rhs_metric[s].ptr()[id], expected, 1e-11,
                                                    "CCZ4 metric RHS");
                  }
                  for (int a = 0; a < 3; ++a) {
                      double gamma_beta = 0.0;
                      for (int b = 0; b < 3; ++b)
                          gamma_beta += gamma[b] * d_beta[a][b];
                      const double expected = -gamma_beta + 2.0 / 3.0 * gamma[a] * div -
                                              4.0 / 3.0 * 0.4 * gamma[a] / 2.0;
                      TENSORIUM_TEST_ASSERT(std::isfinite(rhs_gamma[a].ptr()[id]));
                      tensorium::tests::expect_near(rhs_gamma[a].ptr()[id], expected, 1e-11,
                                                    "CCZ4 Gamma shift RHS");
                  }
              });
