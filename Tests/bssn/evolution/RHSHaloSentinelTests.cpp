#ifndef TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP
#define TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/CUDA/BSSNCudaGridOps.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridDevice.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridViews.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;
using Stepper = tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative>;

bool field_interior_finite(const Grid &grid, const tensorium_RG::Field3D<double> &field,
                           size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t guard = std::max<size_t>(padding, size_t(2));
    const size_t i0 = std::min(I0 + guard, I1);
    const size_t j0 = std::min(J0 + guard, J1);
    const size_t k0 = std::min(K0 + guard, K1);
    const size_t i1 = (I1 > guard) ? I1 - guard : I1;
    const size_t j1 = (J1 > guard) ? J1 - guard : J1;
    const size_t k1 = (K1 > guard) ? K1 - guard : K1;
    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return true;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t idx = field.idx(i, j, k);
                const double val = field.ptr()[idx];
                if (!std::isfinite(val))
                    return false;
            }
    return true;
}

void expect_grid_interior_finite(const Grid &grid, size_t padding) {
    TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.alpha, padding));
    TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.chi, padding));
    TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.K, padding));
    for (int c = 0; c < 3; ++c) {
        TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.beta[c], padding));
        TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.B[c], padding));
        TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.tildeGamma[c], padding));
    }
    for (int s = 0; s < 6; ++s) {
        TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.gamma_tilde[s], padding));
        TENSORIUM_TEST_ASSERT(field_interior_finite(grid, grid.A_tilde[s], padding));
    }
}

void poison_field(tensorium_RG::Field3D<double> &field) {
    double *ptr = field.ptr();
    const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
    const double poison = std::numeric_limits<double>::quiet_NaN();
    for (size_t idx = 0; idx < total; ++idx)
        ptr[idx] = poison;
}

void poison_workspace(tensorium_RG::bssn::BSSNRHSWorkspace<double> &rhs) {
    poison_field(rhs.alpha);
    poison_field(rhs.chi);
    poison_field(rhs.K);
    for (int c = 0; c < 3; ++c) {
        poison_field(rhs.beta[c]);
        poison_field(rhs.B[c]);
        poison_field(rhs.tildeGamma[c]);
    }
    for (int s = 0; s < 6; ++s) {
        poison_field(rhs.gamma_tilde[s]);
        poison_field(rhs.A_tilde[s]);
    }
}

void fill_field_pattern(tensorium_RG::Field3D<double> &field, double base, double slope) {
    const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
    for (size_t idx = 0; idx < total; ++idx)
        field.ptr()[idx] = base + slope * static_cast<double>(idx);
}

void fill_field_constant(tensorium_RG::Field3D<double> &field, double value) {
    const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
    for (size_t idx = 0; idx < total; ++idx)
        field.ptr()[idx] = value;
}

void expect_field_equal(const tensorium_RG::Field3D<double> &lhs,
                        const tensorium_RG::Field3D<double> &rhs, const std::string &label) {
    const size_t total = lhs.st.nx_tot * lhs.st.ny_tot * lhs.st.nz_tot;
    TENSORIUM_TEST_ASSERT(total == rhs.st.nx_tot * rhs.st.ny_tot * rhs.st.nz_tot);
    for (size_t idx = 0; idx < total; ++idx)
        tensorium::tests::expect_near(lhs.ptr()[idx], rhs.ptr()[idx], 1.0e-12, label);
}

void expect_field_domain_equal(const Grid &grid, const tensorium_RG::Field3D<double> &lhs,
                               const tensorium_RG::Field3D<double> &rhs, const std::string &label) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t idx = lhs.idx(i, j, k);
                tensorium::tests::expect_near(lhs.ptr()[idx], rhs.ptr()[idx], 1.0e-12, label);
            }
}

void expect_field_domain_near(const Grid &grid, const tensorium_RG::Field3D<double> &lhs,
                              const tensorium_RG::Field3D<double> &rhs, double tol,
                              const std::string &label) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t idx = lhs.idx(i, j, k);
                tensorium::tests::expect_near(lhs.ptr()[idx], rhs.ptr()[idx], tol, label);
            }
}

void expect_field_bulk_near(const Grid &grid, const tensorium_RG::Field3D<double> &lhs,
                            const tensorium_RG::Field3D<double> &rhs, size_t padding, double tol,
                            const std::string &label) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i0 = std::min(I0 + padding, I1);
    const size_t j0 = std::min(J0 + padding, J1);
    const size_t k0 = std::min(K0 + padding, K1);
    const size_t i1 = (I1 > padding) ? I1 - padding : I1;
    const size_t j1 = (J1 > padding) ? J1 - padding : J1;
    const size_t k1 = (K1 > padding) ? K1 - padding : K1;
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t idx = lhs.idx(i, j, k);
                tensorium::tests::expect_near(lhs.ptr()[idx], rhs.ptr()[idx], tol, label);
            }
}

void expect_field_halo_constant(const Grid &grid, const tensorium_RG::Field3D<double> &field,
                                double value, const std::string &label) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = 0; i < grid.alpha.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.alpha.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.alpha.st.nz_tot; ++k) {
                const bool in_domain = (i >= I0 && i < I1 && j >= J0 && j < J1 && k >= K0 && k < K1);
                if (in_domain)
                    continue;
                const size_t idx = field.idx(i, j, k);
                tensorium::tests::expect_near(field.ptr()[idx], value, 1.0e-12, label);
            }
}

void seed_cuda_backend_case(Grid &grid) {
    tensorium_RG::init::minkowski(grid, 0.0);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const double denom_i = (I1 > I0 + 1) ? static_cast<double>(I1 - I0 - 1) : 1.0;
    const double denom_j = (J1 > J0 + 1) ? static_cast<double>(J1 - J0 - 1) : 1.0;
    const double denom_k = (K1 > K0 + 1) ? static_cast<double>(K1 - K0 - 1) : 1.0;

    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const double u = static_cast<double>(i - I0) / denom_i;
                const double v = static_cast<double>(j - J0) / denom_j;
                const double w = static_cast<double>(k - K0) / denom_k;
                const double phase = 1.3 * u - 0.9 * v + 0.7 * w;
                const double bend = 0.5 * u * u - 0.25 * v * w;
                const size_t idx = grid.alpha.idx(i, j, k);

                grid.alpha.ptr()[idx] = 1.0 + 2.0e-2 * std::sin(phase);
                grid.chi.ptr()[idx] = 1.0 + 1.5e-2 * std::cos(phase + 0.25);
                grid.K.ptr()[idx] = 1.0e-3 * std::sin(2.0 * phase);
                grid.Theta.ptr()[idx] = -5.0e-4 * std::cos(1.5 * phase);

                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[idx] = 1.0e-3 * (c + 1) * std::sin(phase + 0.2 * c);
                    grid.B[c].ptr()[idx] = -2.5e-4 * (c + 1) * std::cos(phase - 0.15 * c);
                    grid.tildeGamma[c].ptr()[idx] =
                        4.0e-4 * (c + 1) * std::sin(phase + bend + 0.1 * c);
                    grid.Z[c].ptr()[idx] = -2.0e-4 * (c + 1) * std::cos(phase - bend + 0.05 * c);
                }

                grid.gamma_tilde[tensorium_RG::XX].ptr()[idx] = 1.0 + 7.0e-4 * std::sin(phase);
                grid.gamma_tilde[tensorium_RG::YY].ptr()[idx] =
                    1.0 - 5.0e-4 * std::cos(phase + 0.3);
                grid.gamma_tilde[tensorium_RG::ZZ].ptr()[idx] =
                    1.0 + 3.5e-4 * std::sin(phase - 0.2);
                grid.gamma_tilde[tensorium_RG::XY].ptr()[idx] = 1.5e-4 * std::sin(phase + bend);
                grid.gamma_tilde[tensorium_RG::XZ].ptr()[idx] = -1.0e-4 * std::cos(phase - bend);
                grid.gamma_tilde[tensorium_RG::YZ].ptr()[idx] = 1.2e-4 * std::sin(phase + 0.4);

                for (int s = 0; s < 6; ++s)
                    grid.A_tilde[s].ptr()[idx] = 1.0e-4 * (s + 1) * std::cos(phase + 0.1 * s);
            }
}

void copy_grid_state(const Grid &src, Grid &dst) {
    dst.x0 = src.x0;
    dst.y0 = src.y0;
    dst.z0 = src.z0;

    const size_t total = src.alpha.st.nx_tot * src.alpha.st.ny_tot * src.alpha.st.nz_tot;
    auto copy_field = [&](const tensorium_RG::Field3D<double> &from, tensorium_RG::Field3D<double> &to) {
        for (size_t idx = 0; idx < total; ++idx)
            to.ptr()[idx] = from.ptr()[idx];
    };

    copy_field(src.alpha, dst.alpha);
    copy_field(src.chi, dst.chi);
    copy_field(src.K, dst.K);
    copy_field(src.Theta, dst.Theta);
    for (int c = 0; c < 3; ++c) {
        copy_field(src.beta[c], dst.beta[c]);
        copy_field(src.B[c], dst.B[c]);
        copy_field(src.tildeGamma[c], dst.tildeGamma[c]);
        copy_field(src.Z[c], dst.Z[c]);
    }
    for (int s = 0; s < 6; ++s) {
        copy_field(src.gamma_tilde[s], dst.gamma_tilde[s]);
        copy_field(src.A_tilde[s], dst.A_tilde[s]);
    }
}

void copy_rhs_state(const tensorium_RG::bssn::BSSNRHSWorkspace<double> &src,
                    tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> &dst) {
    dst.alpha.copy_from_host(src.alpha);
    dst.chi.copy_from_host(src.chi);
    dst.K.copy_from_host(src.K);
    dst.Theta.copy_from_host(src.Theta);
    for (int c = 0; c < 3; ++c) {
        dst.beta[c].copy_from_host(src.beta[c]);
        dst.B[c].copy_from_host(src.B[c]);
        dst.tildeGamma[c].copy_from_host(src.tildeGamma[c]);
        dst.Z[c].copy_from_host(src.Z[c]);
    }
    for (int s = 0; s < 6; ++s) {
        dst.gamma_tilde[s].copy_from_host(src.gamma_tilde[s]);
        dst.A_tilde[s].copy_from_host(src.A_tilde[s]);
    }
}

tensorium_RG::bssn::cuda::GaugeRHSCudaConfig<double>
make_cuda_gauge_rhs_config(const tensorium_RG::bssn::GaugeParameters<double> &gauge) {
    tensorium_RG::bssn::cuda::GaugeRHSCudaConfig<double> cfg;
    cfg.ko_sigma = tensorium_RG::bssn::scaled_ko_sigma(gauge.ko_sigma);
    cfg.ko_boundary_floor = tensorium_RG::bssn::current_ko_boundary_floor();
    cfg.beta_B_coeff = gauge.beta_B_coeff;
    cfg.eta_coeff = gauge.effective_eta();
    cfg.lapse_harmonicf = gauge.lapse_harmonicf;
    cfg.lapse_harmonic = gauge.lapse_harmonic;
    cfg.lapse_oplog = gauge.lapse_oplog;
    cfg.lapse_advect = gauge.lapse_advect;
    cfg.shift_advect = gauge.shift_advect;
    cfg.slow_start_lapse_factor = gauge.slow_start_lapse_factor();
    cfg.spatial_order = tensorium_RG::fd::max_spatial_derivative_order();
    cfg.ko_boundary_width = tensorium_RG::bssn::current_ko_boundary_width();
    cfg.use_theta_in_lapse = gauge.use_theta_in_lapse;
    cfg.use_shift_advection = gauge.use_shift_advection;
    return cfg;
}

tensorium_RG::bssn::cuda::ScalarRHSCudaConfig<double>
make_cuda_scalar_rhs_config(const tensorium_RG::bssn::GaugeParameters<double> &gauge) {
    tensorium_RG::bssn::cuda::ScalarRHSCudaConfig<double> cfg;
    cfg.ko_sigma = tensorium_RG::bssn::scaled_ko_sigma(gauge.ko_sigma);
    cfg.ko_boundary_floor = tensorium_RG::bssn::current_ko_boundary_floor();
    cfg.kappa1 = gauge.kappa1;
    cfg.kappa2 = gauge.kappa2;
    cfg.chi_div_floor = gauge.chi_div_floor;
    cfg.spatial_order = tensorium_RG::fd::max_spatial_derivative_order();
    cfg.ko_boundary_width = tensorium_RG::bssn::current_ko_boundary_width();
    cfg.covariant_z4 = gauge.covariant_z4;
    return cfg;
}

void apply_expected_stage_update(Grid &grid, const Grid &stage,
                                 const tensorium_RG::bssn::BSSNRHSWorkspace<double> &rhs,
                                 double gam0, double gam1, double beta_dt) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[idx] =
                    gam0 * grid.alpha.ptr()[idx] + gam1 * stage.alpha.ptr()[idx] +
                    beta_dt * rhs.alpha.ptr()[idx];
                grid.chi.ptr()[idx] =
                    gam0 * grid.chi.ptr()[idx] + gam1 * stage.chi.ptr()[idx] +
                    beta_dt * rhs.chi.ptr()[idx];
                grid.K.ptr()[idx] = gam0 * grid.K.ptr()[idx] + gam1 * stage.K.ptr()[idx] +
                                    beta_dt * rhs.K.ptr()[idx];
                grid.Theta.ptr()[idx] =
                    gam0 * grid.Theta.ptr()[idx] + gam1 * stage.Theta.ptr()[idx] +
                    beta_dt * rhs.Theta.ptr()[idx];
                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[idx] = gam0 * grid.beta[c].ptr()[idx] +
                                              gam1 * stage.beta[c].ptr()[idx] +
                                              beta_dt * rhs.beta[c].ptr()[idx];
                    grid.B[c].ptr()[idx] = gam0 * grid.B[c].ptr()[idx] +
                                           gam1 * stage.B[c].ptr()[idx] +
                                           beta_dt * rhs.B[c].ptr()[idx];
                    grid.tildeGamma[c].ptr()[idx] = gam0 * grid.tildeGamma[c].ptr()[idx] +
                                                    gam1 * stage.tildeGamma[c].ptr()[idx] +
                                                    beta_dt * rhs.tildeGamma[c].ptr()[idx];
                    grid.Z[c].ptr()[idx] = gam0 * grid.Z[c].ptr()[idx] +
                                           gam1 * stage.Z[c].ptr()[idx] +
                                           beta_dt * rhs.Z[c].ptr()[idx];
                }
                for (int s = 0; s < 6; ++s) {
                    grid.gamma_tilde[s].ptr()[idx] =
                        gam0 * grid.gamma_tilde[s].ptr()[idx] +
                        gam1 * stage.gamma_tilde[s].ptr()[idx] + beta_dt * rhs.gamma_tilde[s].ptr()[idx];
                    grid.A_tilde[s].ptr()[idx] = gam0 * grid.A_tilde[s].ptr()[idx] +
                                                 gam1 * stage.A_tilde[s].ptr()[idx] +
                                                 beta_dt * rhs.A_tilde[s].ptr()[idx];
                }
            }
}

double smooth_floor_reference(double value, double floor) {
    const double delta = 1.0e-10;
    return 0.5 * (value + floor + std::sqrt((value - floor) * (value - floor) + delta));
}

void apply_expected_floor(tensorium_RG::Field3D<double> &field, double floor) {
    const size_t total = field.st.nx_tot * field.st.ny_tot * field.st.nz_tot;
    for (size_t idx = 0; idx < total; ++idx)
        field.ptr()[idx] = smooth_floor_reference(field.ptr()[idx], floor);
}

} // namespace

REGISTER_TEST("bssn.evolution.rhs_halo_sentinel", "RHS halos untouched contract", []() {
    const size_t padding = 4;
    Grid        grid(32, 32, 32, padding, 0.25, 0.25, 0.25);
    tensorium_RG::init::minkowski(grid, 0.0);

    Stepper stepper(grid, padding);
    stepper.set_rhs_prep_callback([](auto &rhs) { poison_workspace(rhs); });

    for (size_t step = 0; step < 2; ++step) {
        const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, 0.25, padding);
        stepper.step(grid, dt, step);
        expect_grid_interior_finite(grid, padding);
    }
});

REGISTER_TEST("bssn.evolution.grid_view_matches_grid_layout",
              "Grid views expose the same raw layout metadata as the host grid", []() {
    Grid grid(10, 8, 6, 4, 0.25, 0.5, 0.75);
    grid.x0 = -1.0;
    grid.y0 = 2.0;
    grid.z0 = 3.5;

    const auto view = tensorium_RG::bssn::make_view(grid);

    TENSORIUM_TEST_ASSERT(view.alpha.ptr() == grid.alpha.ptr());
    TENSORIUM_TEST_ASSERT(view.gamma_tilde[tensorium_RG::XX].ptr() ==
                          grid.gamma_tilde[tensorium_RG::XX].ptr());
    TENSORIUM_TEST_ASSERT(view.st.sx == grid.st.sx);
    TENSORIUM_TEST_ASSERT(view.st.sy == grid.st.sy);
    TENSORIUM_TEST_ASSERT(view.st.sz == grid.st.sz);

    size_t i0, i1, j0, j1, k0, k1;
    view.domain_bounds(i0, i1, j0, j1, k0, k1);
    TENSORIUM_TEST_ASSERT(i0 == grid.dims.ng);
    TENSORIUM_TEST_ASSERT(i1 == grid.dims.ng + grid.dims.nx);
    TENSORIUM_TEST_ASSERT(j0 == grid.dims.ng);
    TENSORIUM_TEST_ASSERT(j1 == grid.dims.ng + grid.dims.ny);
    TENSORIUM_TEST_ASSERT(k0 == grid.dims.ng);
    TENSORIUM_TEST_ASSERT(k1 == grid.dims.ng + grid.dims.nz);

    double x = 0.0, y = 0.0, z = 0.0;
    view.coords(i0 + 2, j0 + 1, k0 + 3, x, y, z);
    TENSORIUM_TEST_ASSERT(std::abs(x - (-0.5)) < 1.0e-12);
    TENSORIUM_TEST_ASSERT(std::abs(y - 2.5) < 1.0e-12);
    TENSORIUM_TEST_ASSERT(std::abs(z - 5.75) < 1.0e-12);
});

REGISTER_TEST("bssn.evolution.cuda_grid_device_roundtrip",
              "BSSNGridDevice preserves the host grid through a CUDA roundtrip", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    Grid host(10, 8, 6, 4, 0.125, 0.25, 0.5);
    host.x0 = -1.5;
    host.y0 = 0.75;
    host.z0 = 2.25;

    fill_field_pattern(host.alpha, 1.0, 1.0e-3);
    fill_field_pattern(host.chi, 0.5, -2.0e-3);
    fill_field_pattern(host.K, -0.25, 3.0e-3);
    fill_field_pattern(host.Theta, 0.125, -4.0e-3);
    for (int c = 0; c < 3; ++c) {
        fill_field_pattern(host.beta[c], 0.1 * (c + 1), 1.0e-3 * (c + 2));
        fill_field_pattern(host.B[c], -0.2 * (c + 1), -1.5e-3 * (c + 2));
        fill_field_pattern(host.tildeGamma[c], 0.3 * (c + 1), 2.0e-3 * (c + 3));
        fill_field_pattern(host.Z[c], -0.4 * (c + 1), -2.5e-3 * (c + 3));
    }
    for (int s = 0; s < 6; ++s) {
        fill_field_pattern(host.gamma_tilde[s], 1.0 + 0.1 * s, 5.0e-4 * (s + 1));
        fill_field_pattern(host.gamma_tilde_inv[s], 0.8 + 0.2 * s, -6.0e-4 * (s + 1));
        fill_field_pattern(host.A_tilde[s], -0.6 - 0.1 * s, 7.0e-4 * (s + 1));
        fill_field_pattern(host.Ricci[s], 0.9 - 0.05 * s, -8.0e-4 * (s + 1));
    }

    tensorium_RG::bssn::BSSNGridDevice<double> device(host);
    device.copy_from_host(host);

    const auto device_view = device.view();
    TENSORIUM_TEST_ASSERT(device_view.dims.nx == host.dims.nx);
    TENSORIUM_TEST_ASSERT(device_view.st.sx == host.st.sx);
    TENSORIUM_TEST_ASSERT(device_view.alpha.ptr() != nullptr);

    Grid roundtrip(host.dims.nx, host.dims.ny, host.dims.nz, host.dims.ng, host.dx, host.dy,
                   host.dz);
    device.copy_to_host(roundtrip);

    tensorium::tests::expect_near(roundtrip.x0, host.x0, 1.0e-12, "cuda_grid_device_roundtrip x0");
    tensorium::tests::expect_near(roundtrip.y0, host.y0, 1.0e-12, "cuda_grid_device_roundtrip y0");
    tensorium::tests::expect_near(roundtrip.z0, host.z0, 1.0e-12, "cuda_grid_device_roundtrip z0");

    expect_field_equal(roundtrip.alpha, host.alpha, "cuda_grid_device_roundtrip alpha");
    expect_field_equal(roundtrip.chi, host.chi, "cuda_grid_device_roundtrip chi");
    expect_field_equal(roundtrip.K, host.K, "cuda_grid_device_roundtrip K");
    expect_field_equal(roundtrip.Theta, host.Theta, "cuda_grid_device_roundtrip Theta");
    for (int c = 0; c < 3; ++c) {
        expect_field_equal(roundtrip.beta[c], host.beta[c], "cuda_grid_device_roundtrip beta");
        expect_field_equal(roundtrip.B[c], host.B[c], "cuda_grid_device_roundtrip B");
        expect_field_equal(roundtrip.tildeGamma[c], host.tildeGamma[c],
                           "cuda_grid_device_roundtrip tildeGamma");
        expect_field_equal(roundtrip.Z[c], host.Z[c], "cuda_grid_device_roundtrip Z");
    }
    for (int s = 0; s < 6; ++s) {
        expect_field_equal(roundtrip.gamma_tilde[s], host.gamma_tilde[s],
                           "cuda_grid_device_roundtrip gamma_tilde");
        expect_field_equal(roundtrip.gamma_tilde_inv[s], host.gamma_tilde_inv[s],
                           "cuda_grid_device_roundtrip gamma_tilde_inv");
        expect_field_equal(roundtrip.A_tilde[s], host.A_tilde[s],
                           "cuda_grid_device_roundtrip A_tilde");
        expect_field_equal(roundtrip.Ricci[s], host.Ricci[s], "cuda_grid_device_roundtrip Ricci");
    }
#endif
});

REGISTER_TEST("bssn.evolution.cuda_stage_copy_kernel_roundtrip",
              "CUDA stage copy kernel updates the physical domain while leaving halos untouched",
              []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    constexpr double sentinel = -99.0;
    Grid             src(10, 8, 6, 4, 0.125, 0.25, 0.5);
    src.x0 = -1.5;
    src.y0 = 0.75;
    src.z0 = 2.25;

    fill_field_pattern(src.alpha, 1.0, 1.0e-3);
    fill_field_pattern(src.chi, 0.5, -2.0e-3);
    fill_field_pattern(src.K, -0.25, 3.0e-3);
    fill_field_pattern(src.Theta, 0.125, -4.0e-3);
    for (int c = 0; c < 3; ++c) {
        fill_field_pattern(src.beta[c], 0.1 * (c + 1), 1.0e-3 * (c + 2));
        fill_field_pattern(src.B[c], -0.2 * (c + 1), -1.5e-3 * (c + 2));
        fill_field_pattern(src.tildeGamma[c], 0.3 * (c + 1), 2.0e-3 * (c + 3));
        fill_field_pattern(src.Z[c], -0.4 * (c + 1), -2.5e-3 * (c + 3));
    }
    for (int s = 0; s < 6; ++s) {
        fill_field_pattern(src.gamma_tilde[s], 1.0 + 0.1 * s, 5.0e-4 * (s + 1));
        fill_field_pattern(src.A_tilde[s], -0.6 - 0.1 * s, 7.0e-4 * (s + 1));
    }

    Grid dst(src.dims.nx, src.dims.ny, src.dims.nz, src.dims.ng, src.dx, src.dy, src.dz);
    fill_field_constant(dst.alpha, sentinel);
    fill_field_constant(dst.chi, sentinel);
    fill_field_constant(dst.K, sentinel);
    fill_field_constant(dst.Theta, sentinel);
    for (int c = 0; c < 3; ++c) {
        fill_field_constant(dst.beta[c], sentinel);
        fill_field_constant(dst.B[c], sentinel);
        fill_field_constant(dst.tildeGamma[c], sentinel);
        fill_field_constant(dst.Z[c], sentinel);
    }
    for (int s = 0; s < 6; ++s) {
        fill_field_constant(dst.gamma_tilde[s], sentinel);
        fill_field_constant(dst.gamma_tilde_inv[s], sentinel);
        fill_field_constant(dst.A_tilde[s], sentinel);
        fill_field_constant(dst.Ricci[s], sentinel);
    }

    tensorium_RG::bssn::BSSNGridDevice<double> src_device(src);
    tensorium_RG::bssn::BSSNGridDevice<double> dst_device(dst);
    src_device.copy_from_host(src);
    dst_device.copy_from_host(dst);
    dst_device.x0 = src.x0;
    dst_device.y0 = src.y0;
    dst_device.z0 = src.z0;

    const auto src_view = static_cast<const tensorium_RG::bssn::BSSNGridDevice<double> &>(src_device).view();
    tensorium_RG::bssn::cuda::copy_stage_reference(src_view, dst_device.view(), 0);
    tensorium::cuda::device_synchronize();

    Grid roundtrip(src.dims.nx, src.dims.ny, src.dims.nz, src.dims.ng, src.dx, src.dy, src.dz);
    dst_device.copy_to_host(roundtrip);

    expect_field_domain_equal(roundtrip, roundtrip.alpha, src.alpha, "cuda_stage_copy alpha");
    expect_field_domain_equal(roundtrip, roundtrip.chi, src.chi, "cuda_stage_copy chi");
    expect_field_domain_equal(roundtrip, roundtrip.K, src.K, "cuda_stage_copy K");
    expect_field_domain_equal(roundtrip, roundtrip.Theta, src.Theta, "cuda_stage_copy Theta");
    for (int c = 0; c < 3; ++c) {
        expect_field_domain_equal(roundtrip, roundtrip.beta[c], src.beta[c], "cuda_stage_copy beta");
        expect_field_domain_equal(roundtrip, roundtrip.B[c], src.B[c], "cuda_stage_copy B");
        expect_field_domain_equal(roundtrip, roundtrip.tildeGamma[c], src.tildeGamma[c],
                                  "cuda_stage_copy tildeGamma");
        expect_field_domain_equal(roundtrip, roundtrip.Z[c], src.Z[c], "cuda_stage_copy Z");
    }
    for (int s = 0; s < 6; ++s) {
        expect_field_domain_equal(roundtrip, roundtrip.gamma_tilde[s], src.gamma_tilde[s],
                                  "cuda_stage_copy gamma_tilde");
        expect_field_domain_equal(roundtrip, roundtrip.A_tilde[s], src.A_tilde[s],
                                  "cuda_stage_copy A_tilde");
    }

    expect_field_halo_constant(roundtrip, roundtrip.alpha, sentinel, "cuda_stage_copy halo alpha");
    expect_field_halo_constant(roundtrip, roundtrip.chi, sentinel, "cuda_stage_copy halo chi");
    expect_field_halo_constant(roundtrip, roundtrip.K, sentinel, "cuda_stage_copy halo K");
    expect_field_halo_constant(roundtrip, roundtrip.Theta, sentinel, "cuda_stage_copy halo Theta");
#endif
});

REGISTER_TEST("bssn.evolution.cuda_explicit_stage_update_kernel_roundtrip",
              "CUDA explicit stage update matches the host RK formula on the physical domain", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    constexpr double sentinel = -77.0;
    constexpr double gam0 = -0.35;
    constexpr double gam1 = 0.9;
    constexpr double beta_dt = 2.5e-2;

    Grid current(10, 8, 6, 4, 0.125, 0.25, 0.5);
    Grid stage(10, 8, 6, 4, 0.125, 0.25, 0.5);
    current.x0 = -1.5;
    current.y0 = 0.75;
    current.z0 = 2.25;
    stage.x0 = current.x0;
    stage.y0 = current.y0;
    stage.z0 = current.z0;

    fill_field_pattern(current.alpha, 1.0, 1.0e-3);
    fill_field_pattern(current.chi, 0.5, -2.0e-3);
    fill_field_pattern(current.K, -0.25, 3.0e-3);
    fill_field_pattern(current.Theta, 0.125, -4.0e-3);
    fill_field_pattern(stage.alpha, -0.2, 1.5e-3);
    fill_field_pattern(stage.chi, 0.9, 7.5e-4);
    fill_field_pattern(stage.K, 0.3, -2.5e-3);
    fill_field_pattern(stage.Theta, -0.15, 3.5e-3);
    for (int c = 0; c < 3; ++c) {
        fill_field_pattern(current.beta[c], 0.1 * (c + 1), 1.0e-3 * (c + 2));
        fill_field_pattern(current.B[c], -0.2 * (c + 1), -1.5e-3 * (c + 2));
        fill_field_pattern(current.tildeGamma[c], 0.3 * (c + 1), 2.0e-3 * (c + 3));
        fill_field_pattern(current.Z[c], -0.4 * (c + 1), -2.5e-3 * (c + 3));

        fill_field_pattern(stage.beta[c], -0.15 * (c + 1), 1.2e-3 * (c + 4));
        fill_field_pattern(stage.B[c], 0.25 * (c + 1), -1.1e-3 * (c + 5));
        fill_field_pattern(stage.tildeGamma[c], -0.35 * (c + 1), 9.0e-4 * (c + 2));
        fill_field_pattern(stage.Z[c], 0.45 * (c + 1), -8.0e-4 * (c + 3));
    }
    for (int s = 0; s < 6; ++s) {
        fill_field_pattern(current.gamma_tilde[s], 1.0 + 0.1 * s, 5.0e-4 * (s + 1));
        fill_field_pattern(current.gamma_tilde_inv[s], sentinel, 0.0);
        fill_field_pattern(current.A_tilde[s], -0.6 - 0.1 * s, 7.0e-4 * (s + 1));
        fill_field_pattern(current.Ricci[s], sentinel, 0.0);

        fill_field_pattern(stage.gamma_tilde[s], -0.9 + 0.12 * s, 6.0e-4 * (s + 1));
        fill_field_pattern(stage.gamma_tilde_inv[s], sentinel, 0.0);
        fill_field_pattern(stage.A_tilde[s], 0.7 - 0.08 * s, -5.0e-4 * (s + 1));
        fill_field_pattern(stage.Ricci[s], sentinel, 0.0);
    }

    tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
    rhs.allocate_like(current);
    fill_field_pattern(rhs.alpha, 0.2, -3.0e-4);
    fill_field_pattern(rhs.chi, -0.3, 4.0e-4);
    fill_field_pattern(rhs.K, 0.4, -5.0e-4);
    fill_field_pattern(rhs.Theta, -0.5, 6.0e-4);
    for (int c = 0; c < 3; ++c) {
        fill_field_pattern(rhs.beta[c], 0.05 * (c + 1), -3.5e-4 * (c + 2));
        fill_field_pattern(rhs.B[c], -0.06 * (c + 1), 4.5e-4 * (c + 2));
        fill_field_pattern(rhs.tildeGamma[c], 0.07 * (c + 1), -5.5e-4 * (c + 3));
        fill_field_pattern(rhs.Z[c], -0.08 * (c + 1), 6.5e-4 * (c + 3));
    }
    for (int s = 0; s < 6; ++s) {
        fill_field_pattern(rhs.gamma_tilde[s], 0.12 * (s + 1), -2.5e-4 * (s + 1));
        fill_field_pattern(rhs.A_tilde[s], -0.14 * (s + 1), 3.5e-4 * (s + 1));
    }

    Grid expected(10, 8, 6, 4, 0.125, 0.25, 0.5);
    copy_grid_state(current, expected);
    apply_expected_stage_update(expected, stage, rhs, gam0, gam1, beta_dt);

    tensorium_RG::bssn::BSSNGridDevice<double> current_device(current);
    tensorium_RG::bssn::BSSNGridDevice<double> stage_device(stage);
    tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> rhs_device;
    rhs_device.allocate_like(current);
    current_device.copy_from_host(current);
    stage_device.copy_from_host(stage);
    copy_rhs_state(rhs, rhs_device);

    const auto stage_view =
        static_cast<const tensorium_RG::bssn::BSSNGridDevice<double> &>(stage_device).view();
    const auto rhs_view =
        static_cast<const tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> &>(rhs_device).view();
    tensorium_RG::bssn::cuda::apply_explicit_stage_update(current_device.view(), stage_view, rhs_view,
                                                          gam0, gam1, beta_dt, 0);
    tensorium::cuda::device_synchronize();

    Grid roundtrip(10, 8, 6, 4, 0.125, 0.25, 0.5);
    current_device.copy_to_host(roundtrip);

    constexpr double tol = 1.0e-12;
    expect_field_domain_near(roundtrip, roundtrip.alpha, expected.alpha, tol,
                             "cuda_stage_update alpha");
    expect_field_domain_near(roundtrip, roundtrip.chi, expected.chi, tol, "cuda_stage_update chi");
    expect_field_domain_near(roundtrip, roundtrip.K, expected.K, tol, "cuda_stage_update K");
    expect_field_domain_near(roundtrip, roundtrip.Theta, expected.Theta, tol,
                             "cuda_stage_update Theta");
    for (int c = 0; c < 3; ++c) {
        expect_field_domain_near(roundtrip, roundtrip.beta[c], expected.beta[c], tol,
                                 "cuda_stage_update beta");
        expect_field_domain_near(roundtrip, roundtrip.B[c], expected.B[c], tol,
                                 "cuda_stage_update B");
        expect_field_domain_near(roundtrip, roundtrip.tildeGamma[c], expected.tildeGamma[c], tol,
                                 "cuda_stage_update tildeGamma");
        expect_field_domain_near(roundtrip, roundtrip.Z[c], expected.Z[c], tol,
                                 "cuda_stage_update Z");
    }
    for (int s = 0; s < 6; ++s) {
        expect_field_domain_near(roundtrip, roundtrip.gamma_tilde[s], expected.gamma_tilde[s], tol,
                                 "cuda_stage_update gamma_tilde");
        expect_field_domain_near(roundtrip, roundtrip.A_tilde[s], expected.A_tilde[s], tol,
                                 "cuda_stage_update A_tilde");
    }

    expect_field_equal(roundtrip.gamma_tilde_inv[tensorium_RG::XX], current.gamma_tilde_inv[tensorium_RG::XX],
                       "cuda_stage_update halo/cache gamma_tilde_inv");
    expect_field_equal(roundtrip.Ricci[tensorium_RG::XX], current.Ricci[tensorium_RG::XX],
                       "cuda_stage_update halo/cache Ricci");
#endif
});

REGISTER_TEST(
    "bssn.evolution.cuda_post_stage_corrections_match_cpu",
    "CUDA algebraic constraints and alpha/chi floors match the host post-stage corrections", []() {
#ifdef TENSORIUM_CUDA
        if (!tensorium::cuda::is_available())
            return;

        constexpr double gam0 = -0.35;
        constexpr double gam1 = 0.9;
        constexpr double beta_dt = 2.5e-2;
        constexpr double alpha_floor = 2.5e-4;
        constexpr double chi_floor = 5.0e-4;
        constexpr double sentinel = -77.0;

        Grid current(10, 8, 6, 4, 0.125, 0.25, 0.5);
        Grid stage(10, 8, 6, 4, 0.125, 0.25, 0.5);
        current.x0 = -1.5;
        current.y0 = 0.75;
        current.z0 = 2.25;
        stage.x0 = current.x0;
        stage.y0 = current.y0;
        stage.z0 = current.z0;

        fill_field_pattern(current.alpha, -6.0e-4, 3.0e-5);
        fill_field_pattern(current.chi, -8.0e-4, 2.0e-5);
        fill_field_pattern(current.K, -0.25, 3.0e-3);
        fill_field_pattern(current.Theta, 0.125, -4.0e-3);
        fill_field_pattern(stage.alpha, 1.0e-4, -2.5e-5);
        fill_field_pattern(stage.chi, 2.0e-4, 1.5e-5);
        fill_field_pattern(stage.K, 0.3, -2.5e-3);
        fill_field_pattern(stage.Theta, -0.15, 3.5e-3);
        for (int c = 0; c < 3; ++c) {
            fill_field_pattern(current.beta[c], 0.1 * (c + 1), 1.0e-3 * (c + 2));
            fill_field_pattern(current.B[c], -0.2 * (c + 1), -1.5e-3 * (c + 2));
            fill_field_pattern(current.tildeGamma[c], 0.3 * (c + 1), 2.0e-3 * (c + 3));
            fill_field_pattern(current.Z[c], -0.4 * (c + 1), -2.5e-3 * (c + 3));

            fill_field_pattern(stage.beta[c], -0.15 * (c + 1), 1.2e-3 * (c + 4));
            fill_field_pattern(stage.B[c], 0.25 * (c + 1), -1.1e-3 * (c + 5));
            fill_field_pattern(stage.tildeGamma[c], -0.35 * (c + 1), 9.0e-4 * (c + 2));
            fill_field_pattern(stage.Z[c], 0.45 * (c + 1), -8.0e-4 * (c + 3));
        }
        for (int s = 0; s < 6; ++s) {
            fill_field_constant(current.gamma_tilde[s], 0.0);
            fill_field_constant(current.gamma_tilde_inv[s], sentinel);
            fill_field_pattern(current.A_tilde[s], -0.6 - 0.1 * s, 7.0e-4 * (s + 1));
            fill_field_constant(current.Ricci[s], sentinel);

            fill_field_constant(stage.gamma_tilde[s], 0.0);
            fill_field_constant(stage.gamma_tilde_inv[s], sentinel);
            fill_field_pattern(stage.A_tilde[s], 0.7 - 0.08 * s, -5.0e-4 * (s + 1));
            fill_field_constant(stage.Ricci[s], sentinel);
        }
        fill_field_pattern(current.gamma_tilde[tensorium_RG::XX], 1.002, 5.0e-6);
        fill_field_pattern(current.gamma_tilde[tensorium_RG::YY], 0.998, -4.0e-6);
        fill_field_pattern(current.gamma_tilde[tensorium_RG::ZZ], 1.001, 3.0e-6);
        fill_field_pattern(current.gamma_tilde[tensorium_RG::XY], 2.0e-4, -1.0e-6);
        fill_field_pattern(current.gamma_tilde[tensorium_RG::XZ], -1.5e-4, 8.0e-7);
        fill_field_pattern(current.gamma_tilde[tensorium_RG::YZ], 1.0e-4, -7.0e-7);

        fill_field_pattern(stage.gamma_tilde[tensorium_RG::XX], 1.001, -3.0e-6);
        fill_field_pattern(stage.gamma_tilde[tensorium_RG::YY], 0.999, 2.5e-6);
        fill_field_pattern(stage.gamma_tilde[tensorium_RG::ZZ], 1.0005, -2.0e-6);
        fill_field_pattern(stage.gamma_tilde[tensorium_RG::XY], -1.8e-4, 7.0e-7);
        fill_field_pattern(stage.gamma_tilde[tensorium_RG::XZ], 1.2e-4, -6.0e-7);
        fill_field_pattern(stage.gamma_tilde[tensorium_RG::YZ], -9.0e-5, 5.0e-7);

        tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
        rhs.allocate_like(current);
        fill_field_pattern(rhs.alpha, 0.2, -3.0e-4);
        fill_field_pattern(rhs.chi, -0.3, 4.0e-4);
        fill_field_pattern(rhs.K, 0.4, -5.0e-4);
        fill_field_pattern(rhs.Theta, -0.5, 6.0e-4);
        for (int c = 0; c < 3; ++c) {
            fill_field_pattern(rhs.beta[c], 0.05 * (c + 1), -3.5e-4 * (c + 2));
            fill_field_pattern(rhs.B[c], -0.06 * (c + 1), 4.5e-4 * (c + 2));
            fill_field_pattern(rhs.tildeGamma[c], 0.07 * (c + 1), -5.5e-4 * (c + 3));
            fill_field_pattern(rhs.Z[c], -0.08 * (c + 1), 6.5e-4 * (c + 3));
        }
        for (int s = 0; s < 6; ++s) {
            fill_field_pattern(rhs.gamma_tilde[s], 0.12 * (s + 1), -2.5e-4 * (s + 1));
            fill_field_pattern(rhs.A_tilde[s], -0.14 * (s + 1), 3.5e-4 * (s + 1));
        }

        Grid expected(10, 8, 6, 4, 0.125, 0.25, 0.5);
        copy_grid_state(current, expected);
        apply_expected_stage_update(expected, stage, rhs, gam0, gam1, beta_dt);
        tensorium_RG::bssn::enforce_algebraic_constraints(expected);
        apply_expected_floor(expected.alpha, alpha_floor);
        apply_expected_floor(expected.chi, chi_floor);

        tensorium_RG::bssn::BSSNGridDevice<double> current_device(current);
        tensorium_RG::bssn::BSSNGridDevice<double> stage_device(stage);
        tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> rhs_device;
        rhs_device.allocate_like(current);
        current_device.copy_from_host(current);
        stage_device.copy_from_host(stage);
        copy_rhs_state(rhs, rhs_device);

        const auto stage_view =
            static_cast<const tensorium_RG::bssn::BSSNGridDevice<double> &>(stage_device).view();
        const auto rhs_view =
            static_cast<const tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> &>(rhs_device)
                .view();
        auto current_view = current_device.view();
        tensorium_RG::bssn::cuda::apply_explicit_stage_update(current_view, stage_view, rhs_view, gam0,
                                                              gam1, beta_dt, 0);
        tensorium_RG::bssn::cuda::enforce_algebraic_constraints(current_view);
        tensorium_RG::bssn::cuda::apply_alpha_floor(current_view, alpha_floor);
        tensorium_RG::bssn::cuda::apply_chi_floor(current_view, chi_floor);
        tensorium::cuda::device_synchronize();

        Grid roundtrip(10, 8, 6, 4, 0.125, 0.25, 0.5);
        current_device.copy_to_host(roundtrip);

        constexpr double tol = 1.0e-10;
        expect_field_domain_near(roundtrip, roundtrip.alpha, expected.alpha, tol,
                                 "cuda_post_stage alpha");
        expect_field_domain_near(roundtrip, roundtrip.chi, expected.chi, tol,
                                 "cuda_post_stage chi");
        for (int s = 0; s < 6; ++s) {
            expect_field_domain_near(roundtrip, roundtrip.gamma_tilde[s], expected.gamma_tilde[s], tol,
                                     "cuda_post_stage gamma_tilde");
            expect_field_domain_near(roundtrip, roundtrip.gamma_tilde_inv[s],
                                     expected.gamma_tilde_inv[s], tol,
                                     "cuda_post_stage gamma_tilde_inv");
            expect_field_domain_near(roundtrip, roundtrip.A_tilde[s], expected.A_tilde[s], tol,
                                     "cuda_post_stage A_tilde");
        }
#endif
    });

REGISTER_TEST("bssn.evolution.cuda_gauge_rhs_kernel_roundtrip",
              "CUDA gauge RHS matches the host alpha/chi/beta/B formulas on the bulk interior", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    const size_t padding = 4;
    Grid         grid(16, 12, 10, padding, 0.25, 0.25, 0.25);
    seed_cuda_backend_case(grid);

    tensorium_RG::bssn::GaugeParameters<double> gauge{};

    tensorium_RG::bssn::BSSNRHSWorkspace<double> cpu_rhs;
    cpu_rhs.allocate_like(grid);
    tensorium_RG::bssn::compute_rhs_Gamma(grid, cpu_rhs.tildeGamma, grid.Z, grid.Theta, gauge, padding);
    tensorium_RG::bssn::compute_rhs_alpha(grid, cpu_rhs.alpha, padding, gauge);
    tensorium_RG::bssn::compute_rhs_chi(grid, cpu_rhs.chi, padding, gauge);
    tensorium_RG::bssn::compute_rhs_beta(grid, cpu_rhs.beta, gauge, padding);
    tensorium_RG::bssn::compute_rhs_B(grid, cpu_rhs.tildeGamma, cpu_rhs.B, gauge, padding);

    tensorium_RG::bssn::BSSNGridDevice<double> current_device(grid);
    tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> rhs_device;
    current_device.copy_from_host(grid);
    rhs_device.allocate_like(grid);
    copy_rhs_state(cpu_rhs, rhs_device);

    const auto grid_view =
        static_cast<const tensorium_RG::bssn::BSSNGridDevice<double> &>(current_device).view();
    tensorium_RG::bssn::cuda::compute_gauge_rhs(grid_view, rhs_device.view(),
                                                make_cuda_gauge_rhs_config(gauge), padding);
    tensorium::cuda::device_synchronize();

    tensorium_RG::bssn::BSSNRHSWorkspace<double> gpu_rhs;
    gpu_rhs.allocate_like(grid);
    rhs_device.alpha.copy_to_host(gpu_rhs.alpha);
    rhs_device.chi.copy_to_host(gpu_rhs.chi);
    for (int c = 0; c < 3; ++c) {
        rhs_device.beta[c].copy_to_host(gpu_rhs.beta[c]);
        rhs_device.B[c].copy_to_host(gpu_rhs.B[c]);
    }

    constexpr double tol = 1.0e-12;
    expect_field_bulk_near(grid, gpu_rhs.alpha, cpu_rhs.alpha, padding, tol, "cuda_gauge_rhs alpha");
    expect_field_bulk_near(grid, gpu_rhs.chi, cpu_rhs.chi, padding, tol, "cuda_gauge_rhs chi");
    for (int c = 0; c < 3; ++c) {
        expect_field_bulk_near(grid, gpu_rhs.beta[c], cpu_rhs.beta[c], padding, tol,
                               "cuda_gauge_rhs beta");
        expect_field_bulk_near(grid, gpu_rhs.B[c], cpu_rhs.B[c], padding, tol,
                               "cuda_gauge_rhs B");
    }
#endif
});

REGISTER_TEST("bssn.evolution.cuda_scalar_rhs_kernel_roundtrip",
              "CUDA K/Theta/Z RHS matches the host formulas on the bulk interior", []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    const size_t padding = 4;
    Grid         grid(16, 12, 10, padding, 0.25, 0.25, 0.25);
    seed_cuda_backend_case(grid);
    tensorium_RG::bssn::enforce_algebraic_constraints(grid);
    tensorium_RG::bssn::compute_ricci_bssn(grid, grid.Ricci, false);

    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    auto z4_trace_cache = tensorium_RG::make_field(grid.alpha.st);

    tensorium_RG::bssn::BSSNRHSWorkspace<double> cpu_rhs;
    cpu_rhs.allocate_like(grid);
    tensorium_RG::bssn::compute_rhs_A_tilde(grid, cpu_rhs.A_tilde, padding, gauge, &z4_trace_cache);
    tensorium_RG::bssn::compute_rhs_K(grid, cpu_rhs.K, padding, gauge);
    tensorium_RG::bssn::compute_rhs_Theta(grid, cpu_rhs.Theta, gauge, padding, &z4_trace_cache);
    tensorium_RG::bssn::compute_rhs_Z(grid, cpu_rhs.Z, gauge, padding);

    tensorium_RG::bssn::BSSNGridDevice<double> current_device(grid);
    tensorium_RG::bssn::BSSNRHSWorkspaceDevice<double> rhs_device;
    tensorium_RG::bssn::DeviceField3D<double> z4_trace_device;
    current_device.copy_from_host(grid);
    rhs_device.allocate_like(grid);
    z4_trace_device.allocate_like(z4_trace_cache);
    copy_rhs_state(cpu_rhs, rhs_device);
    z4_trace_device.copy_from_host(z4_trace_cache);

    const auto grid_view =
        static_cast<const tensorium_RG::bssn::BSSNGridDevice<double> &>(current_device).view();
    tensorium_RG::bssn::cuda::compute_scalar_rhs(
        grid_view, rhs_device.view(),
        static_cast<const tensorium_RG::bssn::DeviceField3D<double> &>(z4_trace_device).view(),
        make_cuda_scalar_rhs_config(gauge),
        padding);
    tensorium::cuda::device_synchronize();

    tensorium_RG::bssn::BSSNRHSWorkspace<double> gpu_rhs;
    gpu_rhs.allocate_like(grid);
    rhs_device.K.copy_to_host(gpu_rhs.K);
    rhs_device.Theta.copy_to_host(gpu_rhs.Theta);
    for (int c = 0; c < 3; ++c)
        rhs_device.Z[c].copy_to_host(gpu_rhs.Z[c]);

    constexpr double tol = 1.0e-12;
    expect_field_bulk_near(grid, gpu_rhs.K, cpu_rhs.K, padding, tol, "cuda_scalar_rhs K");
    expect_field_bulk_near(grid, gpu_rhs.Theta, cpu_rhs.Theta, padding, tol,
                           "cuda_scalar_rhs Theta");
    for (int c = 0; c < 3; ++c)
        expect_field_bulk_near(grid, gpu_rhs.Z[c], cpu_rhs.Z[c], padding, tol,
                               "cuda_scalar_rhs Z");
#endif
});

REGISTER_TEST("bssn.evolution.cuda_backend_matches_cpu_step",
              "BSSNRKStepper CUDA backend follows the CPU stage flow on a perturbed gauge state",
              []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    const size_t padding = 4;
    Grid         cpu_grid(16, 12, 10, padding, 0.25, 0.25, 0.25);
    seed_cuda_backend_case(cpu_grid);
    Grid gpu_grid(16, 12, 10, padding, 0.25, 0.25, 0.25);
    copy_grid_state(cpu_grid, gpu_grid);

    tensorium::backend::Options backend;
    backend.backend = tensorium::backend::Kind::CUDA;

    Stepper cpu_stepper(cpu_grid, padding);
    Stepper gpu_stepper(gpu_grid, backend, padding);

    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    gauge.alpha_floor = 1.0e-6;
    gauge.chi_floor = 1.0e-6;
    cpu_stepper.set_gauge_parameters(gauge);
    gpu_stepper.set_gauge_parameters(gauge);

    const double dt = 1.0e-3;
    cpu_stepper.step(cpu_grid, dt, 0);
    gpu_stepper.step(gpu_grid, dt, 0);

    expect_grid_interior_finite(cpu_grid, padding);
    expect_grid_interior_finite(gpu_grid, padding);

    constexpr double tol = 1.0e-6;
    expect_field_domain_near(gpu_grid, gpu_grid.alpha, cpu_grid.alpha, tol, "cuda_backend alpha");
    expect_field_domain_near(gpu_grid, gpu_grid.chi, cpu_grid.chi, tol, "cuda_backend chi");
    expect_field_domain_near(gpu_grid, gpu_grid.K, cpu_grid.K, tol, "cuda_backend K");
    expect_field_domain_near(gpu_grid, gpu_grid.Theta, cpu_grid.Theta, tol, "cuda_backend Theta");
    for (int c = 0; c < 3; ++c) {
        expect_field_domain_near(gpu_grid, gpu_grid.beta[c], cpu_grid.beta[c], tol,
                                 "cuda_backend beta");
        expect_field_domain_near(gpu_grid, gpu_grid.B[c], cpu_grid.B[c], tol, "cuda_backend B");
        expect_field_domain_near(gpu_grid, gpu_grid.tildeGamma[c], cpu_grid.tildeGamma[c], tol,
                                 "cuda_backend tildeGamma");
        expect_field_domain_near(gpu_grid, gpu_grid.Z[c], cpu_grid.Z[c], tol, "cuda_backend Z");
    }
    for (int s = 0; s < 6; ++s) {
        expect_field_domain_near(gpu_grid, gpu_grid.gamma_tilde[s], cpu_grid.gamma_tilde[s], tol,
                                 "cuda_backend gamma_tilde");
        expect_field_domain_near(gpu_grid, gpu_grid.A_tilde[s], cpu_grid.A_tilde[s], tol,
                                 "cuda_backend A_tilde");
    }
#endif
});

REGISTER_TEST("bssn.evolution.cuda_backend_matches_cpu_radiative_rhs",
              "BSSNRKStepper CUDA backend matches CPU with Sommerfeld RHS and sponge enabled",
              []() {
#ifdef TENSORIUM_CUDA
    if (!tensorium::cuda::is_available())
        return;

    auto restore_boundary = []() {
        tensorium_RG::bssn::BoundaryRadiative::characteristic_speed = 1.0;
        tensorium_RG::bssn::BoundaryRadiative::characteristic_dt = 0.0;
        tensorium_RG::bssn::BoundaryRadiative::gauge_characteristic_speed = 1.0;
        tensorium_RG::bssn::BoundaryRadiative::z4c_characteristic_speed = 1.0;
        tensorium_RG::bssn::BoundaryRadiative::khat_characteristic_speed = std::sqrt(2.0);
        tensorium_RG::bssn::BoundaryRadiative::rhs_collar_width = 4;
        for (int axis = 0; axis < 3; ++axis)
            for (int side = 0; side < 2; ++side) {
                tensorium_RG::bssn::BoundaryRadiative::rhs_sommerfeld_face[axis][side] = true;
                tensorium_RG::bssn::BoundaryRadiative::reflective_face[axis][side] = false;
                tensorium_RG::bssn::BoundaryRadiative::active_face[axis][side] = true;
            }
        tensorium_RG::bssn::BoundaryRadiative::sponge_config = {};
    };
    restore_boundary();

    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.1, 0.9, 1.3);
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_collar_width(3);
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(true, true, true, false, true,
                                                                    true);
    tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(false, false, false, false, true,
                                                                false);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);
    tensorium_RG::bssn::BoundaryRadiative::set_sponge(true, 5, 0.6, 2.5);

    const size_t padding = 4;
    Grid         cpu_grid(18, 16, 14, padding, 0.2, 0.2, 0.2);
    seed_cuda_backend_case(cpu_grid);
    Grid gpu_grid(18, 16, 14, padding, 0.2, 0.2, 0.2);
    copy_grid_state(cpu_grid, gpu_grid);

    tensorium::backend::Options backend;
    backend.backend = tensorium::backend::Kind::CUDA;

    Stepper cpu_stepper(cpu_grid, padding);
    Stepper gpu_stepper(gpu_grid, backend, padding);

    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    gauge.alpha_floor = 1.0e-6;
    gauge.chi_floor = 1.0e-6;
    gauge.apply_rhs_sommerfeld = true;
    cpu_stepper.set_gauge_parameters(gauge);
    gpu_stepper.set_gauge_parameters(gauge);

    const double dt = 7.5e-4;
    cpu_stepper.step(cpu_grid, dt, 0);
    gpu_stepper.step(gpu_grid, dt, 0);

    expect_grid_interior_finite(cpu_grid, padding);
    expect_grid_interior_finite(gpu_grid, padding);

    constexpr double tol = 5.0e-5;
    expect_field_domain_near(gpu_grid, gpu_grid.alpha, cpu_grid.alpha, tol,
                             "cuda_backend_radiative alpha");
    expect_field_domain_near(gpu_grid, gpu_grid.chi, cpu_grid.chi, tol,
                             "cuda_backend_radiative chi");
    expect_field_domain_near(gpu_grid, gpu_grid.K, cpu_grid.K, tol, "cuda_backend_radiative K");
    expect_field_domain_near(gpu_grid, gpu_grid.Theta, cpu_grid.Theta, tol,
                             "cuda_backend_radiative Theta");
    for (int c = 0; c < 3; ++c) {
        expect_field_domain_near(gpu_grid, gpu_grid.beta[c], cpu_grid.beta[c], tol,
                                 "cuda_backend_radiative beta");
        expect_field_domain_near(gpu_grid, gpu_grid.B[c], cpu_grid.B[c], tol,
                                 "cuda_backend_radiative B");
        expect_field_domain_near(gpu_grid, gpu_grid.tildeGamma[c], cpu_grid.tildeGamma[c], tol,
                                 "cuda_backend_radiative tildeGamma");
        expect_field_domain_near(gpu_grid, gpu_grid.Z[c], cpu_grid.Z[c], tol,
                                 "cuda_backend_radiative Z");
    }
    for (int s = 0; s < 6; ++s) {
        expect_field_domain_near(gpu_grid, gpu_grid.gamma_tilde[s], cpu_grid.gamma_tilde[s], tol,
                                 "cuda_backend_radiative gamma_tilde");
        expect_field_domain_near(gpu_grid, gpu_grid.A_tilde[s], cpu_grid.A_tilde[s], tol,
                                 "cuda_backend_radiative A_tilde");
    }

    restore_boundary();
#endif
});

REGISTER_TEST("bssn.evolution.radiative_boundary_shells_are_evolved",
              "Radiative RK4 keeps the first physical layers controlled without ghost blow-up", []() {
    const size_t padding = 4;
    Grid         grid(24, 20, 16, padding, 0.25, 0.25, 0.25);
    tensorium_RG::init::minkowski(grid, 0.0);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[idx] = 1.0;
                grid.chi.ptr()[idx] = 1.0;
                grid.K.ptr()[idx] = 0.1;
                grid.Theta.ptr()[idx] = 0.0;
            }

    Stepper stepper(grid, padding);
    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    gauge.apply_rhs_sommerfeld = true;
    stepper.set_gauge_parameters(gauge);

    const size_t j = J0 + (J1 - J0) / 2;
    const size_t k = K0 + (K1 - K0) / 2;
    const size_t idx_face = grid.alpha.idx(I0, j, k);
    const size_t idx_shell3 = grid.alpha.idx(I0 + 3, j, k);
    const double alpha_face_before = grid.alpha.ptr()[idx_face];
    const double alpha_shell3_before = grid.alpha.ptr()[idx_shell3];

    stepper.step(grid, 1.0e-3, 0);

    TENSORIUM_TEST_ASSERT(std::isfinite(grid.alpha.ptr()[idx_face]));
    TENSORIUM_TEST_ASSERT(std::isfinite(grid.alpha.ptr()[idx_shell3]));
    tensorium::tests::expect_le(grid.alpha.ptr()[idx_face], alpha_face_before + 1.0e-12,
                                "Radiative surface cell stays bounded by the asymptotic state");
    tensorium::tests::expect_le(grid.alpha.ptr()[idx_shell3], alpha_shell3_before + 1.0e-12,
                                "Inner collar cell stays bounded by the asymptotic state");
});

REGISTER_TEST("bssn.evolution.z4c_rhs_boundary_operator",
              "Z4c boundary helper applies face-normal outgoing modes on the radiative surface",
              []() {
    const size_t padding = 4;
    Grid         grid(16, 14, 12, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    const size_t nx_tot = grid.alpha.st.nx_tot;
    const size_t ny_tot = grid.alpha.st.ny_tot;
    const size_t nz_tot = grid.alpha.st.nz_tot;

    auto fill_linear = [&](tensorium_RG::Field3D<double> &field, double c0, double cx, double cy,
                           double cz) {
        for (size_t i = 0; i < nx_tot; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k) {
                    double x, y, z;
                    grid.coords(i, j, k, x, y, z);
                    field.ptr()[field.idx(i, j, k)] = c0 + cx * x + cy * y + cz * z;
                }
    };

    fill_linear(grid.alpha, 1.0, 0.02, -0.01, 0.03);
    fill_linear(grid.chi, 1.0, -0.01, 0.015, -0.02);
    fill_linear(grid.beta[0], 0.3, 0.01, 0.02, -0.01);
    fill_linear(grid.B[0], -0.2, 0.03, -0.01, 0.02);
    fill_linear(grid.gamma_tilde[tensorium_RG::XX], 1.0, 0.02, 0.01, -0.01);
    fill_linear(grid.Theta, 0.2, 0.04, -0.03, 0.01);
    fill_linear(grid.tildeGamma[0], -0.1, 0.03, 0.01, -0.02);
    fill_linear(grid.tildeGamma[1], 0.05, -0.02, 0.04, 0.01);
    fill_linear(grid.tildeGamma[2], 0.08, 0.01, -0.03, 0.05);
    for (int s = 0; s < 6; ++s)
        fill_linear(grid.A_tilde[s], 0.03 * double(s + 1), 0.01, -0.015, 0.02);

    tensorium_RG::Field3D<double> khat_field = tensorium_RG::make_field(grid.alpha.st);
    fill_linear(khat_field, -0.15, 0.06, 0.02, -0.04);
    for (size_t i = 0; i < nx_tot; ++i)
        for (size_t j = 0; j < ny_tot; ++j)
            for (size_t k = 0; k < nz_tot; ++k) {
                const size_t idx = grid.K.idx(i, j, k);
                grid.K.ptr()[idx] = khat_field.ptr()[idx] + 2.0 * grid.Theta.ptr()[idx];
            }

    tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
    rhs.allocate_like(grid);
    poison_workspace(rhs);
    poison_field(rhs.Theta);
    for (int c = 0; c < 3; ++c)
        poison_field(rhs.Z[c]);

    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));

    tensorium_RG::bssn::RHSBoundaryFaceMask mask{};
    mask.face[0][0] = true;
    mask.face[0][1] = false;
    mask.face[1][0] = false;
    mask.face[1][1] = false;
    mask.face[2][0] = false;
    mask.face[2][1] = false;

    tensorium_RG::bssn::apply_z4c_rhs_boundary(grid, rhs, mask, 1);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i = I0;
    const size_t j = J0 + (J1 - J0) / 2;
    const size_t k = K0 + (K1 - K0) / 2;
    const size_t idx = grid.alpha.idx(i, j, k);

    double x, y, z;
    grid.coords(i, j, k, x, y, z);
    const double r = std::sqrt(x * x + y * y + z * z);
    const double sqrt2 = std::sqrt(2.0);

    auto deriv_axis = [&](const tensorium_RG::Field3D<double> &field, ptrdiff_t stride, size_t pos,
                          size_t lo, size_t hi, double inv_h, double inv_2h) {
        const double *p = field.ptr() + idx;
        if (pos == lo && lo + 2 < hi)
            return (-1.5 * p[0] + 2.0 * p[stride] - 0.5 * p[2 * stride]) * inv_h;
        if (pos + 1 == hi && lo + 2 < hi)
            return (1.5 * p[0] - 2.0 * p[-stride] + 0.5 * p[-2 * stride]) * inv_h;
        return (p[stride] - p[-stride]) * inv_2h;
    };

    auto expected_rhs = [&](const tensorium_RG::Field3D<double> &field, double asymptotic,
                            double speed) {
        const double value = field.ptr()[idx];
        const double dn = -deriv_axis(field, field.st.sx, i, I0, I1, 1.0 / grid.dx, 0.5 / grid.dx);
        return -speed * (dn + (value - asymptotic) / r);
    };

    const double expected_theta = expected_rhs(grid.Theta, 0.0, 1.0);
    const double expected_khat = expected_rhs(khat_field, 0.0, sqrt2);
    const double expected_alpha = expected_rhs(grid.alpha, 1.0, 1.0);
    const double expected_chi = expected_rhs(grid.chi, 1.0, 1.0);
    const double expected_beta0 = expected_rhs(grid.beta[0], 0.0, 1.0);
    const double expected_B0 = expected_rhs(grid.B[0], 0.0, 1.0);
    const double expected_gt_xx = expected_rhs(grid.gamma_tilde[tensorium_RG::XX], 1.0, 1.0);
    const double expected_gamma0 = expected_rhs(grid.tildeGamma[0], 0.0, 1.0);
    const double expected_Axx = expected_rhs(grid.A_tilde[tensorium_RG::XX], 0.0, 1.0);

    tensorium::tests::expect_le(std::abs(rhs.alpha.ptr()[idx] - expected_alpha), 1e-12,
                                "Alpha boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.chi.ptr()[idx] - expected_chi), 1e-12,
                                "Chi boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.beta[0].ptr()[idx] - expected_beta0), 1e-12,
                                "Beta boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.B[0].ptr()[idx] - expected_B0), 1e-12,
                                "B boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.gamma_tilde[tensorium_RG::XX].ptr()[idx] - expected_gt_xx),
                                1e-12, "Gamma_tilde boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.Theta.ptr()[idx] - expected_theta), 1e-12,
                                "Theta boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.K.ptr()[idx] - expected_khat), 1e-12,
                                "Khat boundary RHS matches fast outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.tildeGamma[0].ptr()[idx] - expected_gamma0), 1e-12,
                                "Gamma boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.A_tilde[tensorium_RG::XX].ptr()[idx] - expected_Axx),
                                1e-12, "A_tilde boundary RHS matches outgoing mode");
    TENSORIUM_TEST_ASSERT(std::isfinite(rhs.Theta.ptr()[idx]));

    const size_t interior_idx = grid.alpha.idx(I0 + 1, J0 + 2, K0 + 2);
    tensorium::tests::expect_le(std::abs(rhs.Theta.ptr()[interior_idx]), 1e-15,
                                "Interior RHS entry stays untouched");
});

REGISTER_TEST("bssn.evolution.z4c_rhs_boundary_collar",
              "Radiative RHS collar sanitizes the ghost-dependent physical shell", []() {
    const size_t padding = 4;
    Grid         grid(18, 14, 12, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    const size_t nx_tot = grid.alpha.st.nx_tot;
    const size_t ny_tot = grid.alpha.st.ny_tot;
    const size_t nz_tot = grid.alpha.st.nz_tot;
    for (size_t i = 0; i < nx_tot; ++i)
        for (size_t j = 0; j < ny_tot; ++j)
            for (size_t k = 0; k < nz_tot; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const size_t idx = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[idx] = 1.0 + 0.03 * x - 0.01 * y;
                grid.Theta.ptr()[idx] = 0.2 + 0.02 * x + 0.01 * z;
            }

    tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
    rhs.allocate_like(grid);
    poison_workspace(rhs);
    poison_field(rhs.Theta);

    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));

    tensorium_RG::bssn::RHSBoundaryFaceMask mask{};
    mask.face[0][0] = true;
    mask.face[0][1] = false;
    mask.face[1][0] = false;
    mask.face[1][1] = false;
    mask.face[2][0] = false;
    mask.face[2][1] = false;

    tensorium_RG::bssn::apply_z4c_rhs_boundary(grid, rhs, mask, 4);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t j = J0 + (J1 - J0) / 2;
    const size_t k = K0 + (K1 - K0) / 2;

    for (size_t layer = 0; layer < 4; ++layer) {
        const size_t idx = grid.alpha.idx(I0 + layer, j, k);
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs.alpha.ptr()[idx]));
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs.Theta.ptr()[idx]));
    }

    const size_t idx_bulk = grid.alpha.idx(I0 + 4, j, k);
    TENSORIUM_TEST_ASSERT(!std::isfinite(rhs.alpha.ptr()[idx_bulk]));
    TENSORIUM_TEST_ASSERT(!std::isfinite(rhs.Theta.ptr()[idx_bulk]));
});

REGISTER_TEST("bssn.evolution.z4c_rhs_boundary_collar_tapers",
              "Radiative RHS collar blends toward the interior instead of imposing a flat shell",
              []() {
    const size_t padding = 4;
    Grid         grid(18, 14, 12, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    const size_t nx_tot = grid.alpha.st.nx_tot;
    const size_t ny_tot = grid.alpha.st.ny_tot;
    const size_t nz_tot = grid.alpha.st.nz_tot;
    for (size_t i = 0; i < nx_tot; ++i)
        for (size_t j = 0; j < ny_tot; ++j)
            for (size_t k = 0; k < nz_tot; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const size_t idx = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[idx] = 1.0 + 0.03 * x - 0.01 * y;
            }

    tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
    rhs.allocate_like(grid);
    const double sentinel = 7.0;
    std::fill(rhs.alpha.ptr(), rhs.alpha.ptr() + rhs.alpha.st.nx_tot * rhs.alpha.st.ny_tot *
                                             rhs.alpha.st.nz_tot,
              sentinel);

    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));

    tensorium_RG::bssn::RHSBoundaryFaceMask mask{};
    mask.face[0][0] = true;

    tensorium_RG::bssn::apply_z4c_rhs_boundary(grid, rhs, mask, 4);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t j = J0 + (J1 - J0) / 2;
    const size_t k = K0 + (K1 - K0) / 2;

    const size_t idx_surface = grid.alpha.idx(I0, j, k);
    const size_t idx_inner = grid.alpha.idx(I0 + 3, j, k);

    const double surface_dev = std::abs(rhs.alpha.ptr()[idx_surface] - sentinel);
    const double inner_dev = std::abs(rhs.alpha.ptr()[idx_inner] - sentinel);
    TENSORIUM_TEST_ASSERT(surface_dev > 1.0e-8);
    TENSORIUM_TEST_ASSERT(inner_dev > 1.0e-8);
    tensorium::tests::expect_le(inner_dev, surface_dev,
                                "Deepest collar layer is damped less than the surface");
});

REGISTER_TEST("bssn.evolution.constraint_halos_use_sommerfeld",
              "Radiative ghosts use Sommerfeld halo fills on radiative faces", []() {
    const size_t padding = 4;
    Grid         grid(12, 10, 8, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    tensorium_RG::bssn::BoundaryRadiative::set_characteristic(1.0, 0.0);
    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(true, true, true, true, true,
                                                                     true);
    tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(false, false, false, false, false,
                                                                false);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    auto fill_linear = [&](tensorium_RG::Field3D<double> &field, double c0, double cx, double cy,
                           double cz) {
        const size_t nx_tot = field.st.nx_tot;
        const size_t ny_tot = field.st.ny_tot;
        const size_t nz_tot = field.st.nz_tot;
        for (size_t i = 0; i < nx_tot; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k) {
                    double x, y, z;
                    grid.coords(i, j, k, x, y, z);
                    field.ptr()[field.idx(i, j, k)] = c0 + cx * x + cy * y + cz * z;
                }
    };

    fill_linear(grid.alpha, 1.0, 0.02, -0.01, 0.00);
    fill_linear(grid.Theta, 0.2, 0.03, 0.01, -0.02);
    fill_linear(grid.tildeGamma[0], -0.1, 0.04, -0.02, 0.01);

    tensorium_RG::bssn::BoundaryRadiative::apply_halo(grid.alpha, grid,
                                                      tensorium_RG::bssn::BoundaryField::Alpha, 0);
    tensorium_RG::bssn::BoundaryRadiative::apply_halo(grid.Theta, grid, tensorium_RG::bssn::BoundaryField::Theta,
                                                      0);
    tensorium_RG::bssn::BoundaryRadiative::apply_halo(
        grid.tildeGamma[0], grid, tensorium_RG::bssn::BoundaryField::TildeGamma, 0);

    const size_t j = J0 + 2;
    const size_t k = K0 + 2;
    const size_t ob1 = grid.alpha.idx(I0 - 1, j, k);
    const size_t ob2 = grid.alpha.idx(I0 - 2, j, k);
    const size_t ib0 = grid.alpha.idx(I0, j, k);
    const size_t ib1 = grid.alpha.idx(I0 + 1, j, k);

    const double theta_linear_1 = 2.0 * grid.Theta.ptr()[ib0] - grid.Theta.ptr()[ib1];
    const double theta_linear_2 = 3.0 * grid.Theta.ptr()[ib0] - 2.0 * grid.Theta.ptr()[ib1];
    tensorium::tests::expect_le(std::abs(grid.Theta.ptr()[ob1]), std::abs(theta_linear_1) + 1e-12,
                                "Theta Sommerfeld ghost damps more than linear extrapolation");
    tensorium::tests::expect_le(std::abs(grid.Theta.ptr()[ob2]), std::abs(theta_linear_2) + 1e-3,
                                "Second Theta Sommerfeld ghost damps more than linear extrapolation");
    TENSORIUM_TEST_ASSERT(std::abs(grid.Theta.ptr()[ob1] - theta_linear_1) > 1e-6);

    const double gamma_linear_1 =
        2.0 * grid.tildeGamma[0].ptr()[ib0] - grid.tildeGamma[0].ptr()[ib1];
    tensorium::tests::expect_le(std::abs(grid.tildeGamma[0].ptr()[ob1]),
                                std::abs(gamma_linear_1) + 1e-12,
                                "TildeGamma Sommerfeld ghost damps more than linear extrapolation");
    TENSORIUM_TEST_ASSERT(std::abs(grid.tildeGamma[0].ptr()[ob1] - gamma_linear_1) > 1e-6);

    const double alpha_linear_extrap = 2.0 * grid.alpha.ptr()[ib0] - grid.alpha.ptr()[ib1];
    tensorium::tests::expect_le(std::abs(grid.alpha.ptr()[ob1] - 1.0),
                                std::abs(alpha_linear_extrap - 1.0) + 1e-12,
                                "Alpha Sommerfeld ghost is pulled toward the asymptotic state");
    TENSORIUM_TEST_ASSERT(std::abs(grid.alpha.ptr()[ob1] - alpha_linear_extrap) > 1e-6);
});

REGISTER_TEST("bssn.evolution.radiative_physical_cells_are_preserved",
              "Radiative halo fill must not overwrite physical cells", []() {
    const size_t padding = 4;
    Grid         grid(16, 12, 12, padding, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);

    tensorium_RG::bssn::BoundaryRadiative::set_characteristic(1.0, 0.0);
    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(true, true, true, true, true,
                                                                     true);
    tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(false, false, false, false, false,
                                                                false);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);

    auto fill_linear = [&](tensorium_RG::Field3D<double> &field, double c0, double cx, double cy,
                           double cz) {
        const size_t nx_tot = field.st.nx_tot;
        const size_t ny_tot = field.st.ny_tot;
        const size_t nz_tot = field.st.nz_tot;
        for (size_t i = 0; i < nx_tot; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k) {
                    double x, y, z;
                    grid.coords(i, j, k, x, y, z);
                    field.ptr()[field.idx(i, j, k)] = c0 + cx * x + cy * y + cz * z;
                }
    };

    fill_linear(grid.alpha, 1.0, 0.02, -0.01, 0.00);
    fill_linear(grid.Theta, 0.2, 0.03, 0.01, -0.02);

    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(grid);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t j = J0 + padding + 1;
    const size_t k = K0 + padding + 1;
    const size_t shell0_alpha = grid.alpha.idx(I0, j, k);
    const size_t shell3_alpha = grid.alpha.idx(I0 + padding - 1, j, k);
    const size_t shell0_theta = grid.Theta.idx(I0, j, k);
    const size_t shell3_theta = grid.Theta.idx(I0 + padding - 1, j, k);

    double x0, y0, z0;
    grid.coords(I0, j, k, x0, y0, z0);
    double x3, y3, z3;
    grid.coords(I0 + padding - 1, j, k, x3, y3, z3);

    const double alpha_shell0_expected = 1.0 + 0.02 * x0 - 0.01 * y0;
    const double alpha_shell3_expected = 1.0 + 0.02 * x3 - 0.01 * y3;
    const double theta_shell0_expected = 0.2 + 0.03 * x0 + 0.01 * y0 - 0.02 * z0;
    const double theta_shell3_expected = 0.2 + 0.03 * x3 + 0.01 * y3 - 0.02 * z3;

    tensorium::tests::expect_le(std::abs(grid.alpha.ptr()[shell0_alpha] - alpha_shell0_expected),
                                1e-12, "First physical alpha cell is preserved");
    tensorium::tests::expect_le(std::abs(grid.alpha.ptr()[shell3_alpha] - alpha_shell3_expected),
                                1e-12, "Fourth physical alpha cell is preserved");
    tensorium::tests::expect_le(std::abs(grid.Theta.ptr()[shell0_theta] - theta_shell0_expected),
                                1e-12, "First physical Theta cell is preserved");
    tensorium::tests::expect_le(std::abs(grid.Theta.ptr()[shell3_theta] - theta_shell3_expected),
                                1e-12, "Fourth physical Theta cell is preserved");
});

REGISTER_TEST("bssn.evolution.radiative_sponge_is_rhs_only",
              "Radiative sponge must not inject a Cartesian shell during halo refresh", []() {
    const size_t padding = 4;
    Grid         grid(16, 12, 12, padding, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);

    tensorium_RG::bssn::BoundaryRadiative::set_characteristic(1.0, 0.0);
    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(1.0, 1.0,
                                                                           std::sqrt(2.0));
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(true, true, true, true, true,
                                                                     true);
    tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(false, false, false, false, false,
                                                                false);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);
    tensorium_RG::bssn::BoundaryRadiative::set_sponge(true, 6, 8.0, 2.0);

    auto fill_linear = [&](tensorium_RG::Field3D<double> &field, double c0, double cx, double cy,
                           double cz) {
        const size_t nx_tot = field.st.nx_tot;
        const size_t ny_tot = field.st.ny_tot;
        const size_t nz_tot = field.st.nz_tot;
        for (size_t i = 0; i < nx_tot; ++i)
            for (size_t j = 0; j < ny_tot; ++j)
                for (size_t k = 0; k < nz_tot; ++k) {
                    double x, y, z;
                    grid.coords(i, j, k, x, y, z);
                    field.ptr()[field.idx(i, j, k)] = c0 + cx * x + cy * y + cz * z;
                }
    };

    fill_linear(grid.alpha, 1.0, 0.02, -0.01, 0.00);
    fill_linear(grid.Theta, 0.2, 0.03, 0.01, -0.02);

    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(grid);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t j = J0 + 1;
    const size_t k = K0 + 1;
    const size_t idx_alpha = grid.alpha.idx(I0, j, k);
    const size_t idx_theta = grid.Theta.idx(I0 + 1, j, k);

    double x0, y0, z0;
    grid.coords(I0, j, k, x0, y0, z0);
    double x1, y1, z1;
    grid.coords(I0 + 1, j, k, x1, y1, z1);

    tensorium::tests::expect_le(std::abs(grid.alpha.ptr()[idx_alpha] - (1.0 + 0.02 * x0 - 0.01 * y0)),
                                1.0e-12, "Alpha state is unchanged by the radiative sponge");
    tensorium::tests::expect_le(
        std::abs(grid.Theta.ptr()[idx_theta] - (0.2 + 0.03 * x1 + 0.01 * y1 - 0.02 * z1)), 1.0e-12,
        "Theta state is unchanged by the radiative sponge");

    tensorium_RG::bssn::BoundaryRadiative::set_sponge(false, 0, 0.0, 2.0);
});

REGISTER_TEST("bssn.evolution.radiative_field_speeds_are_grouped",
              "Radiative halo closure uses separate gauge and Z4c characteristic speeds", []() {
    const size_t padding = 4;
    Grid         grid(12, 10, 8, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    tensorium_RG::bssn::BoundaryRadiative::set_characteristic(1.0, 1.0);
    tensorium_RG::bssn::BoundaryRadiative::set_field_characteristic_speeds(2.0, 1.0,
                                                                           std::sqrt(2.0));
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(true, true, true, true, true,
                                                                     true);
    tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(false, false, false, false, false,
                                                                false);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t j = J0 + 2;
    const size_t k = K0 + 2;

    auto fill_linear = [&](tensorium_RG::Field3D<double> &field, double c0, double cx) {
        const size_t nx_tot = field.st.nx_tot;
        const size_t ny_tot = field.st.ny_tot;
        const size_t nz_tot = field.st.nz_tot;
        for (size_t i = 0; i < nx_tot; ++i)
            for (size_t jj = 0; jj < ny_tot; ++jj)
                for (size_t kk = 0; kk < nz_tot; ++kk) {
                    double x, y, z;
                    grid.coords(i, jj, kk, x, y, z);
                    field.ptr()[field.idx(i, jj, kk)] = c0 + cx * x;
                }
    };

    fill_linear(grid.alpha, 1.0, 0.02);
    fill_linear(grid.Theta, 0.2, 0.02);

    tensorium_RG::bssn::BoundaryRadiative::apply_halo(grid.alpha, grid,
                                                      tensorium_RG::bssn::BoundaryField::Alpha, 0);
    tensorium_RG::bssn::BoundaryRadiative::apply_halo(grid.Theta, grid,
                                                      tensorium_RG::bssn::BoundaryField::Theta, 0);

    const size_t ob = grid.alpha.idx(I0 - 1, j, k);
    const size_t ib = grid.alpha.idx(I0, j, k);
    const double alpha_dev = std::abs(grid.alpha.ptr()[ob] - 1.0);
    const double theta_dev = std::abs(grid.Theta.ptr()[ob]);

    TENSORIUM_TEST_ASSERT(alpha_dev < theta_dev);
});

REGISTER_TEST("bssn.evolution.boundary_characteristics_flag_outflow_gauge_face",
              "Boundary characteristic analysis flags outflow faces as gauge mismatches", []() {
    const size_t padding = 4;
    Grid         grid(12, 10, 8, padding, 0.5, 0.4, 0.3);
    tensorium_RG::init::minkowski(grid, 0.0);

    tensorium_RG::bssn::MovingPunctureEnvConfig cfg;
    cfg.radiative_collar_width = 2;
    cfg.boundary_faces.rhs_ix1 = false;
    cfg.boundary_faces.rhs_ox1 = true;
    cfg.boundary_faces.rhs_ix2 = true;
    cfg.boundary_faces.rhs_ox2 = true;
    cfg.boundary_faces.rhs_ix3 = true;
    cfg.boundary_faces.rhs_ox3 = true;
    tensorium_RG::bssn::apply_boundary_configuration(cfg);
    tensorium_RG::bssn::BoundaryRadiative::set_active_faces(true, true, true, true, true, true);

    auto params = cfg.gauge_params;
    const auto summaries =
        tensorium_RG::bssn::configure_boundary_characteristics_from_state(grid, cfg, params);

    const auto &ix1 = summaries[0];
    const auto &ox1 = summaries[1];
    TENSORIUM_TEST_ASSERT(ix1.mismatch);
    TENSORIUM_TEST_ASSERT(ix1.configured_gauge_speed == 0.0);
    TENSORIUM_TEST_ASSERT(ix1.gauge_speed_max > 1.0);
    TENSORIUM_TEST_ASSERT(!ox1.mismatch);
    TENSORIUM_TEST_ASSERT(params.boundary_gauge_characteristic_speed >= std::sqrt(2.0));
});

#endif
