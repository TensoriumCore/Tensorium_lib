#ifndef TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP
#define TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <limits>

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

REGISTER_TEST("bssn.evolution.radiative_boundary_shells_are_evolved",
              "Radiative RK4 evolves the first physical layers instead of freezing them", []() {
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

    const size_t j = J0 + 2;
    const size_t k = K0 + 2;
    const size_t idx_face = grid.alpha.idx(I0, j, k);
    const size_t idx_shell3 = grid.alpha.idx(I0 + 3, j, k);
    const double alpha_face_before = grid.alpha.ptr()[idx_face];
    const double alpha_shell3_before = grid.alpha.ptr()[idx_shell3];

    stepper.step(grid, 1.0e-3, 0);

    TENSORIUM_TEST_ASSERT(grid.alpha.ptr()[idx_face] < alpha_face_before);
    TENSORIUM_TEST_ASSERT(grid.alpha.ptr()[idx_shell3] < alpha_shell3_before);
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
    const size_t j = J0 + 2;
    const size_t k = K0 + 2;
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
              "Radiative RHS collar overwrites the ghost-dependent physical shell", []() {
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
    const size_t j = J0 + 2;
    const size_t k = K0 + 2;

    for (size_t layer = 0; layer < 4; ++layer) {
        const size_t idx = grid.alpha.idx(I0 + layer, j, k);
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs.alpha.ptr()[idx]));
        TENSORIUM_TEST_ASSERT(std::isfinite(rhs.Theta.ptr()[idx]));
    }

    const size_t idx_bulk = grid.alpha.idx(I0 + 4, j, k);
    TENSORIUM_TEST_ASSERT(!std::isfinite(rhs.alpha.ptr()[idx_bulk]));
    TENSORIUM_TEST_ASSERT(!std::isfinite(rhs.Theta.ptr()[idx_bulk]));
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
