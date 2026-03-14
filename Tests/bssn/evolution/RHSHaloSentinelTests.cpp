#ifndef TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP
#define TENSORIUM_BSSN_RHS_HALO_SENTINEL_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
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

REGISTER_TEST("bssn.evolution.z4c_rhs_boundary_operator",
              "Z4c boundary helper applies outgoing Theta/Khat/Gamma/A modes", []() {
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
    rhs.zero();

    tensorium_RG::bssn::RHSBoundaryFaceMask mask{};
    mask.face[0][0] = true;
    mask.face[0][1] = false;
    mask.face[1][0] = false;
    mask.face[1][1] = false;
    mask.face[2][0] = false;
    mask.face[2][1] = false;

    tensorium_RG::bssn::apply_z4c_rhs_boundary(grid, rhs, mask);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i = I0;
    const size_t j = J0 + 2;
    const size_t k = K0 + 2;
    const size_t idx = grid.alpha.idx(i, j, k);

    double x, y, z;
    grid.coords(i, j, k, x, y, z);
    const double r = std::sqrt(x * x + y * y + z * z);
    const double sx = x / r;
    const double sy = y / r;
    const double sz = z / r;
    const double sqrt2 = std::sqrt(2.0);

    auto expected_rhs = [&](const tensorium_RG::Field3D<double> &field, double asymptotic,
                            double speed) {
        const double value = field.ptr()[idx];
        const double radial = sx * ((field.ptr()[idx + field.st.sx] - field.ptr()[idx - field.st.sx]) /
                                    (2.0 * grid.dx)) +
                              sy * ((field.ptr()[idx + field.st.sy] - field.ptr()[idx - field.st.sy]) /
                                    (2.0 * grid.dy)) +
                              sz * ((field.ptr()[idx + 1] - field.ptr()[idx - 1]) /
                                    (2.0 * grid.dz));
        return -speed * (radial + (value - asymptotic) / r);
    };

    const double expected_theta = expected_rhs(grid.Theta, 0.0, 1.0);
    const double expected_khat = expected_rhs(khat_field, 0.0, sqrt2);
    const double expected_gamma0 = expected_rhs(grid.tildeGamma[0], 0.0, 1.0);
    const double expected_Axx = expected_rhs(grid.A_tilde[tensorium_RG::XX], 0.0, 1.0);

    tensorium::tests::expect_le(std::abs(rhs.Theta.ptr()[idx] - expected_theta), 1e-12,
                                "Theta boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.K.ptr()[idx] - expected_khat), 1e-12,
                                "Khat boundary RHS matches fast outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.tildeGamma[0].ptr()[idx] - expected_gamma0), 1e-12,
                                "Gamma boundary RHS matches outgoing mode");
    tensorium::tests::expect_le(std::abs(rhs.A_tilde[tensorium_RG::XX].ptr()[idx] - expected_Axx),
                                1e-12, "A_tilde boundary RHS matches outgoing mode");

    const size_t interior_idx = grid.alpha.idx(I0 + 1, J0 + 2, K0 + 2);
    tensorium::tests::expect_le(std::abs(rhs.Theta.ptr()[interior_idx]), 1e-15,
                                "Interior RHS entry stays untouched");
});

#endif
