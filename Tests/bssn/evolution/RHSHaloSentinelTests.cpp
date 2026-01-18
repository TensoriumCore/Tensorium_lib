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

#endif
