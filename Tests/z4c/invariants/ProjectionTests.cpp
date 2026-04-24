#ifndef TENSORIUM_BSSN_PROJECTION_TESTS_CPP
#define TENSORIUM_BSSN_PROJECTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjection.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

REGISTER_TEST("z4c.invariants.projection", "Projection enforces algebraic constraints", []() {
    tensorium_RG::Z4cGridSoA<double> grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    tensorium_RG::init::minkowski(grid, 0.0);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ic = I0 + grid.dims.nx / 2;
    const size_t jc = J0 + grid.dims.ny / 2;
    const size_t kc = K0 + grid.dims.nz / 2;
    const size_t id = grid.gamma_tilde[tensorium_RG::XX].idx(ic, jc, kc);

    // Perturb gamma_tilde and A_tilde locally
    grid.gamma_tilde[tensorium_RG::XX].ptr()[id] *= 1.01;
    grid.gamma_tilde[tensorium_RG::YY].ptr()[id] *= 1.02;
    grid.gamma_tilde[tensorium_RG::ZZ].ptr()[id] *= 0.99;
    grid.A_tilde[tensorium_RG::XX].ptr()[id] += 1e-2;

    auto before = tensorium_RG::z4c::compute_invariant_stats(grid, 4);
    TENSORIUM_TEST_ASSERT(before.max_det_deviation > 1e-6);
    TENSORIUM_TEST_ASSERT(before.max_trace_A > 1e-6);

    tensorium_RG::z4c::project_z4c_state(grid);

    auto after = tensorium_RG::z4c::compute_invariant_stats(grid, 4);
    TENSORIUM_TEST_ASSERT(after.max_det_deviation < before.max_det_deviation);
    TENSORIUM_TEST_ASSERT(after.max_trace_A < before.max_trace_A);
    TENSORIUM_TEST_ASSERT(after.max_metric_identity < 1e-9);
});

#endif // TENSORIUM_BSSN_PROJECTION_TESTS_CPP
