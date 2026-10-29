#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/FMR/BSSNFixedMeshRefinement.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using Grid = tensorium_RG::BSSNGridSoA<double>;
namespace bssn = tensorium_RG::bssn;

struct TimeProbeBoundary : bssn::BoundaryRadiative {
    static constexpr bool evolve_physical_cells = true;
    static constexpr bool radiative_rhs_collar_enabled = false;
    inline static std::vector<double> fractions;
    static void set_stage_fraction(double fraction) { fractions.push_back(fraction); }
};

void initialize_homogeneous_flat(Grid &grid) {
    const size_t total = grid.st.nx_tot * grid.st.ny_tot * grid.st.nz_tot;
    for (size_t id = 0; id < total; ++id) {
        grid.alpha.ptr()[id] = 2.0;
        grid.chi.ptr()[id] = 1.0;
        grid.K.ptr()[id] = grid.Theta.ptr()[id] = 0.0;
        for (int c = 0; c < 3; ++c) {
            grid.beta[c].ptr()[id] = grid.B[c].ptr()[id] = 0.0;
            grid.tildeGamma[c].ptr()[id] = grid.Z[c].ptr()[id] = 0.0;
        }
        for (int s = 0; s < 6; ++s) {
            const double value = (s == tensorium_RG::XX || s == tensorium_RG::YY ||
                                  s == tensorium_RG::ZZ) ? 1.0 : 0.0;
            grid.gamma_tilde[s].ptr()[id] = grid.gamma_tilde_inv[s].ptr()[id] = value;
            grid.A_tilde[s].ptr()[id] = grid.Ricci[s].ptr()[id] = 0.0;
        }
    }
}

double evolve_time_dependent_lapse(int steps) {
    Grid grid(8, 8, 8, 4, 0.5, 0.5, 0.5);
    initialize_homogeneous_flat(grid);
    bssn::BSSNRKStepper<double, TimeProbeBoundary> stepper(grid, 0);
    bssn::GaugeParameters<double> params;
    params.slow_start_lapse_article = true;
    params.slow_start_lapse_h = 0.6;
    params.slow_start_lapse_sigma = 0.7;
    stepper.set_gauge_parameters(params);
    stepper.set_state_log_stride(10000);
    TimeProbeBoundary::fractions.clear();
    for (int i = 0; i < steps; ++i)
        stepper.step(grid, 1.0 / steps, i);
    return grid.alpha.ptr()[grid.alpha.idx(8, 8, 8)];
}
} // namespace

REGISTER_TEST("z4c.evolution.reject_insufficient_ghosts",
              "Evolution and FMR reject halos narrower than the widest stencil", []() {
    for (size_t ng : {size_t(0), size_t(2), size_t(3)}) {
        Grid grid(8, 8, 8, ng, 0.5, 0.5, 0.5);
        bool rejected = false;
        try {
            bssn::BSSNRKStepper<double, bssn::BoundaryRadiative> stepper(grid, 0);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        TENSORIUM_TEST_ASSERT(rejected);
    }
    Grid grid(16, 16, 16, 4, 0.5, 0.5, 0.5);
    bssn::BSSNRKStepper<double, bssn::BoundaryRadiative> stepper(grid, 0);
    bool rejected = false;
    try {
        const bssn::fmr::LevelConfig cfg{{4, 12, 4, 12, 4, 12}, 2, 3};
        (void)bssn::fmr::detail::make_child_geometry(grid, cfg);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    TENSORIUM_TEST_ASSERT(rejected);
});

REGISTER_TEST("z4c.evolution.rk_time_dependent_lapse",
              "RK stage times preserve fourth-order accuracy for a time-dependent lapse", []() {
    // With K=Theta=0 and chi=1, alpha'=-h exp(-t^2/(2 sigma^2)) (alpha-1).
    constexpr double h = 0.6, sigma = 0.7;
    const double exact = 1.0 + std::exp(-h * sigma * std::sqrt(std::acos(-1.0) / 2.0) *
                                       std::erf(1.0 / (std::sqrt(2.0) * sigma)));
    const double coarse = evolve_time_dependent_lapse(4);
    const double fine = evolve_time_dependent_lapse(8);
    const double err_coarse = std::abs(coarse - exact);
    const double err_fine = std::abs(fine - exact);
    TENSORIUM_TEST_ASSERT(std::isfinite(err_coarse) && std::isfinite(err_fine));
    tensorium::tests::expect_le(err_fine, 2e-5, "time-dependent lapse error");
    TENSORIUM_TEST_ASSERT(err_coarse > 10.0 * err_fine);
    TENSORIUM_TEST_ASSERT(TimeProbeBoundary::fractions.size() == 8 * 5);
    for (int stage = 0; stage < 4; ++stage)
        tensorium::tests::expect_near(TimeProbeBoundary::fractions[stage],
                                      bssn::rk_stage_times[stage], 1e-14, "RK boundary stage time");
});
