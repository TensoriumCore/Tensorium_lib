#include "../framework/TestRegistry.hpp"

#include "stability_common.hpp"

#include <cmath>

namespace {

using tensorium::tests::RunSummary;
using tensorium::tests::StabilityRunConfig;
using tensorium::tests::make_cfl_control;
using tensorium::tests::run_stability_case;
using tensorium::tests::Grid;

constexpr double kMaxHThreshold = 1e-10;
constexpr double kMaxGammaThreshold = 5e-9;
constexpr double kMinAlphaFloor = 0.1;

RunSummary run_minkowski(const StabilityRunConfig &cfg, bool allow_failure = false) {
    auto init = [](Grid &grid) { tensorium_RG::init::minkowski(grid, 0.0); };
    return run_stability_case("minkowski", cfg, init, make_cfl_control(cfg), allow_failure);
}

} // namespace

REGISTER_TEST("bssn.stability.minkowski", "Gauge-aware CFL keeps Minkowski stable", []() {
    StabilityRunConfig old_cfg;
    old_cfg.steps = 150;
    old_cfg.cfl = 0.25; // historical value
    old_cfg.gauge_factor = 1.0;
    old_cfg.log_path = "Output/tests/minkowski_cfl_legacy.csv";
    old_cfg.spacing = 0.5;
    old_cfg.padding = 4;
    old_cfg.freeze_gauge = false;
    old_cfg.simulate_only = true;

    StabilityRunConfig new_cfg = old_cfg;
    new_cfg.cfl = 0.15;
    new_cfg.gauge_factor = std::sqrt(2.0);
    new_cfg.log_path = "Output/tests/minkowski_cfl_gaugeaware.csv";

    const RunSummary old_summary = run_minkowski(old_cfg, true);
    const RunSummary new_summary = run_minkowski(new_cfg);

    TENSORIUM_TEST_ASSERT(new_summary.max_H < kMaxHThreshold);
    TENSORIUM_TEST_ASSERT(new_summary.max_gamma < kMaxGammaThreshold);
    TENSORIUM_TEST_ASSERT(new_summary.min_alpha > kMinAlphaFloor);

    if (std::isfinite(old_summary.max_H))
        TENSORIUM_TEST_ASSERT(new_summary.max_H <= old_summary.max_H + 1e-12);
    if (std::isfinite(old_summary.max_gamma))
        TENSORIUM_TEST_ASSERT(new_summary.max_gamma <= old_summary.max_gamma + 1e-9);
});
