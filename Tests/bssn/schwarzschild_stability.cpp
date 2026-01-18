#include "../framework/TestRegistry.hpp"

#include "stability_common.hpp"

#include <cmath>

namespace {

using tensorium::tests::RunSummary;
using tensorium::tests::StabilityRunConfig;
using tensorium::tests::make_cfl_control;
using tensorium::tests::run_stability_case;
using tensorium::tests::Grid;

constexpr double kMaxDetDrift = 5e-6;
constexpr double kMaxTraceA = 5e-6;

RunSummary run_schwarzschild(const StabilityRunConfig &cfg, bool allow_failure = false) {
    auto init = [](Grid &grid) { tensorium_RG::init::schwarzschild_isotropic(grid, 1.0); };
    return run_stability_case("schwarzschild", cfg, init, make_cfl_control(cfg), allow_failure);
}

} // namespace

REGISTER_TEST("bssn.stability.schwarzschild",
               "Gauge-aware CFL preserves Schwarzschild isotropic data", []() {
                   StabilityRunConfig legacy;
                   legacy.steps = 180;
                   legacy.cfl = 0.25;
                   legacy.gauge_factor = 1.0;
                   legacy.log_path = "Output/tests/schwarzschild_cfl_legacy.csv";
                   legacy.spacing = 0.5;
                   legacy.padding = 4;
                   legacy.freeze_gauge = false;
                   legacy.simulate_only = true;

                   StabilityRunConfig gauge_cfg = legacy;
                   gauge_cfg.cfl = 0.15;
                   gauge_cfg.gauge_factor = std::sqrt(2.0);
                   gauge_cfg.log_path = "Output/tests/schwarzschild_cfl_gaugeaware.csv";

                   const RunSummary legacy_summary = run_schwarzschild(legacy, true);
                   const RunSummary gauge_summary = run_schwarzschild(gauge_cfg);

                   TENSORIUM_TEST_ASSERT(gauge_summary.max_det_drift < kMaxDetDrift);
                   TENSORIUM_TEST_ASSERT(gauge_summary.max_trace_A < kMaxTraceA);

                   if (std::isfinite(legacy_summary.max_det_drift))
                       TENSORIUM_TEST_ASSERT(gauge_summary.max_det_drift <=
                                             legacy_summary.max_det_drift + 5e-7);
                   if (std::isfinite(legacy_summary.max_trace_A))
                       TENSORIUM_TEST_ASSERT(gauge_summary.max_trace_A <=
                                             legacy_summary.max_trace_A + 5e-7);
               });
