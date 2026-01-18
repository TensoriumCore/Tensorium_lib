#pragma once

#include "../framework/Assertions.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjectionMonitor.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include "../utils/test_config.hpp"
#include "../utils/test_logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace tensorium::tests {

using Grid = tensorium_RG::BSSNGridSoA<double>;

struct ConstraintScratch {
    tensorium_RG::Field3D<double> H;
    tensorium_RG::Field3D<double> M[3];
    tensorium_RG::Field3D<double> C[3];
};

inline ConstraintScratch make_constraint_scratch(const Grid &grid) {
    ConstraintScratch scratch;
    scratch.H = tensorium_RG::make_field(grid.alpha.st);
    for (int q = 0; q < 3; ++q) {
        scratch.M[q] = tensorium_RG::make_field(grid.alpha.st);
        scratch.C[q] = tensorium_RG::make_field(grid.alpha.st);
    }
    return scratch;
}

inline void center_grid(Grid &grid) {
    const double hx = 0.5 * (static_cast<double>(grid.dims.nx) - 1.0) * grid.dx;
    const double hy = 0.5 * (static_cast<double>(grid.dims.ny) - 1.0) * grid.dy;
    const double hz = 0.5 * (static_cast<double>(grid.dims.nz) - 1.0) * grid.dz;
    grid.x0 = -hx;
    grid.y0 = -hy;
    grid.z0 = -hz;
}

inline void interior_bounds(const Grid &grid, size_t padding, size_t &i0, size_t &i1, size_t &j0,
                            size_t &j1, size_t &k0, size_t &k1) {
    grid.domain_bounds(i0, i1, j0, j1, k0, k1);
    const size_t guard = std::max<size_t>(padding, size_t(2));
    i0 = std::min(i0 + guard, i1);
    j0 = std::min(j0 + guard, j1);
    k0 = std::min(k0 + guard, k1);
    i1 = (i1 > guard) ? i1 - guard : i1;
    j1 = (j1 > guard) ? j1 - guard : j1;
    k1 = (k1 > guard) ? k1 - guard : k1;
}

inline std::pair<double, double> gauge_snapshot(const Grid &grid, size_t padding) {
    size_t i0, i1, j0, j1, k0, k1;
    interior_bounds(grid, padding, i0, i1, j0, j1, k0, k1);
    double max_gamma = 0.0;
    double min_alpha = std::numeric_limits<double>::infinity();

    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return {max_gamma, min_alpha};

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                min_alpha = std::min(min_alpha, static_cast<double>(grid.alpha.ptr()[id]));
                for (int c = 0; c < 3; ++c)
                    max_gamma = std::max(max_gamma, std::abs(grid.tildeGamma[c].ptr()[id]));
            }
    return {max_gamma, min_alpha};
}

inline RunSummary summarize_grid(const Grid &grid, size_t padding) {
    RunSummary summary;
    auto       scratch = make_constraint_scratch(grid);
    auto &     mutable_grid = const_cast<Grid &>(grid);
    tensorium_RG::bssn::compute_bssn_constraints(mutable_grid, grid.Ricci, scratch.H, scratch.M,
                                                 scratch.C, 0.0, std::numeric_limits<double>::max(),
                                                 0.0, 0.0, 0.0, false);
    const auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, scratch.H, padding);
    summary.max_H = stats.max_H;
    summary.max_det_drift = stats.max_det_drift;
    summary.max_trace_A = stats.max_trace_A;
    const auto gauges = gauge_snapshot(grid, padding);
    summary.max_gamma = gauges.first;
    summary.min_alpha = gauges.second;
    return summary;
}

template <typename BoundaryFunctor>
RunSummary run_stability_case_impl(const std::string &label, const StabilityRunConfig &cfg,
                                   const std::function<void(Grid &)>     &initializer,
                                   tensorium_RG::bssn::CFLControl<double> control,
                                   bool                                   allow_failure) {
    Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing, cfg.spacing);
    center_grid(grid);
    initializer(grid);

    TestLogger logger(cfg.log_path);
    if (logger.enabled())
        logger.write_header();

    auto constraint_fields = make_constraint_scratch(grid);

    if (cfg.simulate_only) {
        for (size_t step = 0; step < cfg.steps; ++step) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, control, cfg.padding);
            tensorium_RG::bssn::compute_bssn_constraints(
                grid, grid.Ricci, constraint_fields.H, constraint_fields.M, constraint_fields.C, 0.0,
                std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);
            const auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, constraint_fields.H,
                                                                              cfg.padding);
            if (logger.enabled()) {
                const auto gauges = gauge_snapshot(grid, cfg.padding);
                logger.write_step(step, dt, stats, gauges.first, gauges.second);
            }
        }
        tensorium_RG::bssn::compute_bssn_constraints(
            grid, grid.Ricci, constraint_fields.H, constraint_fields.M, constraint_fields.C, 0.0,
            std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);
        return summarize_grid(grid, cfg.padding);
    }

    tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative>
                                                stepper(grid, cfg.padding);
    tensorium_RG::bssn::GaugeParameters<double> gauge_params;
    gauge_params.beta_B_coeff = cfg.gauge_beta_coeff;
    gauge_params.eta = cfg.gauge_eta;
    stepper.set_gauge_parameters(gauge_params);

    const size_t total_cells = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
    std::vector<double>                alpha_reference;
    std::array<std::vector<double>, 3> beta_reference;
    std::array<std::vector<double>, 3> B_reference;
    if (cfg.freeze_gauge) {
        alpha_reference.assign(grid.alpha.ptr(), grid.alpha.ptr() + total_cells);
        for (int c = 0; c < 3; ++c) {
            beta_reference[c].assign(grid.beta[c].ptr(), grid.beta[c].ptr() + total_cells);
            B_reference[c].assign(grid.B[c].ptr(), grid.B[c].ptr() + total_cells);
        }
    }

    bool        aborted = false;
    std::string failure_message;
    for (size_t step = 0; step < cfg.steps; ++step) {
        const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, control, cfg.padding);
        try {
            stepper.step(grid, dt, step);
            if (cfg.freeze_gauge) {
                std::copy(alpha_reference.begin(), alpha_reference.end(), grid.alpha.ptr());
                for (int c = 0; c < 3; ++c) {
                    std::copy(beta_reference[c].begin(), beta_reference[c].end(),
                              grid.beta[c].ptr());
                    std::copy(B_reference[c].begin(), B_reference[c].end(), grid.B[c].ptr());
                }
            }
        } catch (const std::exception &ex) {
            if (!allow_failure)
                throw;
            aborted = true;
            failure_message = ex.what();
            break;
        }

        tensorium_RG::bssn::compute_bssn_constraints(
            grid, grid.Ricci, constraint_fields.H, constraint_fields.M, constraint_fields.C, 0.0,
            std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);
        if (logger.enabled()) {
            const auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, constraint_fields.H,
                                                                              cfg.padding);
            const auto gauges = gauge_snapshot(grid, cfg.padding);
            logger.write_step(step, dt, stats, gauges.first, gauges.second);
        }
    }

    tensorium_RG::bssn::compute_bssn_constraints(grid, grid.Ricci, constraint_fields.H,
                                                 constraint_fields.M, constraint_fields.C, 0.0,
                                                 std::numeric_limits<double>::max(), 0.0, 0.0, 0.0,
                                                 false);
    auto summary = summarize_grid(grid, cfg.padding);
    if (aborted) {
        (void)failure_message;
        summary.max_H = std::numeric_limits<double>::infinity();
    }
    (void)label;
    return summary;
}

inline RunSummary run_stability_case(const std::string &label, const StabilityRunConfig &cfg,
                                     const std::function<void(Grid &)>     &initializer,
                                     tensorium_RG::bssn::CFLControl<double> control,
                                     bool                                   allow_failure = false) {
    if (cfg.boundary == BoundaryType::Clamp)
        return run_stability_case_impl<tensorium_RG::bssn::BoundaryClamp>(label, cfg, initializer,
                                                                          control, allow_failure);
    return run_stability_case_impl<tensorium_RG::bssn::BoundarySponge>(label, cfg, initializer,
                                                                       control, allow_failure);
}

inline tensorium_RG::bssn::CFLControl<double> make_cfl_control(const StabilityRunConfig &cfg) {
    tensorium_RG::bssn::CFLControl<double> ctrl;
    ctrl.cfl = cfg.cfl;
    ctrl.gauge_speed = cfg.gauge_factor;
    return ctrl;
}

} // namespace tensorium::tests
