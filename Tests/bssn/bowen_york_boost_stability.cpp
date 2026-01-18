#include "../framework/TestRegistry.hpp"
#include "stability_common.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

namespace {

void export_slice_csv(const tensorium::tests::Grid &grid, size_t step,
                      const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/slice_" << std::setw(4) << std::setfill('0') << step << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,alpha,W,mask\n";

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;

            const size_t idx = grid.alpha.idx(i, j, k);

            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            const double W = std::sqrt(std::max(chi, 1e-16));

            const double mask = (alpha < 0.1) ? 1.0 : 0.0;

            file << x << "," << y << "," << alpha << "," << W << "," << mask << "\n";
        }
    }
}

using tensorium::tests::Grid;

void initialize_single_boost(Grid &grid, double separation, double linear_momentum) {
    const double m1 = 1.0;
    const double m2 = 1.0;

    const double x1 = -separation;
    const double x2 = separation;
    const double y = 0.0;
    const double z = 0.0;

    const double P1[3] = {0.0, linear_momentum, 0.0};
    const double P2[3] = {0.0, -linear_momentum, 0.0};
    const double S1[3] = {0.0, 0.0, 0.0};
    const double S2[3] = {0.0, 0.0, 0.0};

    tensorium_RG::init::binary_bowen_york_puncture_init(grid, m1, x1, y, z, P1, S1, m2, x2, y, z,
                                                        P2, S2, 1e-10);
}

} // namespace

REGISTER_TEST(
    "bssn.viz.moving_puncture", "Export CSV slices of a moving spinning black hole", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 128;
        cfg.ny = 128;
        cfg.nz = 128;
        cfg.spacing = 0.4;
        cfg.ng = 6;
        cfg.padding = 8;
        cfg.steps = 400;
        cfg.cfl = 0.08;
        cfg.gauge_factor = 1.0;

        tensorium_RG::fd::set_fd_dx(cfg.spacing);
        (void)system("mkdir -p Output/viz");

        tensorium::tests::Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing,
                                    cfg.spacing);

        grid.x0 = -0.5 * cfg.spacing * cfg.nx + 0.5 * cfg.spacing;
        grid.y0 = -0.5 * cfg.spacing * cfg.ny + 0.5 * cfg.spacing;
        grid.z0 = -0.5 * cfg.spacing * cfg.nz + 0.5 * cfg.spacing;

        const double separation = 5.0;
        const double momentum = 0.25;
        initialize_single_boost(grid, separation, momentum);
        tensorium_RG::init::sanitize_bssn_initial_state(grid);

        tensorium_RG::bssn::ProjectionConfig proj_cfg;
        proj_cfg.padding = 0;
        proj_cfg.renormalize_metric = true;
        proj_cfg.project_A_tilde = true;
        proj_cfg.recompute_inverse = true;

        tensorium_RG::bssn::GaugeParameters<double> params;
        params.eta = 1.5; // Augmenté de 0.5 à 1.5

        params.beta_B_coeff = 0.75;

        params.kappa1 = 0.25; // Augmenté de 0.1 à 0.25
        params.kappa2 = 0.0;
        params.kappa_z = 1.0;
        params.use_theta_in_lapse = true;
        params.ko_sigma = 0.6;
        params.min_lapse_for_K = 1e-4;
        params.max_K_squared = 1e4;
        tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);
        tensorium_RG::init::zero_z4c_fields(grid);

        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, cfg.padding);
        stepper.set_gauge_parameters(params);

        const size_t export_stride = 20;
        stepper.set_snapshot_callback(
            [export_stride](const tensorium::tests::Grid &g, size_t step) {
                if (step % export_stride == 0) {
                    std::cout << ">> Exporting slice " << step << "..." << std::endl;
                    export_slice_csv(g, step, "Output/viz");
                }
            });

        auto constraint_scratch = tensorium::tests::make_constraint_scratch(grid);
        auto log_diagnostics = [&](size_t step_index) {
            const double r_min = 2.0 * cfg.spacing;
            const double r_max = 0.45 * cfg.spacing * (cfg.nx - 1);
            tensorium_RG::bssn::compute_bssn_constraints(grid, grid.Ricci, constraint_scratch.H,
                                                         constraint_scratch.M, constraint_scratch.C,
                                                         r_min, r_max, 0.0, 0.0, 0.0, false);
            auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, constraint_scratch.H,
                                                                        cfg.padding);
            tensorium_RG::bssn::populate_constraint_norms(grid, constraint_scratch.M, stats,
                                                          cfg.padding);
            std::cout << "[diag] step=" << step_index << " ||Theta||2=" << stats.l2_theta
                      << " ||Z||2=" << stats.l2_Z << " ||H||2=" << stats.l2_H
                      << " ||M||2=" << stats.l2_M << " det_drift=" << stats.max_det_drift
                      << " trA=" << stats.max_trace_A << std::endl;
            constexpr double theta_cap = 6e-2;
            constexpr double constraint_cap = 6e-1;
            if (stats.l2_theta > theta_cap || stats.l2_Z > theta_cap ||
                stats.l2_H > constraint_cap || stats.l2_M > constraint_cap) {
                std::cout << "[warn] aborting run due to constraint growth" << std::endl;
                return false;
            }
            return true;
        };

        auto guard_gauge = [&](size_t step_index) {
            size_t i0, i1, j0, j1, k0, k1;
            tensorium::tests::interior_bounds(grid, cfg.padding, i0, i1, j0, j1, k0, k1);
            double alpha_min = std::numeric_limits<double>::infinity();
            double chi_min = std::numeric_limits<double>::infinity();
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    for (size_t k = k0; k < k1; ++k) {
                        const size_t idx = grid.alpha.idx(i, j, k);
                        alpha_min = std::min(alpha_min, double(grid.alpha.ptr()[idx]));
                        chi_min = std::min(chi_min, double(grid.chi.ptr()[idx]));
                    }
            if (alpha_min < 1e-4 ||
                chi_min < 1e-4) { // Limite abaissée pour laisser vivre le trou noir
                std::cout << "[warn] gauge collapse alpha_min=" << alpha_min
                          << " chi_min=" << chi_min << " at step=" << step_index << std::endl;
                return false;
            }
            return true;
        };

        double t = 0.0;

        const size_t projection_stride = 25;
        const size_t log_stride = 20;

        bool   constraint_violation = false;
        bool   gauge_instability = false;
        size_t failure_step = std::numeric_limits<size_t>::max();

        for (size_t n = 0; n <= cfg.steps; ++n) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(
                grid, tensorium::tests::make_cfl_control(cfg), cfg.padding);

            stepper.step(grid, dt, n);
            t += dt;

            if (n % projection_stride == 0)
                tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);

            if (n % log_stride == 0 && !log_diagnostics(n)) {
                constraint_violation = true;
                failure_step = n;
                break;
            }

            if (!guard_gauge(n)) {
                gauge_instability = true;
                failure_step = n;
                break;
            }

            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n, cfg.steps, t);
        }

        if (constraint_violation) {
            std::ostringstream oss;
            oss << "Constraint growth exceeded stability limits at step " << failure_step;
            tensorium::tests::raise_failure(oss.str());
        }

        if (gauge_instability) {
            std::ostringstream oss;
            oss << "Gauge collapse detected (alpha or chi became singular) at step "
                << failure_step;
            tensorium::tests::raise_failure(oss.str());
        }
    });
REGISTER_TEST(
    "bssn.init.kerr_schild", "Print formatted BSSN fields after Kerr–Schild initialization", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 128;
        cfg.ny = 128;
        cfg.nz = 128;
        cfg.spacing = 0.5;
        cfg.ng = 4;
        cfg.steps = 60;
        cfg.cfl = 0.15;
        cfg.gauge_factor = 1.0;

        tensorium_RG::fd::set_fd_dx(cfg.spacing);

        tensorium::tests::Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing,
                                    cfg.spacing);

        grid.x0 = -0.5 * cfg.spacing * cfg.nx + 0.5 * cfg.spacing;
        grid.y0 = -0.5 * cfg.spacing * cfg.ny + 0.5 * cfg.spacing;
        grid.z0 = -0.5 * cfg.spacing * cfg.nz + 0.5 * cfg.spacing;

        tensorium_RG::init::kerr_schild_single(grid, 1.0, 0.9);

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

        const size_t                                ic = I0 + grid.dims.nx / 2;
        const size_t                                jc = J0 + grid.dims.ny / 2;
        const size_t                                kc = K0 + grid.dims.nz / 2;
        tensorium_RG::bssn::GaugeParameters<double> params;
        params.eta = 2.0;
        params.beta_B_coeff = 0.75;

        tensorium_RG::init::print_bssn_state_at(grid, ic, jc, kc, "Kerr–Schild single BH");
        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, cfg.padding);
        stepper.set_gauge_parameters(params);

        stepper.set_snapshot_callback([](const tensorium::tests::Grid &g, size_t step) {
            if (step % 1 == 0) {
                std::cout << ">> Exporting slice " << step << "..." << std::endl;
                export_slice_csv(g, step, "Output/viz");
            }
        });

        double t = 0.0;

        for (size_t n = 0; n <= cfg.steps; ++n) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(
                grid, tensorium::tests::make_cfl_control(cfg), cfg.padding);

            stepper.step(grid, dt, n);
            t += dt;

            std::printf("dt = %.4e\n", dt);
            std::printf("Step %zu / %zu (t=%.4f)\n", n, cfg.steps, t);
        }
        std::cout << "Kerr–Schild init test completed.\n";
    });
