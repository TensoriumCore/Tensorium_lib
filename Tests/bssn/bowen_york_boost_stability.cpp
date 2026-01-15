#include "../framework/TestRegistry.hpp"
#include "stability_common.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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

void initialize_single_boost(Grid &grid) {
    const double m1 = 1.0;
    const double m2 = 1.0;

    const double x1 = -3.0, y1 = 0.0, z1 = 0.0;
    const double x2 = 3.0, y2 = 0.0, z2 = 0.0;

    const double Py = 0.095;
    const double P1[3] = {0.0, Py, 0.0};
    const double P2[3] = {0.0, -Py, 0.0};

    const double S1[3] = {0.0, 0.0, 0.1};
    const double S2[3] = {0.0, 0.0, 0.1};

    tensorium_RG::init::binary_bowen_york_puncture_init(grid, m1, x1, y1, z1, P1, S1, m2, x2, y2,
                                                        z2, P2, S2, 1e-10);
}

} // namespace

REGISTER_TEST(
    "bssn.viz.moving_puncture", "Export CSV slices of a moving spinning black hole", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 64;
        cfg.ny = 64;
        cfg.nz = 64;
        cfg.spacing = 0.1;
        cfg.ng = 4;
        cfg.padding = 4;
        cfg.steps = 600;
        cfg.cfl = 0.25;
        cfg.gauge_factor = 1.0;

        tensorium_RG::fd::set_fd_dx(cfg.spacing);
        (void)system("mkdir -p Output/viz");

        tensorium::tests::Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing,
                                    cfg.spacing);

        grid.x0 = -0.5 * cfg.spacing * cfg.nx + 0.5 * cfg.spacing;
        grid.y0 = -0.5 * cfg.spacing * cfg.ny + 0.5 * cfg.spacing;
        grid.z0 = -0.5 * cfg.spacing * cfg.nz + 0.5 * cfg.spacing;

        initialize_single_boost(grid);

        tensorium_RG::bssn::ProjectionConfig proj_cfg;
        proj_cfg.padding = 0;
        proj_cfg.renormalize_metric = true;
        proj_cfg.project_A_tilde = true;
        proj_cfg.recompute_inverse = true;

        tensorium_RG::bssn::GaugeParameters<double> params;
        params.eta = 3.0;
        params.beta_B_coeff = 0.5;

        tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);

        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, cfg.padding);
        stepper.set_gauge_parameters(params);

        stepper.set_snapshot_callback([](const tensorium::tests::Grid &g, size_t step) {
            if (step % 10 == 0) {
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
    });
REGISTER_TEST(
    "bssn.init.kerr_schild", "Print formatted BSSN fields after Kerr–Schild initialization", []() {
        tensorium::tests::StabilityRunConfig cfg;
        cfg.nx = 64;
        cfg.ny = 64;
        cfg.nz = 64;
        cfg.spacing = 0.1;
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

        tensorium_RG::init::kerr_schild_single(grid, 1.0, 0.0);

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

        const size_t                                ic = I0 + grid.dims.nx / 2;
        const size_t                                jc = J0 + grid.dims.ny / 2;
        const size_t                                kc = K0 + grid.dims.nz / 2;
        tensorium_RG::bssn::GaugeParameters<double> params;
        params.eta = 3.0;
        params.beta_B_coeff = 0.5;

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
