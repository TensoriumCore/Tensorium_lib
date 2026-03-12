#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void export_slice_csv(const tensorium_RG::BSSNGridSoA<double> &grid, size_t step,
                      const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/slice_" << std::setw(4) << std::setfill('0') << step << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,alpha,chi,mask\n";
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
            const double mask = (alpha < 0.1) ? 1.0 : 0.0;
            file << x << "," << y << "," << alpha << "," << chi << "," << mask << "\n";
        }
    }
}

size_t parse_env_stride_or(const char *name, size_t fallback) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && parsed > 0)
            return static_cast<size_t>(parsed);
    }
    return fallback;
}

} // namespace

int main() {
    using Grid = tensorium_RG::BSSNGridSoA<double>;

    const auto cfg = tensorium_RG::bssn::load_moving_puncture_env();

    std::cout << "[mesh] nx=" << cfg.nx << " ny=" << cfg.ny << " nz=" << cfg.nz
              << " spacing=" << cfg.spacing
              << " box=(" << cfg.nx * cfg.spacing << ", " << cfg.ny * cfg.spacing << ", "
              << cfg.nz * cfg.spacing << ")\n";

    tensorium_RG::fd::set_max_spatial_derivative_order(cfg.spatial_derivative_order);
    std::cout << "[num] spatial_derivative_order=" << cfg.spatial_derivative_order << std::endl;
    tensorium_RG::fd::set_fd_dx(cfg.spacing);

    std::filesystem::create_directories("Output/viz");

    Grid grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing, cfg.spacing);
    tensorium_RG::bssn::center_cell_centered_origin(grid);

    if (cfg.print_suggested_momentum) {
        if (cfg.circular_hint.valid) {
            std::cout << "[Z4c.Init] Physics Diagnostic for d = " << cfg.circular_hint.d
                      << "\n[Z4c.Init] > Suggested P_tang (Newtonian): "
                      << cfg.circular_hint.p_newtonian
                      << "\n[Z4c.Init] > Suggested P_tang (Post-Newtonian): "
                      << cfg.circular_hint.p_pn
                      << "\n[Z4c.Init] Current P_tang (User): " << cfg.user_tangential_momentum
                      << std::endl;
        } else {
            std::cout << "[Z4c.Init] Physics Diagnostic unavailable: invalid masses/separation."
                      << std::endl;
        }
    }
    if (cfg.auto_circular) {
        if (cfg.circular_hint.valid) {
            std::cout << "[Z4c.Init] AUTO_CIRCULAR enabled: overriding P_tang from "
                      << cfg.user_tangential_momentum << " to " << cfg.tangential_momentum
                      << " (Post-Newtonian)." << std::endl;
        } else {
            std::cout << "[Z4c.Init] AUTO_CIRCULAR requested but suggestion is invalid; keeping "
                         "user momentum."
                      << std::endl;
        }
    }

    std::cout << "[init] puncture separation=" << cfg.separation
              << " momentum_tan=" << cfg.tangential_momentum
              << " momentum_rad=" << cfg.radial_momentum
              << " m1=" << cfg.mass1
              << " m2=" << cfg.mass2 << std::endl;
    std::cout << "[init] mode="
              << (cfg.use_interpolated_init ? tensorium_RG::bssn::moving_puncture_interpolated_mode_name()
                                            : "bowen_york")
              << " seed_n=" << cfg.interp_seed_n << std::endl;

    tensorium_RG::bssn::initialize_moving_puncture_data(grid, cfg);

    tensorium_RG::bssn::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;
    tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);
    tensorium_RG::init::zero_z4c_fields(grid);

    tensorium_RG::bssn::apply_boundary_faces(cfg.boundary_faces);
    std::cout << "[bc] reflective_faces="
              << " ix1=" << cfg.boundary_faces.rf_ix1 << " ox1=" << cfg.boundary_faces.rf_ox1
              << " ix2=" << cfg.boundary_faces.rf_ix2 << " ox2=" << cfg.boundary_faces.rf_ox2
              << " ix3=" << cfg.boundary_faces.rf_ix3 << " ox3=" << cfg.boundary_faces.rf_ox3
              << std::endl;
    std::cout << "[bc] rhs_sommerfeld_faces="
              << " ix1=" << cfg.boundary_faces.rhs_ix1 << " ox1=" << cfg.boundary_faces.rhs_ox1
              << " ix2=" << cfg.boundary_faces.rhs_ix2 << " ox2=" << cfg.boundary_faces.rhs_ox2
              << " ix3=" << cfg.boundary_faces.rhs_ix3 << " ox3=" << cfg.boundary_faces.rhs_ox3
              << std::endl;

    tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
        grid, cfg.padding);
    stepper.set_gauge_parameters(cfg.gauge_params);
    stepper.set_state_log_stride(cfg.state_log_stride);
    std::cout << "[log] state_log_stride=" << cfg.state_log_stride << std::endl;
    const size_t export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 10);
    std::cout << "[viz] slice_export_stride=" << export_stride << std::endl;
    stepper.set_snapshot_callback([export_stride](const Grid &g, size_t step) {
        if (step % export_stride == 0) {
            std::cout << ">> Exporting slice " << step << "..." << std::endl;
            export_slice_csv(g, step, "Output/viz");
        }
    });

    tensorium_RG::bssn::CFLControl<double> control;
    control.cfl = cfg.cfl;
    control.gauge_speed = cfg.gauge_speed;

    double t = 0.0;
    for (size_t n = 0; n < cfg.steps; ++n) {
        const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, control, cfg.padding);
        stepper.step(grid, dt, n);
        t += dt;

        if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
            tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);

        std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n + 1, cfg.steps, t);
    }

    std::cout << "[done] steps=" << cfg.steps << " t_final=" << t << std::endl;
    return 0;
}
