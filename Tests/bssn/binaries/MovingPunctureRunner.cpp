#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct PuncturePlaneSample {
    double x_left = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double chi_left = std::numeric_limits<double>::quiet_NaN();
    double alpha_left = std::numeric_limits<double>::quiet_NaN();
    bool   has_left = false;

    double x_right = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
    double chi_right = std::numeric_limits<double>::quiet_NaN();
    double alpha_right = std::numeric_limits<double>::quiet_NaN();
    bool   has_right = false;
};

PuncturePlaneSample sample_puncture_minima(const tensorium_RG::BSSNGridSoA<double> &grid) {
    PuncturePlaneSample sample;

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

            if (!std::isfinite(alpha) || !std::isfinite(chi))
                continue;

            if (x <= 0.0) {
                if (!sample.has_left || alpha < sample.alpha_left) {
                    sample.has_left = true;
                    sample.x_left = x;
                    sample.y_left = y;
                    sample.chi_left = chi;
                    sample.alpha_left = alpha;
                }
            } else {
                if (!sample.has_right || alpha < sample.alpha_right) {
                    sample.has_right = true;
                    sample.x_right = x;
                    sample.y_right = y;
                    sample.chi_right = chi;
                    sample.alpha_right = alpha;
                }
            }
        }
    }

    return sample;
}

} // namespace

int main(int argc, char **argv) {
    using Grid = tensorium_RG::BSSNGridSoA<double>;

    const auto cfg = tensorium_RG::bssn::load_moving_puncture_env();
    size_t output_stride = cfg.state_log_stride;
    size_t slice_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 10);
    size_t constraint_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_EXPORT_STRIDE", 5);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output-stride") == 0 && i + 1 < argc) {
            output_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--slice-export-stride") == 0 && i + 1 < argc) {
            slice_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--constraint-export-stride") == 0 && i + 1 < argc) {
            constraint_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        }
    }

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
    stepper.set_state_log_stride(output_stride);
    std::cout << "[log] state_log_stride=" << output_stride << std::endl;
    std::cout << "[viz] slice_export_stride=" << slice_export_stride << std::endl;
    std::cout << "[constraints] norms_stride=" << constraint_export_stride << std::endl;
    if (slice_export_stride > 0) {
        stepper.set_snapshot_callback([slice_export_stride](const Grid &g, size_t step) {
            if (step % slice_export_stride == 0) {
                std::cout << ">> Exporting slice " << step << "..." << std::endl;
                export_slice_csv(g, step, "Output/viz");
            }
        });
    }

    (void)std::remove("Output/viz/constraints_norms.csv");
    std::ofstream constraint_log("Output/viz/constraints_norms.csv",
                                 std::ios::out | std::ios::trunc);
    if (constraint_log.is_open()) {
        constraint_log.setf(std::ios::unitbuf);
        constraint_log
            << "step,t,dt,l2_theta,l2_Z,l2_H,l2_M,max_H,max_det_drift,max_trace_A,samples\n";
    } else {
        std::cout << "[warn] could not open Output/viz/constraints_norms.csv for writing"
                  << std::endl;
    }

    (void)std::remove("Output/viz/puncture_track.csv");
    std::ofstream puncture_track("Output/viz/puncture_track.csv", std::ios::out | std::ios::trunc);
    if (puncture_track.is_open()) {
        puncture_track.setf(std::ios::unitbuf);
        puncture_track << "step,t,"
                       << "x_left,y_left,chi_left,alpha_left,"
                       << "x_right,y_right,chi_right,alpha_right\n";
    } else {
        std::cout << "[warn] could not open Output/viz/puncture_track.csv for writing"
                  << std::endl;
    }

    (void)std::remove("Output/viz/puncture_track_minima.csv");
    std::ofstream puncture_track_minima("Output/viz/puncture_track_minima.csv",
                                        std::ios::out | std::ios::trunc);
    if (puncture_track_minima.is_open()) {
        puncture_track_minima.setf(std::ios::unitbuf);
        puncture_track_minima << "step,t,"
                              << "x_left,y_left,chi_left,alpha_left,"
                              << "x_right,y_right,chi_right,alpha_right\n";
    } else {
        std::cout << "[warn] could not open Output/viz/puncture_track_minima.csv for writing"
                  << std::endl;
    }

    auto write_puncture_row = [&](std::ofstream &file, size_t step, double time,
                                  const PuncturePlaneSample &sample) {
        if (!file.is_open())
            return;
        file << step << "," << time << ",";
        if (sample.has_left) {
            file << sample.x_left << "," << sample.y_left << "," << sample.chi_left << ","
                 << sample.alpha_left << ",";
        } else {
            file << "nan,nan,nan,nan,";
        }
        if (sample.has_right) {
            file << sample.x_right << "," << sample.y_right << "," << sample.chi_right << ","
                 << sample.alpha_right << "\n";
        } else {
            file << "nan,nan,nan,nan\n";
        }
    };

    size_t current_step = 0;
    double current_time = 0.0;
    double current_dt = 0.0;
    stepper.set_constraint_callback(
        [&](const Grid &, const tensorium_RG::bssn::ConstraintMonitorStats &stats) {
            if (constraint_export_stride == 0 || !constraint_log.is_open() ||
                (current_step % constraint_export_stride) != 0)
                return;
            constraint_log << current_step << "," << current_time << "," << current_dt << ","
                           << stats.l2_theta << "," << stats.l2_Z << "," << stats.l2_H << ","
                           << stats.l2_M << "," << stats.max_H << "," << stats.max_det_drift
                           << "," << stats.max_trace_A << "," << stats.samples << "\n";
        });


    tensorium_RG::bssn::CFLControl<double> control;
    control.cfl = cfg.cfl;
    control.gauge_speed = cfg.gauge_speed;

    double t = 0.0;
    for (size_t n = 0; n < cfg.steps; ++n) {
        const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, control, cfg.padding);
        current_step = n;
        current_dt = dt;
        current_time = t + dt;
        stepper.step(grid, dt, n);
        t += dt;

        const auto puncture_sample = sample_puncture_minima(grid);
        write_puncture_row(puncture_track, n, t, puncture_sample);
        write_puncture_row(puncture_track_minima, n, t, puncture_sample);

        if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
            tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);

        std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n + 1, cfg.steps, t);
    }

    std::cout << "[done] steps=" << cfg.steps << " t_final=" << t << std::endl;
    return 0;
}
