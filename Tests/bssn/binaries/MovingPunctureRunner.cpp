#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;
using MovingPunctureHierarchy =
    tensorium_RG::bssn::fmr::FixedMeshRefinementHierarchy<double, tensorium_RG::bssn::BoundaryRadiative>;
using FineBoundary = MovingPunctureHierarchy::FineBoundary;

void export_slice_csv(const Grid &grid, size_t step, const std::string &output_dir) {
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

struct ConstraintScratch {
    tensorium_RG::Field3D<double> H;
    tensorium_RG::Field3D<double> M[3];
    tensorium_RG::Field3D<double> C[3];

    explicit ConstraintScratch(const Grid &grid) : H(tensorium_RG::make_field(grid.alpha.st)) {
        for (int q = 0; q < 3; ++q) {
            M[q] = tensorium_RG::make_field(grid.alpha.st);
            C[q] = tensorium_RG::make_field(grid.alpha.st);
        }
    }
};

tensorium_RG::bssn::ConstraintMonitorStats
compute_constraint_stats(Grid &grid, ConstraintScratch &scratch, size_t padding) {
    tensorium_RG::fd::set_fd_dx(grid.dx);
    const double min_extent =
        std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                  (grid.dims.nz - 1) * grid.dz});
    const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
    const double r_max = 0.45 * min_extent;
    tensorium_RG::bssn::compute_bssn_constraints(grid, grid.Ricci, scratch.H, scratch.M, scratch.C,
                                                 r_min, r_max, 0.0, 0.0, 0.0);
    auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, scratch.H, padding);
    tensorium_RG::bssn::populate_constraint_norms(grid, scratch.M, stats, padding);
    return stats;
}

void initialize_moving_puncture_level(Grid &grid, const tensorium_RG::bssn::MovingPunctureEnvConfig &cfg,
                                      const tensorium_RG::bssn::ProjectionConfig &proj_cfg,
                                      bool zero_z4c = true) {
    tensorium_RG::bssn::initialize_moving_puncture_data(grid, cfg);
    tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);
    if (zero_z4c)
        tensorium_RG::init::zero_z4c_fields(grid);
}

Grid make_grid_like(const Grid &grid) {
    Grid out(grid.dims.nx, grid.dims.ny, grid.dims.nz, grid.dims.ng, grid.dx, grid.dy, grid.dz);
    out.x0 = grid.x0;
    out.y0 = grid.y0;
    out.z0 = grid.z0;
    return out;
}

double moving_puncture_grid_half_width(const Grid &grid) {
    const auto axis_half_width = [](double origin, double spacing, size_t cells) {
        const double xmin = std::abs(origin - 0.5 * spacing);
        const double xmax = std::abs(origin + (double(cells) - 0.5) * spacing);
        return std::min(xmin, xmax);
    };

    return std::min({axis_half_width(grid.x0, grid.dx, grid.dims.nx),
                     axis_half_width(grid.y0, grid.dy, grid.dims.ny),
                     axis_half_width(grid.z0, grid.dz, grid.dims.nz)});
}

struct MovingPunctureInitBlendRegion {
    double core_half_width = 0.0;
    double transition_width = 0.0;
    double tp_outer_half_width = 0.0;
};

MovingPunctureInitBlendRegion
moving_puncture_init_blend_region(const Grid &grid,
                                  const tensorium_RG::bssn::MovingPunctureEnvConfig &cfg) {
    const double max_half_width =
        std::max(0.0, moving_puncture_grid_half_width(grid) - 0.5 * std::max({grid.dx, grid.dy, grid.dz}));
    const double physical_buffer =
        (cfg.fmr.puncture_buffer > 0.0) ? cfg.fmr.puncture_buffer : std::max(2.0, 4.0 * cfg.spacing);
    const double requested_transition = (cfg.fmr.init_transition_width > 0.0)
                                            ? cfg.fmr.init_transition_width
                                            : std::max(0.5 * physical_buffer, 6.0 * std::max({grid.dx, grid.dy, grid.dz}));

    const double desired_outer_half_width =
        (cfg.fmr.init_core_half_width > 0.0) ? (cfg.fmr.init_core_half_width + requested_transition)
                                             : (cfg.separation + physical_buffer);
    const double tp_outer_half_width = std::clamp(desired_outer_half_width, 0.0, max_half_width);
    const double requested_core_half_width =
        (cfg.fmr.init_core_half_width > 0.0) ? cfg.fmr.init_core_half_width
                                             : std::max(0.0, tp_outer_half_width - requested_transition);
    const double core_half_width = std::clamp(requested_core_half_width, 0.0, tp_outer_half_width);

    MovingPunctureInitBlendRegion region;
    region.core_half_width = core_half_width;
    region.transition_width = std::max(0.0, tp_outer_half_width - core_half_width);
    region.tp_outer_half_width = tp_outer_half_width;
    return region;
}

void initialize_moving_puncture_hierarchy(MovingPunctureHierarchy &hierarchy,
                                          const tensorium_RG::bssn::MovingPunctureEnvConfig &cfg,
                                          const tensorium_RG::bssn::ProjectionConfig &proj_cfg) {
    initialize_moving_puncture_level(hierarchy.root_grid(), cfg, proj_cfg, true);
    tensorium_RG::fd::set_fd_dx(hierarchy.root_grid().dx);
    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(hierarchy.root_grid());

    for (size_t level = 1; level < hierarchy.num_levels(); ++level) {
        hierarchy.prolongate_level_from_parent(level);
        tensorium_RG::bssn::project_bssn_state(hierarchy.level_grid(level), proj_cfg);

        if (cfg.fmr.fine_levels_use_parent_init) {
            hierarchy.apply_level_boundaries();
            std::cout << "[fmr.init] level=" << level << " source=parent_prolongation"
                      << std::endl;
            continue;
        }

        Grid tp_reference = make_grid_like(hierarchy.level_grid(level));
        initialize_moving_puncture_level(tp_reference, cfg, proj_cfg, true);

        const auto region = moving_puncture_init_blend_region(hierarchy.level_grid(level), cfg);
        hierarchy.blend_level_centered_core_from_reference(level, tp_reference,
                                                           region.core_half_width,
                                                           region.transition_width);
        tensorium_RG::bssn::project_bssn_state(hierarchy.level_grid(level), proj_cfg);
        hierarchy.apply_level_boundaries();

        std::cout << "[fmr.init] level=" << level << " core_half_width=" << region.core_half_width
                  << " transition_width=" << region.transition_width
                  << " tp_outer_half_width=" << region.tp_outer_half_width << std::endl;
    }

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

void project_hierarchy_levels(MovingPunctureHierarchy &hierarchy,
                              const tensorium_RG::bssn::ProjectionConfig &proj_cfg) {
    for (size_t level = 0; level < hierarchy.num_levels(); ++level)
        tensorium_RG::bssn::project_bssn_state(hierarchy.level_grid(level), proj_cfg);

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

PuncturePlaneSample sample_puncture_minima(const Grid &grid) {
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

    Grid root_grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing, cfg.spacing);
    tensorium_RG::bssn::center_cell_centered_origin(root_grid);
    const auto level_cfgs = tensorium_RG::bssn::build_moving_puncture_fmr_levels(root_grid, cfg);
    const bool use_fmr = !level_cfgs.empty();

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

    tensorium_RG::bssn::apply_boundary_configuration(cfg);

    tensorium_RG::bssn::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;

    std::unique_ptr<MovingPunctureHierarchy> hierarchy;
    Grid                                    *root_state = &root_grid;
    Grid                                    *puncture_state = &root_grid;
    if (use_fmr) {
        hierarchy = std::make_unique<MovingPunctureHierarchy>(root_grid, level_cfgs, cfg.padding);
        initialize_moving_puncture_hierarchy(*hierarchy, cfg, proj_cfg);
        root_state = &hierarchy->root_grid();
        puncture_state = &hierarchy->level_grid(hierarchy->num_levels() - 1);
    } else {
        initialize_moving_puncture_level(root_grid, cfg, proj_cfg, true);
        tensorium_RG::fd::set_fd_dx(root_grid.dx);
        tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root_grid);
    }

    auto params = cfg.gauge_params;
    const auto bc_char =
        tensorium_RG::bssn::configure_boundary_characteristics_from_state(*root_state, cfg, params);

    std::cout << "[bc] allow_reflective=" << cfg.allow_reflective_bc
              << " fail_on_gauge_bc_mismatch=" << cfg.fail_on_gauge_bc_mismatch
              << " sponge_enable=" << cfg.sponge.enabled
              << " sponge_width=" << cfg.sponge.width
              << " sponge_strength=" << cfg.sponge.strength
              << " sponge_exponent=" << cfg.sponge.exponent
              << " radiative_collar_width=" << cfg.radiative_collar_width
              << " ko_boundary_width=" << cfg.ko_boundary_width
              << " ko_boundary_floor=" << cfg.ko_boundary_floor << std::endl;
    for (int axis = 0; axis < 3; ++axis) {
        std::cout << "[bc] "
                  << tensorium_RG::bssn::describe_boundary_face_mode(
                         cfg.boundary_faces, cfg.sponge, axis, false)
                  << std::endl;
        std::cout << "[bc] "
                  << tensorium_RG::bssn::describe_boundary_face_mode(
                         cfg.boundary_faces, cfg.sponge, axis, true)
                  << std::endl;
    }
    tensorium_RG::bssn::report_boundary_characteristics(
        params, cfg.fail_on_gauge_bc_mismatch, bc_char);

    if (use_fmr) {
        std::cout << "[fmr] enabled=1 levels=" << (hierarchy->num_levels() - 1)
                  << " ratio=" << cfg.fmr.refinement_ratio
                  << " finest_dx=" << puncture_state->dx << std::endl;
        for (size_t level = 1; level < hierarchy->num_levels(); ++level) {
            const Grid &g = hierarchy->level_grid(level);
            std::cout << "[fmr] level=" << level << " nx=" << g.dims.nx << " ny=" << g.dims.ny
                      << " nz=" << g.dims.nz << " spacing=" << g.dx << " box=("
                      << g.dims.nx * g.dx << ", " << g.dims.ny * g.dy << ", "
                      << g.dims.nz * g.dz << ")\n";
        }
    } else {
        std::cout << "[fmr] enabled=0" << std::endl;
    }

    std::cout << "[log] state_log_stride=" << output_stride << std::endl;
    std::cout << "[viz] slice_export_stride=" << slice_export_stride << std::endl;
    std::cout << "[constraints] norms_stride=" << constraint_export_stride << std::endl;

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

    tensorium_RG::bssn::CFLControl<double> control;
    control.cfl = cfg.cfl;
    control.gauge_speed = cfg.gauge_speed;

    double t = 0.0;
    if (use_fmr) {
        hierarchy->set_gauge_parameters(params);
        hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
        ConstraintScratch constraint_scratch(*root_state);

        for (size_t n = 0; n < cfg.steps; ++n) {
            const double dt = hierarchy->compute_dt(control);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            hierarchy->step(dt);
            t += dt;

            if (constraint_export_stride > 0 && constraint_log.is_open() &&
                (current_step % constraint_export_stride) == 0) {
                auto stats = compute_constraint_stats(*root_state, constraint_scratch, cfg.padding);
                constraint_log << current_step << "," << current_time << "," << current_dt << ","
                               << stats.l2_theta << "," << stats.l2_Z << "," << stats.l2_H << ","
                               << stats.l2_M << "," << stats.max_H << "," << stats.max_det_drift
                               << "," << stats.max_trace_A << "," << stats.samples << "\n";
            }

            if (slice_export_stride > 0 && (n % slice_export_stride) == 0) {
                std::cout << ">> Exporting finest slice " << n << "..." << std::endl;
                export_slice_csv(*puncture_state, n, "Output/viz");
            }

            const auto puncture_sample = sample_puncture_minima(*puncture_state);
            write_puncture_row(puncture_track, n, t, puncture_sample);
            write_puncture_row(puncture_track_minima, n, t, puncture_sample);

            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
                project_hierarchy_levels(*hierarchy, proj_cfg);

            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n + 1, cfg.steps, t);
        }
    } else {
        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            *root_state, cfg.padding);
        stepper.set_gauge_parameters(params);
        stepper.set_state_log_stride(output_stride);
        if (slice_export_stride > 0) {
            stepper.set_snapshot_callback([slice_export_stride](const Grid &g, size_t step) {
                if (step % slice_export_stride == 0) {
                    std::cout << ">> Exporting slice " << step << "..." << std::endl;
                    export_slice_csv(g, step, "Output/viz");
                }
            });
        }
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

        for (size_t n = 0; n < cfg.steps; ++n) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(*root_state, control, cfg.padding);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            stepper.step(*root_state, dt, n);
            t += dt;

            const auto puncture_sample = sample_puncture_minima(*root_state);
            write_puncture_row(puncture_track, n, t, puncture_sample);
            write_puncture_row(puncture_track_minima, n, t, puncture_sample);

            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
                tensorium_RG::bssn::project_bssn_state(*root_state, proj_cfg);

            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f\n", dt, n + 1, cfg.steps, t);
        }
    }

    std::cout << "[done] steps=" << cfg.steps << " t_final=" << t << std::endl;
    return 0;
}
