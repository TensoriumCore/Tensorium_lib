/**
 * @file mpi_moving_puncture.cpp
 * @brief MPI-distributed moving puncture binary black hole evolution.
 *
 * Example usage (2 nodes, 64 cores each):
 * @code
 * mpirun -n 2 -ppn 1 --hostfile hosts \
 *   -genv OMP_NUM_THREADS=64 \
 *   -genv KMP_AFFINITY=granularity=fine,balanced \
 *   ./mpi_moving_puncture --nx 128 --steps 1000
 * @endcode
 */

#ifndef TENSORIUM_ENABLE_MPI
#define TENSORIUM_ENABLE_MPI
#endif
#include "Tensorium/MPI/MPI.hpp"
#include "Tensorium/MPI/MPIBSSNRK4.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

using namespace tensorium::mpi;
using namespace tensorium_RG;
using namespace tensorium_RG::bssn;

namespace {

size_t parse_env_stride_or(const char* name, size_t fallback) {
    if (const char* raw = std::getenv(name)) {
        char* end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && parsed >= 0) {
            return static_cast<size_t>(parsed);
        }
    }
    return fallback;
}

void export_xy_midplane_slice_csv(const BSSNGridSoA<double>& grid, const MPIDomain& domain,
                                  size_t step, const std::string& output_dir) {
    const size_t target_k_global = domain.global_nz() / 2;
    const size_t k_begin = domain.global_k0();
    const size_t k_end = domain.global_k0() + domain.local_nz();
    if (target_k_global < k_begin || target_k_global >= k_end) {
        return;
    }

    const size_t k_local = grid.dims.ng + (target_k_global - domain.global_k0());

    std::ostringstream path;
    path << output_dir << "/slice_step_" << std::setw(5) << std::setfill('0') << step
         << "_rank_" << std::setw(4) << std::setfill('0') << domain.rank() << ".csv";

    std::ofstream file(path.str());
    if (!file.is_open()) {
        return;
    }

    file << "rank,global_i,global_j,global_k,x,y,z,alpha,chi,mask\n";
    const size_t ng = grid.dims.ng;
    for (size_t i = ng; i < ng + grid.dims.nx; ++i) {
        for (size_t j = ng; j < ng + grid.dims.ny; ++j) {
            const size_t idx = grid.alpha.idx(i, j, k_local);
            const size_t global_i = domain.global_i0() + (i - ng);
            const size_t global_j = domain.global_j0() + (j - ng);
            const double x = grid.x0 + double(i - ng) * grid.dx;
            const double y = grid.y0 + double(j - ng) * grid.dy;
            const double z = grid.z0 + double(k_local - ng) * grid.dz;
            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            const double mask = (alpha < 0.1) ? 1.0 : 0.0;
            file << domain.rank() << "," << global_i << "," << global_j << "," << target_k_global
                 << "," << x << "," << y << "," << z << "," << alpha << "," << chi << ","
                 << mask << "\n";
        }
    }
}

} // namespace

struct RunConfig {
    size_t global_nx = 96;
    size_t global_ny = 96;
    size_t global_nz = 96;
    size_t ng = 6;
    double box_length = 128.0;
    int spatial_derivative_order = 4;

    int proc_x = 0;
    int proc_y = 0;
    int proc_z = 1;

    size_t nsteps = 1000;
    double cfl = 0.25;
    size_t output_stride = 100;
    size_t slice_export_stride = 0;
    std::string output_dir = "Output/viz/mpi";

    double mass1 = 0.48847892;
    double mass2 = 0.48847892;
    double separation = 6.10679;
    double tangential_momentum = 0.0841746;
    double radial_momentum = -0.000510846;
    bool use_interpolated_init = false;
    size_t interp_seed_n = 64;
    MovingPunctureBoundaryFaces boundary_faces{};
    GaugeParameters<double> gauge_params{};
};

RunConfig make_default_config() {
    RunConfig cfg;
    const auto mp = tensorium_RG::bssn::load_moving_puncture_env();

    cfg.global_nx = mp.nx;
    cfg.global_ny = mp.ny;
    cfg.global_nz = mp.nz;
    cfg.ng = mp.ng;
    cfg.box_length = mp.box_length;
    cfg.spatial_derivative_order = mp.spatial_derivative_order;
    cfg.nsteps = mp.steps;
    cfg.cfl = mp.cfl;
    cfg.output_stride = mp.state_log_stride;
    cfg.slice_export_stride = parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 0);
    cfg.mass1 = mp.mass1;
    cfg.mass2 = mp.mass2;
    cfg.separation = mp.separation * 2.0;
    cfg.tangential_momentum = mp.tangential_momentum;
    cfg.radial_momentum = mp.radial_momentum;
    cfg.use_interpolated_init = mp.use_interpolated_init;
    cfg.interp_seed_n = mp.interp_seed_n;
    cfg.boundary_faces = mp.boundary_faces;
    cfg.gauge_params = mp.gauge_params;

    return cfg;
}

void parse_args(int argc, char** argv, RunConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            cfg.global_nx = cfg.global_ny = cfg.global_nz = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--ny") == 0 && i + 1 < argc) {
            cfg.global_ny = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--nz") == 0 && i + 1 < argc) {
            cfg.global_nz = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cfg.nsteps = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--box") == 0 && i + 1 < argc) {
            cfg.box_length = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--sep") == 0 && i + 1 < argc) {
            cfg.separation = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--momentum") == 0 && i + 1 < argc) {
            cfg.tangential_momentum = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--radial-momentum") == 0 && i + 1 < argc) {
            cfg.radial_momentum = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--cfl") == 0 && i + 1 < argc) {
            cfg.cfl = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--proc-x") == 0 && i + 1 < argc) {
            cfg.proc_x = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--proc-y") == 0 && i + 1 < argc) {
            cfg.proc_y = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--proc-z") == 0 && i + 1 < argc) {
            cfg.proc_z = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-stride") == 0 && i + 1 < argc) {
            cfg.output_stride = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--slice-export-stride") == 0 && i + 1 < argc) {
            cfg.slice_export_stride = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--tp") == 0) {
            cfg.use_interpolated_init = true;
        } else if (strcmp(argv[i], "--no-tp") == 0) {
            cfg.use_interpolated_init = false;
        } else if (strcmp(argv[i], "--interp-seed-n") == 0 && i + 1 < argc) {
            cfg.interp_seed_n = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--spatial-order") == 0 && i + 1 < argc) {
            cfg.spatial_derivative_order = std::stoi(argv[++i]);
        }
    }
}

template <typename T>
void initialize_binary_puncture(BSSNGridSoA<T>& grid, const RunConfig& cfg) {
    const T x1 = T(-0.5) * T(cfg.separation);
    const T x2 = T(0.5) * T(cfg.separation);
    const T y = T(0);
    const T z = T(0);
    const T P1[3] = {T(cfg.radial_momentum), T(-cfg.tangential_momentum), T(0)};
    const T P2[3] = {T(-cfg.radial_momentum), T(cfg.tangential_momentum), T(0)};
    const T S1[3] = {T(0), T(0), T(0)};
    const T S2[3] = {T(0), T(0), T(0)};

    if (cfg.use_interpolated_init) {
        tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init(
            grid, T(cfg.mass1), x1, y, z, P1, S1, T(cfg.mass2), x2, y, z, P2, S2,
            cfg.interp_seed_n, T(1e-10));
    } else {
        tensorium_RG::init::binary_bowen_york_puncture_init(
            grid, T(cfg.mass1), x1, y, z, P1, S1, T(cfg.mass2), x2, y, z, P2, S2, T(1e-10));
    }

    tensorium_RG::init::zero_z4c_fields(grid);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPIContext ctx(argc, argv, ThreadLevel::Funneled);

    // Parse configuration
    RunConfig cfg = make_default_config();
    parse_args(argc, argv, cfg);

    tensorium_RG::fd::set_max_spatial_derivative_order(cfg.spatial_derivative_order);

    // Print configuration from root
    ctx.root_printf("=== MPI Moving Puncture Evolution ===\n");
    ctx.root_printf("Grid: %zu x %zu x %zu, box = %.1f\n",
                    cfg.global_nx, cfg.global_ny, cfg.global_nz, cfg.box_length);
    ctx.root_printf("Binary: m1=%.4f, m2=%.4f, d=%.4f\n",
                    cfg.mass1, cfg.mass2, cfg.separation);
    ctx.root_printf("Init: %s (seed_n=%zu)\n",
                    cfg.use_interpolated_init ? moving_puncture_interpolated_mode_name()
                                              : "bowen_york",
                    cfg.interp_seed_n);
    ctx.root_printf("Steps: %zu, CFL: %.3f\n", cfg.nsteps, cfg.cfl);
    ctx.root_printf("Viz: slice_export_stride=%zu output_dir=%s\n", cfg.slice_export_stride,
                    cfg.output_dir.c_str());
    ctx.root_printf("MPI processes: %d\n", ctx.world_size());
    ctx.root_printf("=====================================\n\n");

    // Create domain decomposition
    DomainConfig domain_cfg;
    domain_cfg.global_nx = cfg.global_nx;
    domain_cfg.global_ny = cfg.global_ny;
    domain_cfg.global_nz = cfg.global_nz;
    domain_cfg.ng = cfg.ng;

    const double half_box = cfg.box_length / 2.0;
    domain_cfg.x0 = -half_box;
    domain_cfg.x1 = half_box;
    domain_cfg.y0 = -half_box;
    domain_cfg.y1 = half_box;
    domain_cfg.z0 = -half_box;
    domain_cfg.z1 = half_box;

    domain_cfg.proc_x = cfg.proc_x;
    domain_cfg.proc_y = cfg.proc_y;
    domain_cfg.proc_z = cfg.proc_z;

    MPIDomain domain(domain_cfg);
    domain.print_info();
    tensorium_RG::fd::set_fd_dx(domain.dx());
    std::filesystem::create_directories(cfg.output_dir);

    // Create local BSSN grid
    auto grid = create_local_bssn_grid<double>(domain);

    // Initialize binary puncture data
    ctx.root_printf("\nInitializing binary puncture data...\n");
    tensorium_RG::bssn::apply_boundary_faces(cfg.boundary_faces);
    initialize_binary_puncture(*grid, cfg);
    tensorium_RG::bssn::apply_boundary_faces(cfg.boundary_faces);

    // Create MPI stepper
    MPIBSSNRKStepper<double> stepper(domain, *grid);

    // Configure gauge parameters
    stepper.set_gauge_parameters(cfg.gauge_params);

    // Set output stride
    stepper.set_state_log_stride(cfg.output_stride);

    if (cfg.slice_export_stride > 0) {
        stepper.set_snapshot_callback(
            [&](const BSSNGridSoA<double>& local_grid, size_t step_index) {
                if (step_index % cfg.slice_export_stride == 0) {
                    export_xy_midplane_slice_csv(local_grid, domain, step_index, cfg.output_dir);
                }
            });
    }

    std::optional<GlobalConstraintStats> last_stats;
    std::ofstream constraints_log;
    if (domain.is_root()) {
        constraints_log.open(cfg.output_dir + "/constraints_norms.csv",
                             std::ios::out | std::ios::trunc);
        if (constraints_log.is_open()) {
            constraints_log.setf(std::ios::unitbuf);
            constraints_log << "step,t,dt,l2_theta,max_hamiltonian,total_samples\n";
        }
    }

    // Set constraint callback
    stepper.set_constraint_callback(
        [&](const BSSNGridSoA<double>&, const GlobalConstraintStats& stats) {
            last_stats = stats;
            printf("  Constraints: L2(Theta)=%.3e, max(H)=%.3e, samples=%zu\n",
                   stats.l2_theta, stats.max_hamiltonian, stats.total_samples);
        });

    // Evolution loop
    ctx.root_printf("\nStarting evolution...\n\n");
    domain.barrier();

    double t = 0.0;
    for (size_t step = 0; step < cfg.nsteps; ++step) {
        // Compute global time step
        double dt = stepper.compute_dt(*grid, cfg.cfl);

        // Take RK4 step
        stepper.step(*grid, dt, step);

        t += dt;

        if (domain.is_root() && last_stats && constraints_log.is_open()) {
            constraints_log << step << "," << t << "," << dt << "," << last_stats->l2_theta << ","
                            << last_stats->max_hamiltonian << "," << last_stats->total_samples
                            << "\n";
            last_stats.reset();
        }

        // Progress output
        if (domain.is_root() && (step + 1) % cfg.output_stride == 0) {
            printf("Step %zu/%zu complete, t = %.4f, dt = %.6f\n",
                   step + 1, cfg.nsteps, t, dt);
            fflush(stdout);
        }
    }

    // Final output
    domain.barrier();
    ctx.root_printf("\n=== Evolution complete ===\n");
    ctx.root_printf("Final time: t = %.4f\n", t);
    ctx.root_printf("==========================\n");

    return 0;
}
