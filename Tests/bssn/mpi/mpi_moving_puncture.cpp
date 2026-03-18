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

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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

struct PuncturePlaneSample {
    double x_left = 0.0;
    double y_left = 0.0;
    double chi_left = std::numeric_limits<double>::infinity();
    double alpha_left = std::numeric_limits<double>::infinity();
    bool has_left = false;

    double x_right = 0.0;
    double y_right = 0.0;
    double chi_right = std::numeric_limits<double>::infinity();
    double alpha_right = std::numeric_limits<double>::infinity();
    bool has_right = false;
};

struct PunctureCandidate {
    size_t i = 0;
    size_t j = 0;
    double x = 0.0;
    double y = 0.0;
    double chi = std::numeric_limits<double>::infinity();
    double alpha = std::numeric_limits<double>::infinity();
    bool found = false;
};

struct PunctureTrackHistory {
    PuncturePlaneSample prev{};
    PuncturePlaneSample last{};
    double t_prev = 0.0;
    double t_last = 0.0;
    bool has_prev = false;
    bool has_last = false;
};

struct PuncturePrediction {
    double x_left = 0.0;
    double y_left = 0.0;
    double x_right = 0.0;
    double y_right = 0.0;
    bool valid = false;
};

PuncturePrediction predict_puncture_positions(const PunctureTrackHistory& history,
                                             double current_time) {
    PuncturePrediction pred;
    if (!history.has_last || !history.last.has_left || !history.last.has_right) {
        return pred;
    }

    pred.x_left = history.last.x_left;
    pred.y_left = history.last.y_left;
    pred.x_right = history.last.x_right;
    pred.y_right = history.last.y_right;

    if (history.has_prev && history.prev.has_left && history.prev.has_right) {
        const double dt_prev = history.t_last - history.t_prev;
        const double dt_now = current_time - history.t_last;
        if (dt_prev > 1e-12 && std::isfinite(dt_prev) && dt_now >= 0.0 && std::isfinite(dt_now)) {
            const double ratio = std::min(dt_now / dt_prev, 1.5);
            pred.x_left += ratio * (history.last.x_left - history.prev.x_left);
            pred.y_left += ratio * (history.last.y_left - history.prev.y_left);
            pred.x_right += ratio * (history.last.x_right - history.prev.x_right);
            pred.y_right += ratio * (history.last.y_right - history.prev.y_right);
        }
    }

    pred.valid = true;
    return pred;
}

bool clamp_jump_to_radius(double x_ref, double y_ref, double max_jump, double& x, double& y) {
    const double dx = x - x_ref;
    const double dy = y - y_ref;
    const double r = std::sqrt(dx * dx + dy * dy);
    if (!std::isfinite(r) || r <= max_jump || max_jump <= 0.0) {
        return false;
    }
    const double s = max_jump / r;
    x = x_ref + s * dx;
    y = y_ref + s * dy;
    return true;
}

double parabolic_subcell_offset(double fm, double f0, double fp) {
    const double denom = fm - 2.0 * f0 + fp;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-30) {
        return 0.0;
    }
    double offset = 0.5 * (fm - fp) / denom;
    if (!std::isfinite(offset)) {
        return 0.0;
    }
    if (offset > 0.5) {
        offset = 0.5;
    }
    if (offset < -0.5) {
        offset = -0.5;
    }
    return offset;
}

size_t nearest_index(double x, double origin, double spacing, size_t lo, size_t hi_inclusive) {
    const long idx = std::lround((x - origin) / spacing + double(lo));
    const long clamped = std::max<long>(static_cast<long>(lo),
                                        std::min<long>(static_cast<long>(hi_inclusive), idx));
    return static_cast<size_t>(clamped);
}

PunctureCandidate find_min_candidate_in_window(const BSSNGridSoA<double>& grid, size_t k_local,
                                               size_t i_lo, size_t i_hi_exclusive, size_t j_lo,
                                               size_t j_hi_exclusive, bool use_exclusion = false,
                                               double ex_x = 0.0, double ex_y = 0.0,
                                               double ex_r2 = 0.0, bool use_anchor = false,
                                               double x_ref = 0.0, double y_ref = 0.0,
                                               double dist_weight = 0.0,
                                               double alpha_weight = 0.0) {
    PunctureCandidate out;
    double best_score = std::numeric_limits<double>::infinity();
    const double inv_dx = 1.0 / std::max(grid.dx, 1e-14);
    const double inv_dy = 1.0 / std::max(grid.dy, 1e-14);

    for (size_t i = i_lo; i < i_hi_exclusive; ++i) {
        for (size_t j = j_lo; j < j_hi_exclusive; ++j) {
            const size_t idx = grid.alpha.idx(i, j, k_local);
            const double x = grid.x0 + double(i - grid.dims.ng) * grid.dx;
            const double y = grid.y0 + double(j - grid.dims.ng) * grid.dy;
            if (use_exclusion) {
                const double dx = x - ex_x;
                const double dy = y - ex_y;
                if (dx * dx + dy * dy < ex_r2) {
                    continue;
                }
            }

            const double chi = grid.chi.ptr()[idx];
            const double alpha = grid.alpha.ptr()[idx];
            if (!std::isfinite(chi) || !std::isfinite(alpha)) {
                continue;
            }

            double score = chi + alpha_weight * alpha;
            if (use_anchor) {
                const double dx_cells = (x - x_ref) * inv_dx;
                const double dy_cells = (y - y_ref) * inv_dy;
                score += dist_weight * (dx_cells * dx_cells + dy_cells * dy_cells);
            }

            if (!out.found || score < best_score ||
                (score == best_score && (chi < out.chi || (chi == out.chi && alpha < out.alpha)))) {
                out.found = true;
                out.i = i;
                out.j = j;
                out.x = x;
                out.y = y;
                out.chi = chi;
                out.alpha = alpha;
                best_score = score;
            }
        }
    }
    return out;
}

void refine_candidate_xy(const BSSNGridSoA<double>& grid, PunctureCandidate& cand, size_t k_local) {
    if (!cand.found) {
        return;
    }

    const size_t ng = grid.dims.ng;
    const size_t i_min = ng;
    const size_t i_max = ng + grid.dims.nx - 1;
    const size_t j_min = ng;
    const size_t j_max = ng + grid.dims.ny - 1;
    const size_t i = cand.i;
    const size_t j = cand.j;
    const size_t cidx = grid.alpha.idx(i, j, k_local);

    const double x_center = grid.x0 + double(i - ng) * grid.dx;
    const double y_center = grid.y0 + double(j - ng) * grid.dy;
    double dx_sub = 0.0;
    double dy_sub = 0.0;

    if (i > i_min && i < i_max) {
        const double chim = grid.chi.ptr()[grid.alpha.idx(i - 1, j, k_local)];
        const double chi0 = grid.chi.ptr()[cidx];
        const double chip = grid.chi.ptr()[grid.alpha.idx(i + 1, j, k_local)];
        dx_sub = parabolic_subcell_offset(chim, chi0, chip);
    }
    if (j > j_min && j < j_max) {
        const double chim = grid.chi.ptr()[grid.alpha.idx(i, j - 1, k_local)];
        const double chi0 = grid.chi.ptr()[cidx];
        const double chip = grid.chi.ptr()[grid.alpha.idx(i, j + 1, k_local)];
        dy_sub = parabolic_subcell_offset(chim, chi0, chip);
    }

    cand.x = x_center + dx_sub * grid.dx;
    cand.y = y_center + dy_sub * grid.dy;
}

PuncturePlaneSample sample_local_equatorial_minima(const BSSNGridSoA<double>& grid,
                                                   const MPIDomain& domain,
                                                   const PunctureTrackHistory* history,
                                                   double current_time) {
    PuncturePlaneSample sample;
    const size_t target_k_global = domain.global_nz() / 2;
    const size_t k_begin = domain.global_k0();
    const size_t k_end = domain.global_k0() + domain.local_nz();
    if (target_k_global < k_begin || target_k_global >= k_end) {
        return sample;
    }
    if (domain.local_nx() != domain.global_nx() || domain.local_ny() != domain.global_ny()) {
        return sample;
    }

    const size_t ng = grid.dims.ng;
    const size_t i_min = ng;
    const size_t i_max = ng + grid.dims.nx - 1;
    const size_t j_min = ng;
    const size_t j_max = ng + grid.dims.ny - 1;
    const size_t k_local = ng + (target_k_global - domain.global_k0());
    const double h = std::max(grid.dx, grid.dy);
    const bool have_prev = (history != nullptr && history->has_last && history->last.has_left &&
                            history->last.has_right && std::isfinite(history->last.x_left) &&
                            std::isfinite(history->last.y_left) &&
                            std::isfinite(history->last.x_right) &&
                            std::isfinite(history->last.y_right));
    const PuncturePrediction pred =
        (history != nullptr) ? predict_puncture_positions(*history, current_time)
                             : PuncturePrediction{};

    constexpr size_t kLocalRadiusCells = 14;
    constexpr size_t kDistinctRadiusCells = 5;
    constexpr double kDistWeight = 3e-3;
    constexpr double kAlphaWeight = 5e-2;

    auto local_search = [&](double x_ref, double y_ref, bool with_exclusion = false,
                            double ex_x = 0.0, double ex_y = 0.0,
                            double ex_r2 = 0.0) -> PunctureCandidate {
        const size_t ic = nearest_index(x_ref, grid.x0, grid.dx, i_min, i_max);
        const size_t jc = nearest_index(y_ref, grid.y0, grid.dy, j_min, j_max);
        const size_t i_lo = (ic > kLocalRadiusCells) ? ic - kLocalRadiusCells : i_min;
        const size_t j_lo = (jc > kLocalRadiusCells) ? jc - kLocalRadiusCells : j_min;
        const size_t i_hi = std::min(i_max, ic + kLocalRadiusCells) + 1;
        const size_t j_hi = std::min(j_max, jc + kLocalRadiusCells) + 1;
        return find_min_candidate_in_window(grid, k_local, i_lo, i_hi, j_lo, j_hi, with_exclusion,
                                            ex_x, ex_y, ex_r2, true, x_ref, y_ref, kDistWeight,
                                            kAlphaWeight);
    };

    PunctureCandidate a;
    PunctureCandidate b;
    bool have_pair = false;

    if (pred.valid) {
        a = local_search(pred.x_left, pred.y_left);
        const double ex_r = double(kDistinctRadiusCells) * h;
        b = local_search(pred.x_right, pred.y_right, true, a.x, a.y, ex_r * ex_r);
        have_pair = (a.found && b.found);
    }

    if (!have_pair && have_prev) {
        a = local_search(history->last.x_left, history->last.y_left);
        const double ex_r = double(kDistinctRadiusCells) * h;
        b = local_search(history->last.x_right, history->last.y_right, true, a.x, a.y,
                         ex_r * ex_r);
        have_pair = (a.found && b.found);
    }

    if (!have_pair) {
        a = find_min_candidate_in_window(grid, k_local, i_min, i_max + 1, j_min, j_max + 1);
        if (a.found) {
            const double ex_r = double(kDistinctRadiusCells) * h;
            b = find_min_candidate_in_window(grid, k_local, i_min, i_max + 1, j_min, j_max + 1,
                                             true, a.x, a.y, ex_r * ex_r);
            have_pair = b.found;
        }
    }

    if (!have_pair) {
        return sample;
    }

    refine_candidate_xy(grid, a, k_local);
    refine_candidate_xy(grid, b, k_local);

    PunctureCandidate left = a;
    PunctureCandidate right = b;
    if (have_prev) {
        const auto dist = [](double xr, double yr, const PunctureCandidate& c) {
            const double dx = c.x - xr;
            const double dy = c.y - yr;
            return std::sqrt(dx * dx + dy * dy);
        };
        const double xL_ref = pred.valid ? pred.x_left : history->last.x_left;
        const double yL_ref = pred.valid ? pred.y_left : history->last.y_left;
        const double xR_ref = pred.valid ? pred.x_right : history->last.x_right;
        const double yR_ref = pred.valid ? pred.y_right : history->last.y_right;
        const double keep = dist(xL_ref, yL_ref, a) + dist(xR_ref, yR_ref, b);
        const double swap = dist(xL_ref, yL_ref, b) + dist(xR_ref, yR_ref, a);
        if (swap < keep) {
            left = b;
            right = a;
        }

        const double max_jump = 6.0 * h;
        (void)clamp_jump_to_radius(history->last.x_left, history->last.y_left, max_jump, left.x,
                                   left.y);
        (void)clamp_jump_to_radius(history->last.x_right, history->last.y_right, max_jump,
                                   right.x, right.y);
    } else if (a.x > b.x) {
        left = b;
        right = a;
    }

    if (left.found) {
        sample.has_left = true;
        sample.x_left = left.x;
        sample.y_left = left.y;
        sample.chi_left = left.chi;
        sample.alpha_left = left.alpha;
    }
    if (right.found) {
        sample.has_right = true;
        sample.x_right = right.x;
        sample.y_right = right.y;
        sample.chi_right = right.chi;
        sample.alpha_right = right.alpha;
    }
    return sample;
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
    size_t padding = 0;
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
    MovingPunctureSpongeConfig sponge{};
    size_t radiative_collar_width = 4;
    size_t ko_boundary_width = 4;
    double ko_boundary_floor = 0.0;
    bool allow_reflective_bc = false;
    GaugeParameters<double> gauge_params{};
};

RunConfig make_default_config() {
    RunConfig cfg;
    const auto mp = tensorium_RG::bssn::load_moving_puncture_env();

    cfg.global_nx = mp.nx;
    cfg.global_ny = mp.ny;
    cfg.global_nz = mp.nz;
    cfg.ng = mp.ng;
    cfg.padding = mp.padding;
    cfg.box_length = mp.box_length;
    cfg.spatial_derivative_order = mp.spatial_derivative_order;
    cfg.nsteps = mp.steps;
    cfg.cfl = mp.cfl;
    cfg.output_stride = mp.state_log_stride;
    cfg.slice_export_stride = parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 0);
    cfg.mass1 = mp.mass1;
    cfg.mass2 = mp.mass2;
    cfg.separation = mp.separation;
    cfg.tangential_momentum = mp.tangential_momentum;
    cfg.radial_momentum = mp.radial_momentum;
    cfg.use_interpolated_init = mp.use_interpolated_init;
    cfg.interp_seed_n = mp.interp_seed_n;
    cfg.boundary_faces = mp.boundary_faces;
    cfg.sponge = mp.sponge;
    cfg.radiative_collar_width = mp.radiative_collar_width;
    cfg.ko_boundary_width = mp.ko_boundary_width;
    cfg.ko_boundary_floor = mp.ko_boundary_floor;
    cfg.allow_reflective_bc = mp.allow_reflective_bc;
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
        } else if (strcmp(argv[i], "--padding") == 0 && i + 1 < argc) {
            cfg.padding = std::stoul(argv[++i]);
        }
    }
}

template <typename T>
void initialize_binary_puncture(BSSNGridSoA<T>& grid, const RunConfig& cfg) {
    const T S1[3] = {T(0), T(0), T(0)};
    const T S2[3] = {T(0), T(0), T(0)};

    if (cfg.use_interpolated_init) {
        const T x1 = -T(cfg.separation);
        const T x2 = T(cfg.separation);
        const T y = T(0);
        const T z = T(0);
        const T P1[3] = {T(cfg.radial_momentum), T(-cfg.tangential_momentum), T(0)};
        const T P2[3] = {T(-cfg.radial_momentum), T(cfg.tangential_momentum), T(0)};

        tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init(
            grid, T(cfg.mass1), x1, y, z, P1, S1, T(cfg.mass2), x2, y, z, P2, S2,
            cfg.interp_seed_n, T(1e-10));
    } else {
        const T x = T(0);
        const T y1 = T(cfg.separation);
        const T y2 = -T(cfg.separation);
        const T z = T(0);
        const T P1[3] = {T(-cfg.tangential_momentum), T(-cfg.radial_momentum), T(0)};
        const T P2[3] = {T(cfg.tangential_momentum), T(cfg.radial_momentum), T(0)};

        tensorium_RG::init::binary_bowen_york_puncture_init(
            grid, T(cfg.mass1), x, y1, z, P1, S1, T(cfg.mass2), x, y2, z, P2, S2, T(1e-10));
    }

    tensorium_RG::bssn::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;
    tensorium_RG::bssn::project_bssn_state(grid, proj_cfg);
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
    ctx.root_printf("BC: allow_reflective=%d sponge=%d width=%zu strength=%.3f exponent=%.3f "
                    "radiative_collar_width=%zu ko_boundary_width=%zu ko_boundary_floor=%.3f\n",
                    cfg.allow_reflective_bc ? 1 : 0, cfg.sponge.enabled ? 1 : 0, cfg.sponge.width,
                    cfg.sponge.strength, cfg.sponge.exponent, cfg.radiative_collar_width,
                    cfg.ko_boundary_width,
                    cfg.ko_boundary_floor);
    for (int axis = 0; axis < 3; ++axis) {
        ctx.root_printf("BC: %s\n",
                        tensorium_RG::bssn::describe_boundary_face_mode(
                            cfg.boundary_faces, cfg.sponge, axis, false)
                            .c_str());
        ctx.root_printf("BC: %s\n",
                        tensorium_RG::bssn::describe_boundary_face_mode(
                            cfg.boundary_faces, cfg.sponge, axis, true)
                            .c_str());
    }
    ctx.root_printf("MPI processes: %d\n", ctx.world_size());
    ctx.root_printf("=====================================\n\n");

    // Create domain decomposition
    DomainConfig domain_cfg;
    domain_cfg.global_nx = cfg.global_nx;
    domain_cfg.global_ny = cfg.global_ny;
    domain_cfg.global_nz = cfg.global_nz;
    domain_cfg.ng = cfg.ng;

    const double half_box = cfg.box_length / 2.0;
    const double dx = cfg.box_length / static_cast<double>(cfg.global_nx);
    const double dy = cfg.box_length / static_cast<double>(cfg.global_ny);
    const double dz = cfg.box_length / static_cast<double>(cfg.global_nz);
    domain_cfg.x0 = -half_box + 0.5 * dx;
    domain_cfg.x1 = half_box + 0.5 * dx;
    domain_cfg.y0 = -half_box + 0.5 * dy;
    domain_cfg.y1 = half_box + 0.5 * dy;
    domain_cfg.z0 = -half_box + 0.5 * dz;
    domain_cfg.z1 = half_box + 0.5 * dz;

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
    tensorium_RG::bssn::BoundaryRadiative::set_rhs_collar_width(cfg.radiative_collar_width);
    tensorium_RG::bssn::BoundaryRadiative::set_sponge(cfg.sponge.enabled, cfg.sponge.width,
                                                      cfg.sponge.strength, cfg.sponge.exponent);
    tensorium_RG::bssn::configure_ko_boundary_taper(cfg.ko_boundary_width,
                                                    cfg.ko_boundary_floor);
    initialize_binary_puncture(*grid, cfg);
    tensorium_RG::bssn::apply_boundary_faces(cfg.boundary_faces);

    // Create MPI stepper
    MPIBSSNRKStepper<double> stepper(domain, *grid, cfg.padding);

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
    std::ofstream puncture_minima_log;
    PunctureTrackHistory puncture_minima_history;
    if (domain.is_root()) {
        constraints_log.open(cfg.output_dir + "/constraints_norms.csv",
                             std::ios::out | std::ios::trunc);
        if (constraints_log.is_open()) {
            constraints_log.setf(std::ios::unitbuf);
            constraints_log << "step,t,dt,l2_theta,max_hamiltonian,total_samples\n";
        }
    }
    if (cfg.slice_export_stride > 0 && domain.local_nx() == domain.global_nx() &&
        domain.local_ny() == domain.global_ny() &&
        domain.global_nz() / 2 >= domain.global_k0() &&
        domain.global_nz() / 2 < domain.global_k0() + domain.local_nz()) {
        puncture_minima_log.open(cfg.output_dir + "/puncture_track_minima.csv",
                                 std::ios::out | std::ios::trunc);
        if (puncture_minima_log.is_open()) {
            puncture_minima_log.setf(std::ios::unitbuf);
            puncture_minima_log << "step,t,x_left,y_left,chi_left,alpha_left,"
                                  "x_right,y_right,chi_right,alpha_right\n";
            puncture_minima_history.last.has_left = true;
            puncture_minima_history.last.has_right = true;
            if (cfg.use_interpolated_init) {
                puncture_minima_history.last.x_left = -cfg.separation;
                puncture_minima_history.last.y_left = 0.0;
                puncture_minima_history.last.x_right = cfg.separation;
                puncture_minima_history.last.y_right = 0.0;
            } else {
                puncture_minima_history.last.x_left = 0.0;
                puncture_minima_history.last.y_left = -cfg.separation;
                puncture_minima_history.last.x_right = 0.0;
                puncture_minima_history.last.y_right = cfg.separation;
            }
            puncture_minima_history.t_last = 0.0;
            puncture_minima_history.has_last = true;
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

        if (cfg.slice_export_stride > 0 && step % cfg.slice_export_stride == 0 &&
            puncture_minima_log.is_open()) {
            const auto punctures =
                sample_local_equatorial_minima(*grid, domain, &puncture_minima_history, t);
            if (punctures.has_left && punctures.has_right) {
                if (puncture_minima_history.has_last) {
                    puncture_minima_history.prev = puncture_minima_history.last;
                    puncture_minima_history.t_prev = puncture_minima_history.t_last;
                    puncture_minima_history.has_prev = true;
                }
                puncture_minima_history.last = punctures;
                puncture_minima_history.t_last = t;
                puncture_minima_history.has_last = true;
            }
            puncture_minima_log << step << "," << t << ","
                                << (punctures.has_left ? punctures.x_left
                                                       : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_left ? punctures.y_left
                                                       : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_left ? punctures.chi_left
                                                       : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_left ? punctures.alpha_left
                                                       : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_right ? punctures.x_right
                                                        : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_right ? punctures.y_right
                                                        : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_right ? punctures.chi_right
                                                        : std::numeric_limits<double>::quiet_NaN())
                                << ","
                                << (punctures.has_right ? punctures.alpha_right
                                                        : std::numeric_limits<double>::quiet_NaN())
                                << "\n";
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
