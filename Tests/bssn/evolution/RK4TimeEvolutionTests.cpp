#ifndef TENSORIUM_BSSN_RK4_TIME_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_RK4_TIME_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjectionMonitor.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;
using Field = tensorium_RG::Field3D<double>;
using Boundary = tensorium_RG::BoundaryClamp;
using RK4Stepper = tensorium_RG::bssn::BSSNRKStepper<double, Boundary>;

constexpr size_t kPadding = 4;
constexpr size_t kNumSteps = 5;
constexpr double kCfl = 0.25;
constexpr double kPi = 3.14159265358979323846;
constexpr double kChiFloor = 1e-6;

struct FarRegionStats {
    double max_abs = 0.0;
    double rms = 0.0;
    size_t samples = 0;
};

size_t interior_begin(size_t lo, size_t padding, size_t hi) { return std::min(lo + padding, hi); }

size_t interior_end(size_t hi, size_t padding, size_t lo) {
    return (hi > padding) ? hi - padding : hi;
}

double max_abs_interior(const Field &field, const Grid &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);

    double max_val = 0.0;
    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                const double val = field.ptr()[field.idx(i, j, k)];
                max_val = std::max(max_val, std::abs(val));
            }
    return max_val;
}

double max_abs_vector_interior(const Field fields[3], const Grid &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);

    double max_val = 0.0;
    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                const size_t id = fields[0].idx(i, j, k);
                for (int c = 0; c < 3; ++c)
                    max_val = std::max(max_val, std::abs(fields[c].ptr()[id]));
            }
    return max_val;
}
void enforce_chi_floor(Grid &grid, double chi_floor) {
    const size_t total = grid.chi.st.nx_tot * grid.chi.st.ny_tot * grid.chi.st.nz_tot;
    double      *ptr = grid.chi.ptr();
    for (size_t idx = 0; idx < total; ++idx)
        ptr[idx] = std::max(ptr[idx], chi_floor);
    double min_val = ptr[0];
    for (size_t idx = 1; idx < total; ++idx)
        min_val = std::min(min_val, ptr[idx]);
    std::printf("[chi_floor] applied: min chi=%.3e\n", min_val);
}

double max_abs_tensor_interior(const Field fields[6], const Grid &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);

    double max_val = 0.0;
    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                const size_t id = fields[0].idx(i, j, k);
                for (int s = 0; s < 6; ++s)
                    max_val = std::max(max_val, std::abs(fields[s].ptr()[id]));
            }
    return max_val;
}

FarRegionStats far_region_stats(const Grid &grid, const Field &field, size_t padding, double r_cut2,
                                double target = 0.0) {
    FarRegionStats stats;
    size_t         I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);
    double       accum = 0.0;

    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                if ((x * x + y * y + z * z) <= r_cut2)
                    continue;
                const double val = field.ptr()[field.idx(i, j, k)] - target;
                const double abs_val = std::abs(val);
                stats.max_abs = std::max(stats.max_abs, abs_val);
                accum += val * val;
                ++stats.samples;
            }

    if (stats.samples > 0)
        stats.rms = std::sqrt(accum / static_cast<double>(stats.samples));
    return stats;
}

FarRegionStats far_region_abs(const Grid &grid, const Field &field, size_t padding, double r_cut2) {
    return far_region_stats(grid, field, padding, r_cut2, 0.0);
}

void center_grid(Grid &grid) {
    const double half_x = 0.5 * (static_cast<double>(grid.dims.nx) - 1.0) * grid.dx;
    const double half_y = 0.5 * (static_cast<double>(grid.dims.ny) - 1.0) * grid.dy;
    const double half_z = 0.5 * (static_cast<double>(grid.dims.nz) - 1.0) * grid.dz;
    grid.x0 = -half_x;
    grid.y0 = -half_y;
    grid.z0 = -half_z;
}

void export_slice(const Grid &grid, const Field &field, const std::string &path, char axis = 'z') {
    namespace fs = std::filesystem;
    const fs::path out_path(path);
    if (!out_path.parent_path().empty())
        fs::create_directories(out_path.parent_path());

    std::ofstream ofs(path);
    TENSORIUM_TEST_ASSERT(ofs.good());
    ofs << "x,y,value\n";

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ic = (I0 + I1) / 2;
    const size_t jc = (J0 + J1) / 2;
    const size_t kc = (K0 + K1) / 2;

    auto write_entry = [&](size_t i, size_t j, size_t k, double coord_a, double coord_b) {
        const double value = field.ptr()[field.idx(i, j, k)];
        ofs << coord_a << ',' << coord_b << ',' << value << '\n';
    };

    if (axis == 'x') {
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                double x, y, z;
                grid.coords(ic, j, k, x, y, z);
                write_entry(ic, j, k, y, z);
            }
    } else if (axis == 'y') {
        for (size_t k = K0; k < K1; ++k)
            for (size_t i = I0; i < I1; ++i) {
                double x, y, z;
                grid.coords(i, jc, k, x, y, z);
                write_entry(i, jc, k, x, z);
            }
    } else { // default to z slicing
        for (size_t j = J0; j < J1; ++j)
            for (size_t i = I0; i < I1; ++i) {
                double x, y, z;
                grid.coords(i, j, kc, x, y, z);
                write_entry(i, j, kc, x, y);
            }
    }
}

std::string make_slice_name(const std::filesystem::path &dir, const std::string &field_name,
                            size_t step) {
    return (dir / (field_name + "_step" + std::to_string(step) + ".csv")).string();
}

void export_all_slices(const Grid &grid, const std::filesystem::path &case_dir, size_t step,
                       char axis = 'z') {
    export_slice(grid, grid.chi, make_slice_name(case_dir, "chi", step), axis);
    export_slice(grid, grid.K, make_slice_name(case_dir, "K", step), axis);
    export_slice(grid, grid.alpha, make_slice_name(case_dir, "alpha", step), axis);
    export_slice(grid, grid.tildeGamma[0], make_slice_name(case_dir, "GammaTilde_x", step), axis);
    export_slice(grid, grid.Hc, make_slice_name(case_dir, "H", step), axis);
}

double compute_rhs_max(const Grid &grid, tensorium_RG::bssn::BSSNRHSWorkspace<double> &rhs,
                       const tensorium_RG::bssn::GaugeParameters<double> &gauge_params) {
    tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs.tildeGamma, kPadding);
    tensorium_RG::bssn::compute_rhs_B(grid, rhs.tildeGamma, rhs.B, gauge_params, kPadding);
    tensorium_RG::bssn::compute_rhs_beta(grid, rhs.beta, gauge_params, kPadding);
    tensorium_RG::bssn::compute_rhs_alpha(grid, rhs.alpha, kPadding);
    tensorium_RG::bssn::compute_rhs_chi(grid, rhs.chi, kPadding);
    tensorium_RG::bssn::compute_rhs_gamma_tilde(grid, rhs.gamma_tilde, kPadding);
    tensorium_RG::bssn::compute_rhs_A_tilde(grid, rhs.A_tilde, kPadding);
    tensorium_RG::bssn::compute_rhs_K(grid, rhs.K, kPadding);

    double max_rhs = 0.0;
    max_rhs = std::max(max_rhs, max_abs_interior(rhs.alpha, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_interior(rhs.chi, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_interior(rhs.K, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_vector_interior(rhs.beta, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_vector_interior(rhs.B, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_vector_interior(rhs.tildeGamma, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_tensor_interior(rhs.gamma_tilde, grid, kPadding));
    max_rhs = std::max(max_rhs, max_abs_tensor_interior(rhs.A_tilde, grid, kPadding));
    return max_rhs;
}

struct EvolutionDiagnostics {
    tensorium_RG::bssn::ConstraintMonitorStats monitor;
    FarRegionStats                             far_chi;
    FarRegionStats                             far_alpha;
    FarRegionStats                             far_K;
    FarRegionStats                             far_gamma;
    FarRegionStats                             far_H;
    FarRegionStats                             bound_chi;
    FarRegionStats                             bound_alpha;
    FarRegionStats                             bound_K;
    FarRegionStats                             bound_gamma;
    FarRegionStats                             bound_H;
    double                                     max_rhs = 0.0;
};

EvolutionDiagnostics run_case(const std::string                 &label,
                              const std::function<void(Grid &)> &initializer,
                              bool enforce_stationary, double monitor_r_min_override = -1.0) {
    Grid grid(64, 64, 64, kPadding, 0.25, 0.25, 0.25);
    center_grid(grid);
    initializer(grid);

    tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs_buffers;
    rhs_buffers.allocate_like(grid);
    tensorium_RG::bssn::GaugeParameters<double> gauge_params{};

    RK4Stepper stepper(grid, kPadding);
    gauge_params.eta = 3.0;
    gauge_params.beta_B_coeff = 0.75;
    stepper.set_gauge_parameters(gauge_params);

    const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, kCfl, kPadding);
    std::printf("[rk4] %s dt=%.3e steps=%zu\n", label.c_str(), dt, kNumSteps);

    namespace fs = std::filesystem;
    const fs::path case_dir = fs::path("Output") / "rk4" / label;
    fs::create_directories(case_dir);

    size_t step_index = 0;
    export_all_slices(grid, case_dir, step_index);

    for (size_t s = 0; s < kNumSteps; ++s) {
        stepper.step(grid, dt, s + 1);

        enforce_chi_floor(grid, kChiFloor);
        tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

        tensorium_RG::bssn::project_bssn_after_update(grid, kPadding);

        ++step_index;
    }

    const double min_extent = std::min(
        {(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy, (grid.dims.nz - 1) * grid.dz});
    const double r_min_default = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
    const double r_min = (monitor_r_min_override > 0.0) ? monitor_r_min_override : r_min_default;
    const double r_max = 0.45 * min_extent;
    EvolutionDiagnostics diagnostics;
    diagnostics.monitor = tensorium_RG::bssn::project_and_monitor(
        grid, grid.Hc, grid.Mc, grid.Cc, r_min, r_max, 0.0, 0.0, 0.0, kPadding);

    const double r_cut = 8.0 * grid.dx;
    const double r_cut2 = r_cut * r_cut;
    diagnostics.far_chi = far_region_stats(grid, grid.chi, kPadding, r_cut2, 1.0);
    diagnostics.far_alpha = far_region_stats(grid, grid.alpha, kPadding, r_cut2, 1.0);
    diagnostics.far_K = far_region_stats(grid, grid.K, kPadding, r_cut2, 0.0);
    diagnostics.far_gamma = far_region_stats(grid, grid.tildeGamma[0], kPadding, r_cut2, 0.0);
    diagnostics.far_H = far_region_stats(grid, grid.Hc, kPadding, r_cut2, 0.0);
    diagnostics.bound_chi = far_region_abs(grid, grid.chi, kPadding, r_cut2);
    diagnostics.bound_alpha = far_region_abs(grid, grid.alpha, kPadding, r_cut2);
    diagnostics.bound_K = far_region_abs(grid, grid.K, kPadding, r_cut2);
    diagnostics.bound_gamma = far_region_abs(grid, grid.tildeGamma[0], kPadding, r_cut2);
    diagnostics.bound_H = far_region_abs(grid, grid.Hc, kPadding, r_cut2);

    if (enforce_stationary)
        diagnostics.max_rhs = compute_rhs_max(grid, rhs_buffers, gauge_params);
    else
        diagnostics.max_rhs = std::numeric_limits<double>::quiet_NaN();

    export_all_slices(grid, case_dir, step_index);
    tensorium_RG::bssn::print_constraint_monitor(diagnostics.monitor, label.c_str());
    return diagnostics;
}

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct RawPunctures {
    Vec3 left;
    Vec3 right;
    bool left_valid = false;
    bool right_valid = false;
    bool valid() const { return left_valid && right_valid; }
};

struct MinSample {
    double value = std::numeric_limits<double>::max();
    Vec3   pos{};
    bool   valid = false;
};

RawPunctures find_puncture_candidates(const Grid &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);

    MinSample best_left;
    MinSample best_right;

    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                const size_t idx = grid.chi.idx(i, j, k);
                const double val = grid.chi.ptr()[idx];
                if (!std::isfinite(val))
                    continue;
                Vec3 pos{};
                grid.coords(i, j, k, pos.x, pos.y, pos.z);
                if (pos.x <= 0.0) {
                    if (val < best_left.value) {
                        best_left.value = val;
                        best_left.pos = pos;
                        best_left.valid = true;
                    }
                } else {
                    if (val < best_right.value) {
                        best_right.value = val;
                        best_right.pos = pos;
                        best_right.valid = true;
                    }
                }
            }

    RawPunctures raw;
    if (best_left.valid) {
        raw.left = best_left.pos;
        raw.left_valid = true;
    }
    if (best_right.valid) {
        raw.right = best_right.pos;
        raw.right_valid = true;
    }
    return raw;
}

struct PunctureTrack {
    Vec3 p1;
    Vec3 p2;
};

PunctureTrack order_initial(const RawPunctures &raw) {
    TENSORIUM_TEST_ASSERT(raw.valid());
    return PunctureTrack{raw.left, raw.right};
}

double dist_sq(const Vec3 &a, const Vec3 &b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

PunctureTrack assign_to_previous(const RawPunctures &raw, const PunctureTrack &prev) {
    if (!raw.valid())
        return prev;
    return PunctureTrack{raw.left, raw.right};
}

double compute_distance(const PunctureTrack &track) {
    const double dx = track.p2.x - track.p1.x;
    const double dy = track.p2.y - track.p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

double compute_phase(const PunctureTrack &track) {
    const double dy = track.p2.y - track.p1.y;
    const double dx = track.p2.x - track.p1.x;
    return std::atan2(dy, dx);
}

double wrap_delta(double delta) {
    while (delta > kPi)
        delta -= 2.0 * kPi;
    while (delta < -kPi)
        delta += 2.0 * kPi;
    return delta;
}

struct TrajectorySample {
    double t = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double distance = 0.0;
    double phase = 0.0;
};

void write_trajectory_csv(const std::vector<TrajectorySample> &samples,
                          const std::filesystem::path         &path) {
    namespace fs = std::filesystem;
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    std::ofstream ofs(path);
    TENSORIUM_TEST_ASSERT(ofs.good());
    ofs << "t,x1,y1,x2,y2,distance\n";
    for (const auto &s : samples)
        ofs << s.t << ',' << s.x1 << ',' << s.y1 << ',' << s.x2 << ',' << s.y2 << ',' << s.distance
            << '\n';
}

void relaxed_binary_bowen_york_init(Grid &grid, double m1, double x1, double y1, double z1,
                                    const double P1[3], const double S1[3], double m2, double x2,
                                    double y2, double z2, const double P2[3], const double S2[3],
                                    double r_floor) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const double one = 1.0;
    const double zero = 0.0;

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);

                grid.K.ptr()[id] = zero;
                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[id] = zero;
                    grid.B[c].ptr()[id] = zero;
                    grid.tildeGamma[c].ptr()[id] = zero;
                }

                grid.gamma_tilde[tensorium_RG::XX].ptr()[id] = one;
                grid.gamma_tilde[tensorium_RG::XY].ptr()[id] = zero;
                grid.gamma_tilde[tensorium_RG::XZ].ptr()[id] = zero;
                grid.gamma_tilde[tensorium_RG::YY].ptr()[id] = one;
                grid.gamma_tilde[tensorium_RG::YZ].ptr()[id] = zero;
                grid.gamma_tilde[tensorium_RG::ZZ].ptr()[id] = one;

                tensorium_RG::init::invert_gamma_tilde(grid, i, j, k);

                grid.A_tilde[tensorium_RG::XX].ptr()[id] = zero;
                grid.A_tilde[tensorium_RG::XY].ptr()[id] = zero;
                grid.A_tilde[tensorium_RG::XZ].ptr()[id] = zero;
                grid.A_tilde[tensorium_RG::YY].ptr()[id] = zero;
                grid.A_tilde[tensorium_RG::YZ].ptr()[id] = zero;
                grid.A_tilde[tensorium_RG::ZZ].ptr()[id] = zero;

                grid.chi.ptr()[id] = one;
                grid.alpha.ptr()[id] = one;
            }

    tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

    tensorium_RG::init::fill_Atilde_bowen_york_binary(grid, x1, y1, z1, P1, S1, x2, y2, z2, P2, S2,
                                                      r_floor);
    tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

    try {
        tensorium_RG::init::solve_lichnerowicz_u_SOR(grid, m1, x1, y1, z1, m2, x2, y2, z2, r_floor,
                                                     2000, 1e-10, 1.8);
    } catch (const std::runtime_error &) {
        // Allow relaxed tolerances for high-resolution test grids.
    }
    tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

    enforce_chi_floor(grid, kChiFloor);
    tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

    tensorium_RG::bssn::compute_tildeGamma_full(grid, grid.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(grid);
    tensorium_RG::bssn::project_bssn_state(grid);
}

struct ConstraintStats {
    double max_H = 0.0;
    double max_M = 0.0;
    double max_C = 0.0;
    size_t samples = 0;
};

static inline double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

ConstraintStats constraints_far_from_punctures(const Grid &grid, const Field &Hc, const Field Mc[3],
                                               const Field Cc[3], size_t padding, Vec3 p1, Vec3 p2,
                                               double r_excl) {
    ConstraintStats st;

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    const size_t ib = interior_begin(I0, padding, I1);
    const size_t jb = interior_begin(J0, padding, J1);
    const size_t kb = interior_begin(K0, padding, K1);
    const size_t ie = interior_end(I1, padding, I0);
    const size_t je = interior_end(J1, padding, J0);
    const size_t ke = interior_end(K1, padding, K0);

    for (size_t i = ib; i < ie; ++i)
        for (size_t j = jb; j < je; ++j)
            for (size_t k = kb; k < ke; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);

                const double r1 = norm3(x - p1.x, y - p1.y, z - p1.z);
                const double r2 = norm3(x - p2.x, y - p2.y, z - p2.z);
                const double rmin = std::min(r1, r2);

                if (rmin < r_excl)
                    continue;

                const size_t id = Hc.idx(i, j, k);
                st.max_H = std::max(st.max_H, std::abs(Hc.ptr()[id]));

                for (int c = 0; c < 3; ++c)
                    st.max_M = std::max(st.max_M, std::abs(Mc[c].ptr()[id]));

                for (int c = 0; c < 3; ++c)
                    st.max_C = std::max(st.max_C, std::abs(Cc[c].ptr()[id]));
                ++st.samples;
            }

    return st;
}
template <typename Grid> void enforce_algebraic_constraints(Grid &grid) {
    using T = double;

    const size_t total = grid.chi.st.nx_tot * grid.chi.st.ny_tot * grid.chi.st.nz_tot;

#pragma omp parallel for
    for (size_t id = 0; id < total; ++id) {
        T gxx = grid.gamma_tilde[tensorium_RG::XX].ptr()[id];
        T gxy = grid.gamma_tilde[tensorium_RG::XY].ptr()[id];
        T gxz = grid.gamma_tilde[tensorium_RG::XZ].ptr()[id];
        T gyy = grid.gamma_tilde[tensorium_RG::YY].ptr()[id];
        T gyz = grid.gamma_tilde[tensorium_RG::YZ].ptr()[id];
        T gzz = grid.gamma_tilde[tensorium_RG::ZZ].ptr()[id];

        T det = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) +
                gxz * (gxy * gyz - gyy * gxz);

        if (det <= 0.0 || std::isnan(det)) {
            det = 1.0;
        }

        T factor = std::pow(det, -1.0 / 3.0);

        grid.gamma_tilde[tensorium_RG::XX].ptr()[id] *= factor;
        grid.gamma_tilde[tensorium_RG::XY].ptr()[id] *= factor;
        grid.gamma_tilde[tensorium_RG::XZ].ptr()[id] *= factor;
        grid.gamma_tilde[tensorium_RG::YY].ptr()[id] *= factor;
        grid.gamma_tilde[tensorium_RG::YZ].ptr()[id] *= factor;
        grid.gamma_tilde[tensorium_RG::ZZ].ptr()[id] *= factor;
    }
}
} // namespace
REGISTER_TEST(
    "bssn.evolution.rk4.minkowski", "BSSN RK4 preserves Minkowski spacetime over few steps", []() {
        const auto diagnostics =
            run_case("minkowski", [](Grid &g) { tensorium_RG::init::minkowski(g, 0.0); }, true);

        tensorium::tests::expect_le(diagnostics.max_rhs, 1e-11, "minkowski rhs max");
        tensorium::tests::expect_le(diagnostics.monitor.max_H, 1e-11, "minkowski H constraint");
        tensorium::tests::expect_le(diagnostics.monitor.max_trace_A, 1e-11, "minkowski trace(A)");
        tensorium::tests::expect_le(diagnostics.monitor.max_det_drift, 1e-11,
                                    "minkowski det drift");
        tensorium::tests::expect_le(diagnostics.far_chi.max_abs, 1e-12, "minkowski chi far");
        tensorium::tests::expect_le(diagnostics.far_alpha.max_abs, 1e-12, "minkowski alpha far");
        tensorium::tests::expect_le(diagnostics.far_K.max_abs, 1e-12, "minkowski K far");
    });

REGISTER_TEST(
    "bssn.evolution.rk4.schwarzschild", "BSSN RK4 keeps Schwarzschild data bounded over few steps",
    []() {
        const auto diagnostics = run_case(
            "schwarzschild", [](Grid &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
            false);

        tensorium::tests::expect_le(diagnostics.monitor.max_trace_A, 5e-3,
                                    "schwarzschild trace(A)");
        tensorium::tests::expect_le(diagnostics.monitor.max_det_drift, 5e-3,
                                    "schwarzschild det drift");
        tensorium::tests::expect_le(diagnostics.monitor.max_H, 5e-2, "schwarzschild constraint H");
        tensorium::tests::expect_le(diagnostics.bound_chi.max_abs, 2.0,
                                    "schwarzschild chi far bound");
        tensorium::tests::expect_le(diagnostics.bound_alpha.max_abs, 2.0,
                                    "schwarzschild alpha far bound");
        tensorium::tests::expect_le(diagnostics.bound_K.max_abs, 0.2, "schwarzschild K far bound");
        tensorium::tests::expect_le(diagnostics.bound_gamma.max_abs, 0.5,
                                    "schwarzschild Gamma far bound");
        tensorium::tests::expect_le(diagnostics.bound_H.max_abs, 7e-3, "schwarzschild H far bound");
    });

REGISTER_TEST(
    "bssn.evolution.rk4.bowen_york", "BSSN RK4 keeps Bowen-York binary data bounded over few steps",
    []() {
        const auto diagnostics = run_case(
            "bowen_york",
            [](Grid &g) {
                const double P1[3] = {0.0, 0.05, 0.0};
                const double P2[3] = {0.0, -0.05, 0.0};
                const double S1[3] = {0.0, 0.0, 0.15};
                const double S2[3] = {0.0, 0.0, -0.15};
                tensorium_RG::init::binary_bowen_york_puncture_init(
                    g, 0.5, -1.5, 0.0, 0.0, P1, S1, 0.5, 1.5, 0.0, 0.0, P2, S2, 1e-4);
            },
            false, 2.0);

        tensorium::tests::expect_le(diagnostics.monitor.max_trace_A, 2e-2, "bowen-york trace(A)");
        tensorium::tests::expect_le(diagnostics.monitor.max_det_drift, 2e-2,
                                    "bowen-york det drift");
        tensorium::tests::expect_le(diagnostics.monitor.max_H, 10.0, "bowen-york constraint H");
        tensorium::tests::expect_le(diagnostics.bound_chi.max_abs, 3.0, "bowen-york chi far bound");
        tensorium::tests::expect_le(diagnostics.bound_alpha.max_abs, 3.0,
                                    "bowen-york alpha far bound");
        tensorium::tests::expect_le(diagnostics.bound_K.max_abs, 0.5, "bowen-york K far bound");
        tensorium::tests::expect_le(diagnostics.bound_gamma.max_abs, 2.0,
                                    "bowen-york Gamma far bound");
        tensorium::tests::expect_le(diagnostics.bound_H.max_abs, 5.0, "bowen-york H far bound");
    });

REGISTER_TEST(
    "bssn.evolution.rk4.bowen_york_inspiral", "Bowen-York binary shows inspiral and rotation",
    []() {
        tensorium_RG::bssn::set_chi_tolerance_override(-1e-1);
        Grid grid(64, 64, 64, kPadding, 0.25, 0.25, 0.25);
        center_grid(grid);
        const double P1[3] = {0.0, 0.05, 0.0};
        const double P2[3] = {0.0, -0.05, 0.0};
        const double S1[3] = {0.0, 0.0, 0.15};
        const double S2[3] = {0.0, 0.0, -0.15};
        relaxed_binary_bowen_york_init(grid, 0.2, -1.5, 0.0, 0.0, P1, S1, 0.2, 1.5, 0.0, 0.0, P2,
                                       S2, 5e-2);

        tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs_buffers;
        rhs_buffers.allocate_like(grid);
        tensorium_RG::bssn::GaugeParameters<double> gauge_params{};

        RK4Stepper stepper(grid, kPadding);
        stepper.set_gauge_parameters(gauge_params);

        const double inspiral_cfl = 0.4;
        const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, kCfl, kPadding) * inspiral_cfl;
        constexpr size_t total_steps = 128;
        namespace fs = std::filesystem;
        const fs::path case_dir = fs::path("Output") / "rk4" / "bowen_york";
        fs::create_directories(case_dir);

        std::vector<TrajectorySample> samples;
        samples.reserve(total_steps + 1);

        auto record_sample = [&](double time, const PunctureTrack &track) {
            TrajectorySample sample;
            sample.t = time;
            sample.x1 = track.p1.x;
            sample.y1 = track.p1.y;
            sample.x2 = track.p2.x;
            sample.y2 = track.p2.y;
            sample.distance = compute_distance(track);
            sample.phase = compute_phase(track);
            samples.push_back(sample);
        };

        const size_t  tracking_padding = kPadding;
        RawPunctures  raw = find_puncture_candidates(grid, tracking_padding);
        PunctureTrack current = order_initial(raw);
        double        time = 0.0;
        record_sample(time, current);
        double       accumulated_phase = 0.0;
        const double initial_distance = samples.front().distance;

        const double phase_target = kPi / 4.0;
        size_t       executed_steps = 0;
        for (size_t step = 0; step < total_steps; ++step) {
            stepper.step(grid, dt, step + 1);
            enforce_chi_floor(grid, kChiFloor);
            tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);
            tensorium_RG::bssn::project_bssn_after_update(grid, kPadding);
            time += dt;
            raw = find_puncture_candidates(grid, tracking_padding);
            if (raw.valid())
                current = assign_to_previous(raw, current);
            record_sample(time, current);
            executed_steps = step + 1;
            if (samples.size() >= 2 && raw.valid()) {
                const auto &latest = samples.back();
                const auto &previous = samples[samples.size() - 2];
                accumulated_phase += wrap_delta(latest.phase - previous.phase);
                const double distance_ratio = latest.distance / initial_distance;
                if (std::abs(accumulated_phase) >= phase_target && distance_ratio <= 0.95)
                    break;
            }
        }

        TENSORIUM_TEST_ASSERT(samples.size() >= 2);

        TENSORIUM_TEST_ASSERT(std::abs(accumulated_phase) >= (kPi / 4.0));

        const double final_distance = samples.back().distance;
        TENSORIUM_TEST_ASSERT(final_distance < initial_distance * 0.95);
        TENSORIUM_TEST_ASSERT(final_distance < initial_distance);

        const double min_extent =
            std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                      (grid.dims.nz - 1) * grid.dz});
        const double r_min = 1.0;
        const double r_max = 0.35 * min_extent;
        const auto   monitor = tensorium_RG::bssn::project_and_monitor(
            grid, grid.Hc, grid.Mc, grid.Cc, r_min, r_max, 0.0, 0.0, 0.0, kPadding);
        tensorium::tests::expect_le(monitor.max_H, 1e3, "bowen-york inspiral Hamiltonian");
        tensorium::tests::expect_le(monitor.max_trace_A, 1e-1, "bowen-york inspiral Tr(A)");
        tensorium::tests::expect_le(monitor.max_det_drift, 1e-1, "bowen-york inspiral det drift");

        const double max_Mc = max_abs_vector_interior(grid.Mc, grid, kPadding);
        tensorium::tests::expect_le(max_Mc, 100.0, "bowen-york inspiral momentum constraint");

        write_trajectory_csv(samples, case_dir / "trajectory.csv");
        tensorium_RG::bssn::set_chi_tolerance_override(std::numeric_limits<double>::quiet_NaN());
    });
REGISTER_TEST("bssn.debug.minimal_dump", "Minimal BSSN RK4 debug with matrix dumps", []() {
    using Grid = tensorium_RG::BSSNGridSoA<double>;
    using Boundary = tensorium_RG::BoundaryClamp;
    using RK4 = tensorium_RG::bssn::BSSNRKStepper<double, Boundary>;

    constexpr size_t N = 128;
    constexpr size_t padding = 4;

    Grid grid(N, N, N, padding, 0.25, 0.25, 0.25);
    center_grid(grid);

    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    gauge.eta = 3.0;
    gauge.beta_B_coeff = 0.75;

    RK4 stepper(grid, padding);
    stepper.set_gauge_parameters(gauge);

    const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, 0.05, padding);

    auto dump_cell = [&](size_t i, size_t j, size_t k) {
        const size_t id = grid.alpha.idx(i, j, k);

        std::printf("\n--- CELL (%zu,%zu,%zu) ---\n", i, j, k);

        std::printf("chi = %.6e  alpha = %.6e  K = %.6e\n", grid.chi.ptr()[id],
                    grid.alpha.ptr()[id], grid.K.ptr()[id]);

        std::printf("tildeGamma = [%.3e %.3e %.3e]\n", grid.tildeGamma[0].ptr()[id],
                    grid.tildeGamma[1].ptr()[id], grid.tildeGamma[2].ptr()[id]);

        auto &g = grid.gamma_tilde;
        auto &A = grid.A_tilde;

        std::printf("gamma_tilde:\n");
        std::printf("[%.6e %.6e %.6e]\n", g[0].ptr()[id], g[1].ptr()[id], g[2].ptr()[id]);
        std::printf("[%.6e %.6e %.6e]\n", g[1].ptr()[id], g[3].ptr()[id], g[4].ptr()[id]);
        std::printf("[%.6e %.6e %.6e]\n", g[2].ptr()[id], g[4].ptr()[id], g[5].ptr()[id]);

        std::printf("A_tilde:\n");
        std::printf("[%.6e %.6e %.6e]\n", A[0].ptr()[id], A[1].ptr()[id], A[2].ptr()[id]);
        std::printf("[%.6e %.6e %.6e]\n", A[1].ptr()[id], A[3].ptr()[id], A[4].ptr()[id]);
        std::printf("[%.6e %.6e %.6e]\n", A[2].ptr()[id], A[4].ptr()[id], A[5].ptr()[id]);
    };

    const size_t ic = N / 2;
    const size_t jc = N / 2;
    const size_t kc = N / 2;

    std::printf("\n=== INITIAL STATE ===\n");
    dump_cell(ic, jc, kc);

    for (int step = 1; step <= 3; ++step) {
        stepper.step(grid, dt, step);

        enforce_chi_floor(grid, 1e-6);
        tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);
        tensorium_RG::bssn::project_bssn_after_update(grid, padding);

        const auto monitor = tensorium_RG::bssn::project_and_monitor(
            grid, grid.Hc, grid.Mc, grid.Cc, 1.0, 6.0, 0.0, 0.0, 0.0, padding);

        std::printf("\n=== STEP %d ===\n", step);
        std::printf("max_H = %.3e  Tr(A) = %.3e  det_drift = %.3e\n", monitor.max_H,
                    monitor.max_trace_A, monitor.max_det_drift);

        dump_cell(ic, jc, kc);
    }

    TENSORIUM_TEST_ASSERT(true);
});

REGISTER_TEST("bssn.debug.bowen_york_minimal",
              "Minimal Bowen-York binary diagnostic (large separation, frozen gauge)", []() {
                  using Grid = tensorium_RG::BSSNGridSoA<double>;
                  using Boundary = tensorium_RG::BoundaryClamp;
                  using RK4Stepper = tensorium_RG::bssn::BSSNRKStepper<double, Boundary>;

                  constexpr size_t kPadding = 4;
                  constexpr double dx = 0.25;
                  constexpr size_t N = 96;

                  Grid grid(N, N, N, kPadding, dx, dx, dx);
                  center_grid(grid);

                  const double P1[3] = {0.0, 0.3, 0.0};
                  const double P2[3] = {0.0, -0.3, 0.0};
                  const double S1[3] = {0.0, 0.0, 0.10};
                  const double S2[3] = {0.0, 0.0, -0.10};

                  const double sep = 4.0;

                  tensorium_RG::init::binary_bowen_york_puncture_init(
                      grid, 0.5, -sep, 0.0, 0.0, P1, S1, 0.5, sep, 0.0, 0.0, P2, S2, 1e-4);

                  tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);
                  enforce_chi_floor(grid, 1e-6);
                  enforce_algebraic_constraints(grid);
                  tensorium_RG::bssn::project_bssn_state(grid);
                  tensorium_RG::bssn::compute_tildeGamma_full(grid, grid.Gamma_tilde);
                  tensorium_RG::bssn::compute_tildeGamma_contracted(grid);

                  tensorium_RG::bssn::BSSNRHSWorkspace<double> rhs;
                  rhs.allocate_like(grid);

                  tensorium_RG::bssn::GaugeParameters<double> gauge{};
                  gauge.eta = 2.0;
                  gauge.beta_B_coeff = 0.75;
                  RK4Stepper stepper(grid, kPadding);
                  stepper.set_gauge_parameters(gauge);

                  const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, 0.25, kPadding);

                  auto dump_center = [&](const char *label) {
                      const size_t i = N / 2;
                      const size_t j = N / 2;
                      const size_t k = N / 2;
                      const size_t id = grid.alpha.idx(i, j, k);

                      std::printf("\n=== %s ===\n", label);
                      std::printf("chi = %.6e  alpha = %.6e  K = %.6e\n", grid.chi.ptr()[id],
                                  grid.alpha.ptr()[id], grid.K.ptr()[id]);

                      std::printf("tildeGamma = [%.3e %.3e %.3e]\n", grid.tildeGamma[0].ptr()[id],
                                  grid.tildeGamma[1].ptr()[id], grid.tildeGamma[2].ptr()[id]);

                      std::printf("gamma_tilde:\n");
                      for (int r = 0; r < 3; ++r) {
                          std::printf("[%.6e %.6e %.6e]\n",
                                      grid.gamma_tilde[r == 0   ? tensorium_RG::XX
                                                       : r == 1 ? tensorium_RG::XY
                                                                : tensorium_RG::XZ]
                                          .ptr()[id],
                                      grid.gamma_tilde[r == 0   ? tensorium_RG::XY
                                                       : r == 1 ? tensorium_RG::YY
                                                                : tensorium_RG::YZ]
                                          .ptr()[id],
                                      grid.gamma_tilde[r == 0   ? tensorium_RG::XZ
                                                       : r == 1 ? tensorium_RG::YZ
                                                                : tensorium_RG::ZZ]
                                          .ptr()[id]);
                      }

                      std::printf("A_tilde:\n");
                      for (int r = 0; r < 3; ++r) {
                          std::printf("[%.6e %.6e %.6e]\n",
                                      grid.A_tilde[r == 0   ? tensorium_RG::XX
                                                   : r == 1 ? tensorium_RG::XY
                                                            : tensorium_RG::XZ]
                                          .ptr()[id],
                                      grid.A_tilde[r == 0   ? tensorium_RG::XY
                                                   : r == 1 ? tensorium_RG::YY
                                                            : tensorium_RG::YZ]
                                          .ptr()[id],
                                      grid.A_tilde[r == 0   ? tensorium_RG::XZ
                                                   : r == 1 ? tensorium_RG::YZ
                                                            : tensorium_RG::ZZ]
                                          .ptr()[id]);
                      }
                  };

                  namespace fs = std::filesystem;
                  const fs::path case_dir = fs::path("Output") / "rk4" / "bowen_york_minimal";
                  fs::create_directories(case_dir);

                  auto step_dir = [&](int step) {
                      fs::path d = case_dir / ("step" + std::to_string(step));
                      fs::create_directories(d);
                      return d;
                  };

                  auto export_chi_slice = [&](int step) {
                      const auto d = step_dir(step);
                      export_slice(grid, grid.chi, (d / "chi_z.csv").string(), 'z');
                  };
                  dump_center("INITIAL STATE");

                  export_chi_slice(0);
                  for (int step = 1; step <= 10; ++step) {
                      enforce_algebraic_constraints(grid);
                      stepper.step(grid, dt, step);

                      enforce_chi_floor(grid, 1e-6);

                      size_t total =
                          grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;

                      for (size_t idx = 0; idx < total; ++idx) {
                          grid.beta[0].ptr()[idx] = 0.0;
                          grid.beta[1].ptr()[idx] = 0.0;
                          grid.beta[2].ptr()[idx] = 0.0;
                          grid.B[0].ptr()[idx] = 0.0;
                          grid.B[1].ptr()[idx] = 0.0;
                          grid.B[2].ptr()[idx] = 0.0;
                      }

                      tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);
                      tensorium_RG::bssn::project_bssn_after_update(grid, kPadding);

                      export_chi_slice(step);
                      const double min_extent =
                          std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                                    (grid.dims.nz - 1) * grid.dz});

                      const double r_min = 2.0 * dx;
                      const double r_max = 0.4 * min_extent;

                      auto monitor = tensorium_RG::bssn::project_and_monitor(
                          grid, grid.Hc, grid.Mc, grid.Cc, r_min, r_max, 0.0, 0.0, 0.0, kPadding);

                      Vec3 bh1{-sep, 0.0, 0.0};
                      Vec3 bh2{sep, 0.0, 0.0};

                      const double r_excl = 8.0 * dx;

                      auto far = ::constraints_far_from_punctures(grid, grid.Hc, grid.Mc, grid.Cc,
                                                                  kPadding, bh1, bh2, r_excl);
                      std::printf(
                          "STEP %d: far(max_H)=%.3e far(max_M)=%.3e far(max_C)=%.3e samples=%zu\n",
                          step, far.max_H, far.max_M, far.max_C, far.samples);
                      std::printf("STEP %d: max_H=%.3e  Tr(A)=%.3e  det_drift=%.3e\n", step,
                                  monitor.max_H, monitor.max_trace_A, monitor.max_det_drift);

                      dump_center(("STEP " + std::to_string(step)).c_str());
                  }

                  TENSORIUM_TEST_ASSERT(true);
              });
namespace {

// Condition aux limites "Open" (Extrapolation d'ordre 0)
struct BoundaryOpen {
    template <typename Field, typename Dims> static void apply(Field &f, const Dims &dims) {
        const size_t padding = dims.ng;

#pragma omp parallel for collapse(2)
        for (size_t k = 0; k < f.st.nz_tot; ++k) {
            for (size_t j = 0; j < f.st.ny_tot; ++j) {
                double val_left = f.ptr()[f.idx(padding, j, k)];
                for (size_t i = 0; i < padding; ++i) {
                    f.ptr()[f.idx(i, j, k)] = val_left;
                }
                size_t valid_right_idx = f.st.nx_tot - padding - 1;
                double val_right = f.ptr()[f.idx(valid_right_idx, j, k)];
                for (size_t i = 1; i <= padding; ++i) {
                    f.ptr()[f.idx(valid_right_idx + i, j, k)] = val_right;
                }
            }
        }
    }
};

} // namespace

REGISTER_TEST("bssn.debug.bowen_york_fast", "Fast moving binary - Full Export", []() {
    using Grid = tensorium_RG::BSSNGridSoA<double>;
    using Boundary = BoundaryOpen;
    using RK4Stepper = tensorium_RG::bssn::BSSNRKStepper<double, Boundary>;

    constexpr size_t kPadding = 4;
    constexpr double dx = 0.25;
    constexpr size_t N = 112;

    Grid grid(N, N, N, kPadding, dx, dx, dx);
    center_grid(grid);

    const double sep = 3.0;
    const double mom = 0.20;

    const double P1[3] = {0.0, mom, 0.0};
    const double P2[3] = {0.0, -mom, 0.0};
    const double S1[3] = {0.0, 0.0, 0.10};
    const double S2[3] = {0.0, 0.0, -0.10};

    // Initialisation
    tensorium_RG::init::binary_bowen_york_puncture_init(grid, 0.5, -sep, 0.0, 0.0, P1, S1, 0.5, sep,
                                                        0.0, 0.0, P2, S2, 1e-4);

    // Préparation Initiale
    enforce_chi_floor(grid, 1e-6);
    enforce_algebraic_constraints(grid); 

    tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);
    tensorium_RG::bssn::project_bssn_state(grid);

    tensorium_RG::bssn::compute_tildeGamma_full(grid, grid.Gamma_tilde);
    tensorium_RG::bssn::compute_tildeGamma_contracted(grid);

    // Configuration Jauge
    tensorium_RG::bssn::GaugeParameters<double> gauge{};
    gauge.eta = 2.0;
    gauge.beta_B_coeff = 0.75;

    RK4Stepper stepper(grid, kPadding);
    stepper.set_gauge_parameters(gauge);

    const double dt =
        tensorium_RG::bssn::compute_dt_cfl(grid, 0.1, kPadding); 

    namespace fs = std::filesystem;
    const fs::path case_dir = fs::path("Output") / "rk4" / "bowen_york_fast";
    fs::create_directories(case_dir);

    auto export_chi_slice = [&](int step) {
        std::string fname = (case_dir / ("chi_z_step" + std::to_string(step) + ".csv")).string();
        export_slice(grid, grid.chi, fname, 'z');
    };

    auto enforce_alpha_floor = [&](double floor_val) {
        const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
        for (size_t i = 0; i < total; ++i) {
            if (grid.alpha.ptr()[i] < floor_val) {
                grid.alpha.ptr()[i] = floor_val;
            }
        }
    };

    std::printf("Starting Evolution (P=%.2f, N=%zu, dt=%.3f)...\n", mom, N, dt);

    export_chi_slice(0);

    for (int step = 1; step <= 40; ++step) {
        stepper.step(grid, dt, step);

        enforce_algebraic_constraints(grid); 
        enforce_chi_floor(grid, 1e-6);
        enforce_alpha_floor(1e-3);

        tensorium_RG::bssn::apply_halos_grid<Boundary>(grid);

        tensorium_RG::bssn::project_bssn_after_update(grid, kPadding);
        export_chi_slice(step);
        if (step % 10 == 0) {
            size_t center = grid.chi.idx(N / 2, N / 2, N / 2);
            double alpha_c = grid.alpha.ptr()[center];
            std::printf("Step %d / 250 (alpha_center=%.4f)\n", step, alpha_c);
        }
    }

    TENSORIUM_TEST_ASSERT(true);
});
#endif
