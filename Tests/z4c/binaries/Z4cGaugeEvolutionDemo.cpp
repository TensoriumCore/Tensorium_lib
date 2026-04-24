#include "../../framework/Assertions.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGamma.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGauge.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjectionMonitor.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

enum class SlicePlane { XY, XZ, YZ };

struct SliceRequest {
    SlicePlane plane;
    std::string tag;
    double coord = std::numeric_limits<double>::quiet_NaN(); // physical coordinate when needed
};

struct SlicePlaneSpec {
    SlicePlane plane;
    size_t     fixed_index;
    std::string tag;
};

struct DemoConfig {
    size_t nx = 128;
    size_t ng = 4;
    double dx = 0.25;
    size_t padding = 4;
    double eta = 1.0;
    bool   apply_update = false;
    double update_dt = 1e-3;
    bool   debug_perturb = true;
    double perturb_eps = 1e-6;
};

struct CaseOptions {
    std::vector<SliceRequest> slice_requests;
    bool add_perturbation = false;
    bool apply_update = false;
};

size_t clamp_index(size_t idx, size_t lo, size_t hi_exclusive) {
    if (idx < lo)
        return lo;
    if (idx >= hi_exclusive)
        return hi_exclusive - 1;
    return idx;
}

size_t find_index_near_x(const tensorium_RG::Z4cGridSoA<double> &grid, double x_phys) {
    const double rel = (x_phys - grid.x0) / grid.dx + grid.dims.ng;
    const long   idx = std::lround(rel);
    const size_t min_i = grid.dims.ng;
    const size_t max_i = grid.dims.ng + grid.dims.nx;
    const size_t clamped = clamp_index(static_cast<size_t>(std::max<long>(idx, 0)), min_i, max_i);
    return clamped;
}

SlicePlaneSpec realize_request(const tensorium_RG::Z4cGridSoA<double> &grid,
                               const SliceRequest &request) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    switch (request.plane) {
    case SlicePlane::XY: {
        const size_t k = K0 + grid.dims.nz / 2;
        return {SlicePlane::XY, k, request.tag};
    }
    case SlicePlane::XZ: {
        const size_t j = J0 + grid.dims.ny / 2;
        return {SlicePlane::XZ, j, request.tag};
    }
    case SlicePlane::YZ: {
        size_t i = I0 + grid.dims.nx / 2;
        if (!std::isnan(request.coord))
            i = find_index_near_x(grid, request.coord);
        return {SlicePlane::YZ, i, request.tag};
    }
    }
    return {SlicePlane::XY, K0 + grid.dims.nz / 2, request.tag};
}

std::string slugify_label(const char *label) {
    std::string slug(label ? label : "case");
    for (char &c : slug) {
        if (c == ' ')
            c = '_';
        else
            c = static_cast<char>(std::tolower(c));
    }
    return slug;
}

void export_scalar_slice(const tensorium_RG::Z4cGridSoA<double> &grid,
                         const tensorium_RG::Field3D<double> &field, const SlicePlaneSpec &spec,
                         const char *value_name, const std::string &path) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    std::ofstream out(path);
    switch (spec.plane) {
    case SlicePlane::XY: {
        const size_t k = clamp_index(spec.fixed_index, K0, K1);
        out << "x,y," << value_name << "\n";
        for (size_t i = I0; i < I1; ++i)
            for (size_t j = J0; j < J1; ++j) {
                const size_t id = field.idx(i, j, k);
                double       x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)z;
                out << x << ',' << y << ',' << field.ptr()[id] << '\n';
            }
        break;
    }
    case SlicePlane::XZ: {
        const size_t j = clamp_index(spec.fixed_index, J0, J1);
        out << "x,z," << value_name << "\n";
        for (size_t i = I0; i < I1; ++i)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = field.idx(i, j, k);
                double       x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)y;
                out << x << ',' << z << ',' << field.ptr()[id] << '\n';
            }
        break;
    }
    case SlicePlane::YZ: {
        const size_t i = clamp_index(spec.fixed_index, I0, I1);
        out << "y,z," << value_name << "\n";
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = field.idx(i, j, k);
                double       x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)x;
                out << y << ',' << z << ',' << field.ptr()[id] << '\n';
            }
        break;
    }
    }
}

void export_vector_slices(const tensorium_RG::Z4cGridSoA<double> &grid,
                          const tensorium_RG::Field3D<double> vec[3], const SlicePlaneSpec &spec,
                          const std::string &base_label, const std::string &quantity) {
    static const char *comp_name[3] = {"x", "y", "z"};
    for (int c = 0; c < 3; ++c) {
        const std::string path = quantity + "_" + comp_name[c] + "_slice_" + base_label + ".csv";
        const std::string value_name = quantity + "_" + comp_name[c];
        export_scalar_slice(grid, vec[c], spec, value_name.c_str(), path);
    }
}

void alloc_scalar_rhs(const tensorium_RG::Field3D<double> &src, tensorium_RG::Field3D<double> &dst) {
    dst.st = src.st;
    const size_t total = src.st.nx_tot * src.st.ny_tot * src.st.nz_tot;
    dst.data = tensorium_RG::aligned_alloc_n<double>(total);
}

void alloc_vector_rhs(const tensorium_RG::Field3D<double> src[3],
                      tensorium_RG::Field3D<double>      dst[3]) {
    const size_t total = src[0].st.nx_tot * src[0].st.ny_tot * src[0].st.nz_tot;
    for (int c = 0; c < 3; ++c) {
        dst[c].st = src[c].st;
        dst[c].data = tensorium_RG::aligned_alloc_n<double>(total);
    }
}

void add_debug_perturbation(tensorium_RG::Z4cGridSoA<double> &grid, double eps) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t id = grid.B[0].idx(i, j, k);
                grid.B[0].ptr()[id] += eps;
            }
}

void apply_gauge_update(tensorium_RG::Z4cGridSoA<double> &grid, const tensorium_RG::Field3D<double> &rhs_alpha,
                        const tensorium_RG::Field3D<double> rhs_beta[3],
                        const tensorium_RG::Field3D<double> rhs_B[3],
                        const tensorium_RG::Field3D<double> rhs_Gamma[3], double dt,
                        size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i0 = tensorium_RG::z4c::clamped_lower(I0, padding, I1);
    const size_t j0 = tensorium_RG::z4c::clamped_lower(J0, padding, J1);
    const size_t k0 = tensorium_RG::z4c::clamped_lower(K0, padding, K1);
    const size_t i1 = tensorium_RG::z4c::clamped_upper(I1, padding, I0);
    const size_t j1 = tensorium_RG::z4c::clamped_upper(J1, padding, J0);
    const size_t k1 = tensorium_RG::z4c::clamped_upper(K1, padding, K0);

#pragma omp parallel for collapse(2)
    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t id = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[id] += dt * rhs_alpha.ptr()[id];
                for (int c = 0; c < 3; ++c) {
                    grid.beta[c].ptr()[id] += dt * rhs_beta[c].ptr()[id];
                    grid.B[c].ptr()[id] += dt * rhs_B[c].ptr()[id];
                    grid.tildeGamma[c].ptr()[id] += dt * rhs_Gamma[c].ptr()[id];
                }
            }
}

struct FarStats {
    double max_far = 0.0;
    double l2_far = 0.0;
    size_t samples = 0;
    double max_global = 0.0;
};

template <typename FarSelector>
FarStats far_region_stats(const tensorium_RG::Z4cGridSoA<double> &grid,
                          const tensorium_RG::Field3D<double> &rhs_alpha,
                          const tensorium_RG::Field3D<double>  rhs_beta[3],
                          const tensorium_RG::Field3D<double>  rhs_B[3], size_t padding,
                          FarSelector is_far) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    constexpr double n_components = 7.0;

    FarStats stats;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const bool far = is_far(x, y, z);

                const size_t id = rhs_alpha.idx(i, j, k);
                const double alpha_val = rhs_alpha.ptr()[id];
                const double abs_alpha = std::abs(alpha_val);
                stats.max_global = std::max(stats.max_global, abs_alpha);

                double local_sq = alpha_val * alpha_val;
                bool   counted = far;
                if (far) {
                    stats.max_far = std::max(stats.max_far, abs_alpha);
                }

                for (int c = 0; c < 3; ++c) {
                    const double beta_val = rhs_beta[c].ptr()[rhs_beta[c].idx(i, j, k)];
                    const double B_val = rhs_B[c].ptr()[rhs_B[c].idx(i, j, k)];
                    const double abs_beta = std::abs(beta_val);
                    const double abs_B = std::abs(B_val);
                    stats.max_global = std::max({stats.max_global, abs_beta, abs_B});
                    if (far) {
                        stats.max_far = std::max({stats.max_far, abs_beta, abs_B});
                        local_sq += beta_val * beta_val + B_val * B_val;
                    }
                }

                if (counted) {
                    stats.l2_far += local_sq;
                    stats.samples += 1;
                }
            }

    if (stats.samples > 0)
        stats.l2_far = std::sqrt(stats.l2_far / (stats.samples * n_components));
    else
        stats.l2_far = 0.0;

    return stats;
}

void export_all_slices(const tensorium_RG::Z4cGridSoA<double> &grid,
                       const tensorium_RG::Field3D<double> &rhs_alpha,
                       const tensorium_RG::Field3D<double> rhs_beta[3],
                       const tensorium_RG::Field3D<double> rhs_B[3],
                       const tensorium_RG::Field3D<double> rhs_Gamma[3],
                       const std::vector<SlicePlaneSpec> &specs, const std::string &label_slug) {
    for (const auto &spec : specs) {
        const std::string base = label_slug + "_" + spec.tag;
        export_scalar_slice(grid, rhs_alpha, spec, "rhs_alpha",
                            "rhs_alpha_slice_" + base + ".csv");
        export_vector_slices(grid, rhs_beta, spec, base, "rhs_beta");
        export_vector_slices(grid, rhs_B, spec, base, "rhs_B");
        export_vector_slices(grid, rhs_Gamma, spec, base, "rhs_Gamma");
    }
}

template <typename InitFn, typename FarSelector>
void run_case(const char *label, InitFn init, const DemoConfig &cfg, CaseOptions case_opts,
              FarSelector far_selector) {
    tensorium_RG::Z4cGridSoA<double> grid(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double half = 0.5 * cfg.nx * cfg.dx;
    grid.x0 = -half;
    grid.y0 = -half;
    grid.z0 = -half;

    init(grid);

    if (cfg.debug_perturb && case_opts.add_perturbation) {
        add_debug_perturbation(grid, cfg.perturb_eps);
        std::cout << "[gauge demo] applied perturbation eps=" << cfg.perturb_eps << '\n';
    }

    tensorium_RG::Field3D<double> rhs_alpha;
    tensorium_RG::Field3D<double> rhs_beta[3];
    tensorium_RG::Field3D<double> rhs_B[3];
    tensorium_RG::Field3D<double> rhs_Gamma[3];
    tensorium_RG::z4c::GaugeParameters<double> params;
    params.eta = cfg.eta;

    alloc_scalar_rhs(grid.alpha, rhs_alpha);
    alloc_vector_rhs(grid.beta, rhs_beta);
    alloc_vector_rhs(grid.B, rhs_B);
    alloc_vector_rhs(grid.tildeGamma, rhs_Gamma);

    tensorium_RG::z4c::compute_rhs_Gamma(grid, rhs_Gamma, grid.Z, grid.Theta, params,
                                          cfg.padding);
    tensorium_RG::z4c::compute_rhs_alpha(grid, rhs_alpha, cfg.padding);
    tensorium_RG::z4c::compute_rhs_beta(grid, rhs_beta, params, cfg.padding);
    tensorium_RG::z4c::compute_rhs_B(grid, rhs_Gamma, rhs_B, params, cfg.padding);

    if (case_opts.slice_requests.empty())
        case_opts.slice_requests.push_back({SlicePlane::XY, "xy_mid"});

    std::vector<SlicePlaneSpec> specs;
    specs.reserve(case_opts.slice_requests.size());
    for (const auto &req : case_opts.slice_requests)
        specs.push_back(realize_request(grid, req));

    const std::string label_slug = slugify_label(label);
    export_all_slices(grid, rhs_alpha, rhs_beta, rhs_B, rhs_Gamma, specs, label_slug);

    const auto stats = far_region_stats(grid, rhs_alpha, rhs_beta, rhs_B, cfg.padding, far_selector);
    std::cout << "[gauge demo] " << label << " max_rhs_far=" << stats.max_far
              << " L2_rhs_far=" << stats.l2_far << " samples=" << stats.samples << '\n';
    std::cout << "[gauge demo][debug] " << label << " max_rhs_full=" << stats.max_global << '\n';

    if (cfg.apply_update && case_opts.apply_update) {
        apply_gauge_update(grid, rhs_alpha, rhs_beta, rhs_B, rhs_Gamma, cfg.update_dt,
                           cfg.padding);
        const double min_extent =
            std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                      (grid.dims.nz - 1) * grid.dz});
        const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
        const double r_max = 0.45 * min_extent;
        auto                         H_tmp = tensorium_RG::make_field(grid.alpha.st);
        tensorium_RG::Field3D<double> M_tmp[3];
        tensorium_RG::Field3D<double> C_tmp[3];
        for (int q = 0; q < 3; ++q) {
            M_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
            C_tmp[q] = tensorium_RG::make_field(grid.alpha.st);
        }
        const auto monitor = tensorium_RG::z4c::project_and_monitor(grid, H_tmp, M_tmp, C_tmp,
                                                                     r_min, r_max, 0.0, 0.0, 0.0,
                                                                     cfg.padding);
        tensorium_RG::z4c::print_constraint_monitor(
            monitor, (std::string("gauge demo update ") + label).c_str());
    }
}

} // namespace

int main() {
    DemoConfig cfg;
    const double r_cut = 8.0 * cfg.dx;
    const double r_cut2 = r_cut * r_cut;

    std::cout << "[gauge demo] grid=" << cfg.nx << "^3 dx=" << cfg.dx << " eta=" << cfg.eta
              << '\n';

    run_case("Minkowski", [](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, cfg,
             CaseOptions{.slice_requests = {{SlicePlane::XY, "xy_mid"}},
                         .add_perturbation = true},
             [](double, double, double) { return true; });

    run_case("Schwarzschild", [](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
             cfg,
             CaseOptions{.slice_requests = {{SlicePlane::XY, "xy_mid"}}},
             [=](double x, double y, double z) { return (x * x + y * y + z * z) > r_cut2; });

    run_case(
        "BowenYork",
        [&](auto &g) {
            const double P1[3] = {0.0, 0.05, 0.0};
            const double P2[3] = {0.0, -0.05, 0.0};
            const double S1[3] = {0.0, 0.0, 0.15};
            const double S2[3] = {0.0, 0.0, -0.15};
            try {
                tensorium_RG::init::binary_bowen_york_puncture_init(
                    g, 0.5, -1.5, 0.0, 0.0, P1, S1, 0.5, 1.5, 0.0, 0.0, P2, S2, 1e-4);
            } catch (const std::exception &e) {
                std::cerr << "[gauge demo] BowenYork warning: " << e.what() << '\n';
            }
        },
        cfg,
        CaseOptions{.slice_requests = {{SlicePlane::XY, "xy_mid"},
                                       {SlicePlane::YZ, "yz_xneg", -1.5},
                                       {SlicePlane::YZ, "yz_xpos", 1.5}},
                    .apply_update = cfg.apply_update},
        [=](double x, double y, double z) { return (x * x + y * y + z * z) > r_cut2; });

    std::cout << "[gauge demo] done" << std::endl;
    return 0;
}
