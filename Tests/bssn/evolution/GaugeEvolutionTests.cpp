#ifndef TENSORIUM_BSSN_GAUGE_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_GAUGE_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGamma.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGauge.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNProjectionMonitor.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>

namespace {

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

struct FarStats {
    double max_far = 0.0;
    size_t samples = 0;
};

double max_field_interior(const tensorium_RG::BSSNGridSoA<double> &grid,
                          const tensorium_RG::Field3D<double> &field, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    double max_val = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = field.idx(i, j, k);
                max_val = std::max(max_val, std::abs(field.ptr()[id]));
            }
    return max_val;
}

double vector_max_in_region(const tensorium_RG::BSSNGridSoA<double> &grid,
                            const tensorium_RG::Field3D<double> vec[3], double xc, double yc,
                            double zc, double radius, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    const double radius2 = radius * radius;
    double       max_val = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const double dx = x - xc;
                const double dy = y - yc;
                const double dz = z - zc;
                if (dx * dx + dy * dy + dz * dz > radius2)
                    continue;
                const size_t id = vec[0].idx(i, j, k);
                for (int c = 0; c < 3; ++c)
                    max_val = std::max(max_val, std::abs(vec[c].ptr()[id]));
            }
    return max_val;
}

double vector_max_far_region(const tensorium_RG::BSSNGridSoA<double> &grid,
                             const tensorium_RG::Field3D<double> vec[3], double r_cut,
                             size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    const double r_cut2 = r_cut * r_cut;
    double       max_val = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                if (x * x + y * y + z * z <= r_cut2)
                    continue;
                const size_t id = vec[0].idx(i, j, k);
                for (int c = 0; c < 3; ++c)
                    max_val = std::max(max_val, std::abs(vec[c].ptr()[id]));
            }
    return max_val;
}

void add_uniform_B_perturb(tensorium_RG::BSSNGridSoA<double> &grid, double eps) {
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

void add_linear_K_perturb(tensorium_RG::BSSNGridSoA<double> &grid, double eps) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)y;
                (void)z;
                const size_t id = grid.K.idx(i, j, k);
                grid.K.ptr()[id] += eps * x;
            }
}

void add_linear_beta_perturb(tensorium_RG::BSSNGridSoA<double> &grid, double eps) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)y;
                (void)z;
                const size_t id = grid.beta[0].idx(i, j, k);
                grid.beta[0].ptr()[id] += eps * x;
            }
}

void add_linear_alpha_perturb(tensorium_RG::BSSNGridSoA<double> &grid, double eps) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
#pragma omp parallel for collapse(2)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                (void)y;
                (void)z;
                const size_t id = grid.alpha.idx(i, j, k);
                grid.alpha.ptr()[id] += eps * x;
            }
}

template <typename FarSelector>
FarStats far_region_stats(const tensorium_RG::BSSNGridSoA<double> &grid,
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

    FarStats stats;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                if (!is_far(x, y, z))
                    continue;
                stats.samples += 1;
                const size_t id = rhs_alpha.idx(i, j, k);
                stats.max_far =
                    std::max(stats.max_far, std::abs(rhs_alpha.ptr()[id]));
                for (int c = 0; c < 3; ++c) {
                    stats.max_far = std::max(stats.max_far, std::abs(rhs_beta[c].ptr()[rhs_beta[c].idx(i, j, k)]));
                    stats.max_far = std::max(stats.max_far, std::abs(rhs_B[c].ptr()[rhs_B[c].idx(i, j, k)]));
                }
            }
    return stats;
}

double interior_max(const tensorium_RG::BSSNGridSoA<double> &grid,
                    const tensorium_RG::Field3D<double> &    rhs_alpha,
                    const tensorium_RG::Field3D<double>      rhs_beta[3],
                    const tensorium_RG::Field3D<double>      rhs_B[3], size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    double max_val = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = rhs_alpha.idx(i, j, k);
                max_val = std::max(max_val, std::abs(rhs_alpha.ptr()[id]));
                for (int c = 0; c < 3; ++c) {
                    max_val = std::max(max_val,
                                       std::abs(rhs_beta[c].ptr()[rhs_beta[c].idx(i, j, k)]));
                    max_val =
                        std::max(max_val, std::abs(rhs_B[c].ptr()[rhs_B[c].idx(i, j, k)]));
                }
            }
    return max_val;
}

} // namespace

struct GaugeCaseConfig {
    double far_tol;
    double trace_tol;
    double det_tol;
    double constraint_tol;
    bool   stationary = false;
    double gamma_far_tol = -1.0;
    double B_far_tol = -1.0;
    double gamma_min_global = -1.0;
    double B_min_global = -1.0;
    bool   perturb_B = false;
    bool   perturb_K = false;
    bool   perturb_beta = false;
    bool   perturb_alpha = false;
    double perturb_eps = 1e-6;
};

REGISTER_TEST("bssn.evolution.gauge_rhs", "Gauge RHS validation", []() {
    const size_t padding = 4;
    const double r_cut = 4.0;

    auto run_case = [&](auto init_fn, const char *label, const GaugeCaseConfig &cfg) {
        tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
        init_fn(grid);

        if (cfg.perturb_B)
            add_uniform_B_perturb(grid, cfg.perturb_eps);
        if (cfg.perturb_K)
            add_linear_K_perturb(grid, cfg.perturb_eps);
        if (cfg.perturb_beta)
            add_linear_beta_perturb(grid, cfg.perturb_eps);
        if (cfg.perturb_alpha)
            add_linear_alpha_perturb(grid, cfg.perturb_eps);

        tensorium_RG::Field3D<double> rhs_alpha;
        tensorium_RG::Field3D<double> rhs_beta[3];
        tensorium_RG::Field3D<double> rhs_B[3];
        tensorium_RG::Field3D<double> rhs_Gamma[3];
        tensorium_RG::bssn::GaugeParameters<double> params;

        alloc_scalar_rhs(grid.alpha, rhs_alpha);
        alloc_vector_rhs(grid.beta, rhs_beta);
        alloc_vector_rhs(grid.B, rhs_B);
        alloc_vector_rhs(grid.tildeGamma, rhs_Gamma);

        tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs_Gamma, padding);
        tensorium_RG::bssn::compute_rhs_alpha(grid, rhs_alpha, padding);
        tensorium_RG::bssn::compute_rhs_beta(grid, rhs_beta, params, padding);
        tensorium_RG::bssn::compute_rhs_B(grid, rhs_Gamma, rhs_B, params, padding);

        const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
        const double max_int = interior_max(grid, rhs_alpha, rhs_beta, rhs_B, padding);

        const double max_alpha = max_field_interior(grid, rhs_alpha, padding);
        const double max_beta0 = max_field_interior(grid, rhs_beta[0], padding);
        const double max_beta1 = max_field_interior(grid, rhs_beta[1], padding);
        const double max_beta2 = max_field_interior(grid, rhs_beta[2], padding);
        const double max_B0 = max_field_interior(grid, rhs_B[0], padding);
        const double max_B1 = max_field_interior(grid, rhs_B[1], padding);
        const double max_B2 = max_field_interior(grid, rhs_B[2], padding);

        if (cfg.stationary) {
            const double strict_tol = 5.0 * tol.metric_tol;
            tensorium::tests::expect_le(max_alpha, strict_tol,
                                        std::string("alpha stationary (") + label + ")");
            tensorium::tests::expect_le(std::max({max_beta0, max_beta1, max_beta2}), strict_tol,
                                        std::string("beta stationary (") + label + ")");
            tensorium::tests::expect_le(std::max({max_B0, max_B1, max_B2}), strict_tol,
                                        std::string("B stationary (") + label + ")");
        }

        if (cfg.far_tol == 0.0) {
            tensorium::tests::expect_le(max_int, 5.0 * tol.metric_tol,
                                        std::string("Gauge rhs stationary (") + label + ")");
        } else {
            const auto stats = far_region_stats(
                grid, rhs_alpha, rhs_beta, rhs_B, padding, [&](double x, double y, double z) {
                    return (x * x + y * y + z * z) > (r_cut * r_cut);
                });
            TENSORIUM_TEST_ASSERT(stats.samples > 0);
            tensorium::tests::expect_le(stats.max_far, cfg.far_tol,
                                        std::string("Gauge rhs far bound (") + label + ")");
        }

        if (cfg.gamma_min_global > 0.0) {
            const double near_max = vector_max_in_region(grid, rhs_Gamma, 0.0, 0.0, 0.0, 1e9,
                                                         padding);
            TENSORIUM_TEST_ASSERT(near_max > cfg.gamma_min_global);
        }
        if (cfg.B_min_global > 0.0) {
            const double near_max =
                vector_max_in_region(grid, rhs_B, 0.0, 0.0, 0.0, 1e9, padding);
            TENSORIUM_TEST_ASSERT(near_max > cfg.B_min_global);
        }
        if (cfg.gamma_far_tol > 0.0) {
            const double far_max = vector_max_far_region(grid, rhs_Gamma, r_cut, padding);
            tensorium::tests::expect_le(far_max, cfg.gamma_far_tol,
                                        std::string("rhs_Gamma far (") + label + ")");
        }
        if (cfg.B_far_tol > 0.0) {
            const double far_max = vector_max_far_region(grid, rhs_B, r_cut, padding);
            tensorium::tests::expect_le(far_max, cfg.B_far_tol,
                                        std::string("rhs_B far (") + label + ")");
        }

        const double min_extent = std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                                            (grid.dims.nz - 1) * grid.dz});
        const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
        const double r_max = 0.45 * min_extent;
        const auto   monitor = tensorium_RG::bssn::project_and_monitor(grid, grid.Hc, grid.Mc, grid.Cc,
                                                                       r_min, r_max, 0.0, 0.0, 0.0,
                                                                       padding);

        tensorium::tests::expect_le(monitor.max_trace_A, cfg.trace_tol,
                                    std::string("trace projection (") + label + ")");
        tensorium::tests::expect_le(monitor.max_det_drift, cfg.det_tol,
                                    std::string("det projection (") + label + ")");
        if (cfg.constraint_tol > 0.0)
            tensorium::tests::expect_le(monitor.max_H, cfg.constraint_tol,
                                        std::string("H constraint (") + label + ")");
    };

    run_case([](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, "minkowski",
             GaugeCaseConfig{.far_tol = 0.0,
                             .trace_tol = 1e-12,
                             .det_tol = 1e-12,
                             .constraint_tol = 1e-12,
                             .stationary = true});
    run_case([](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
             "schwarzschild",
             GaugeCaseConfig{.far_tol = 1e-2, .trace_tol = 5e-3, .det_tol = 5e-3, .constraint_tol = 5e-3});

    run_case(
        [&](auto &g) {
            const double P1[3] = {0.0, 0.05, 0.0};
            const double P2[3] = {0.0, -0.05, 0.0};
            const double S1[3] = {0.0, 0.0, 0.15};
            const double S2[3] = {0.0, 0.0, -0.15};
            tensorium_RG::init::binary_bowen_york_puncture_init(
                g, 0.5, -1.5, 0.0, 0.0, P1, S1, 0.5, 1.5, 0.0, 0.0, P2, S2, 1e-4);
        },
        "bowen_york",
        GaugeCaseConfig{.far_tol = 2e-2,
                        .trace_tol = 2e-2,
                        .det_tol = 2e-2,
                        .constraint_tol = 2e-2,
                        .gamma_far_tol = 2e-2,
                        .B_far_tol = 2e-2,
                        .gamma_min_global = 5e-7,
                        .B_min_global = 5e-7,
                        .perturb_B = true,
                        .perturb_K = true,
                        .perturb_beta = true,
                        .perturb_alpha = true,
                        .perturb_eps = 1e-6});
});

#endif
