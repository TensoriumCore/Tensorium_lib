#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionATilde.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

namespace {

struct DemoConfig {
    size_t nx = 128;
    size_t ng = 4;
    double dx = 0.25;
};

void alloc_rhs(const tensorium_RG::BSSNGridSoA<double> &grid,
               tensorium_RG::Field3D<double>            rhs[6]) {
    const size_t total =
        grid.A_tilde[0].st.nx_tot * grid.A_tilde[0].st.ny_tot * grid.A_tilde[0].st.nz_tot;
    for (int s = 0; s < 6; ++s) {
        rhs[s].st = grid.A_tilde[s].st;
        rhs[s].data = tensorium_RG::aligned_alloc_n<double>(total);
    }
}

double interior_max_rhs(const tensorium_RG::BSSNGridSoA<double> &grid,
                        const tensorium_RG::Field3D<double> rhs[6], size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    double max_val = 0.0;
    for (size_t i = I0 + padding; i < I1 - padding; ++i)
        for (size_t j = J0 + padding; j < J1 - padding; ++j)
            for (size_t k = K0 + padding; k < K1 - padding; ++k) {
                const size_t id = rhs[0].idx(i, j, k);
                for (int s = 0; s < 6; ++s)
                    max_val = std::max(max_val, std::abs(rhs[s].ptr()[id]));
            }
    return max_val;
}

void export_rhs_slice(const tensorium_RG::BSSNGridSoA<double> &grid,
                      const tensorium_RG::Field3D<double> rhs[6], const std::string &path) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ks = K0 + grid.dims.ny / 2;

    std::ofstream out(path);
    out << "x,y,rhs_Axx\n";
    for (size_t i = I0; i < I1; ++i) {
        for (size_t j = J0; j < J1; ++j) {
            const size_t id = rhs[tensorium_RG::XX].idx(i, j, ks);
            double       x, y, z;
            grid.coords(i, j, ks, x, y, z);
            (void)z;
            out << x << ',' << y << ',' << rhs[tensorium_RG::XX].ptr()[id] << '\n';
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
FarStats far_region_stats(const tensorium_RG::BSSNGridSoA<double> &grid,
                          const tensorium_RG::Field3D<double> rhs[6], size_t padding,
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
                const bool far = is_far(x, y, z);
                for (int s = 0; s < 6; ++s) {
                    const double val = rhs[s].ptr()[rhs[s].idx(i, j, k)];
                    const double abs_val = std::abs(val);
                    stats.max_global = std::max(stats.max_global, abs_val);
                    if (!far)
                        continue;
                    stats.max_far = std::max(stats.max_far, abs_val);
                    stats.l2_far += val * val;
                }
                if (far)
                    stats.samples += 1;
            }
    if (stats.samples > 0)
        stats.l2_far = std::sqrt(stats.l2_far / (stats.samples * 6.0));
    else
        stats.l2_far = 0.0;
    return stats;
}

double metric_inverse_residual(const tensorium_RG::BSSNGridSoA<double> &grid, size_t padding) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t i_begin = I0 + padding;
    const size_t i_end = I1 - padding;
    const size_t j_begin = J0 + padding;
    const size_t j_end = J1 - padding;
    const size_t k_begin = K0 + padding;
    const size_t k_end = K1 - padding;

    double max_err = 0.0;
    for (size_t i = i_begin; i < i_end; ++i)
        for (size_t j = j_begin; j < j_end; ++j)
            for (size_t k = k_begin; k < k_end; ++k) {
                const size_t id = grid.gamma_tilde[tensorium_RG::XX].idx(i, j, k);
                double       g[3][3];
                double       ginv[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b) {
                        const int    idx = tensorium_RG::sym6_index(a, b);
                        const double gv = grid.gamma_tilde[idx].ptr()[id];
                        const double giv = grid.gamma_tilde_inv[idx].ptr()[id];
                        g[a][b] = gv;
                        g[b][a] = gv;
                        ginv[a][b] = giv;
                        ginv[b][a] = giv;
                    }
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double prod = 0.0;
                        for (int m = 0; m < 3; ++m)
                            prod += g[a][m] * ginv[m][b];
                        const double target = (a == b) ? 1.0 : 0.0;
                        max_err = std::max(max_err, std::abs(prod - target));
                    }
            }
    return max_err;
}

template <typename InitFn, typename FarSelector>
void run_case(const char *label, InitFn init, const DemoConfig &cfg, const std::string &csv_path,
              size_t padding, FarSelector far_selector) {
    tensorium_RG::BSSNGridSoA<double> grid(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double                      half = 0.5 * cfg.nx * cfg.dx;
    grid.x0 = -half;
    grid.y0 = -half;
    grid.z0 = -half;

    init(grid);

    tensorium_RG::Field3D<double> rhs[6];
    alloc_rhs(grid, rhs);

    tensorium_RG::bssn::compute_rhs_A_tilde(grid, rhs, padding);

    const double metric_err = metric_inverse_residual(grid, padding);
    const auto   stats = far_region_stats(grid, rhs, padding, far_selector);
    std::cout << "[atilde demo] " << label << " max_rhs_far=" << stats.max_far
              << " L2_rhs_far=" << stats.l2_far << " samples=" << stats.samples
              << " metric_residual=" << metric_err << '\n';
    std::cout << "[atilde demo][debug] " << label << " max_rhs_full=" << stats.max_global << '\n';

    export_rhs_slice(grid, rhs, csv_path);
}

} // namespace

int main() {
    DemoConfig   cfg;
    const size_t padding = 4;
    std::cout << "[atilde demo] grid=" << cfg.nx << "^3 dx=" << cfg.dx << '\n';

    const double r_cut = 8.0 * cfg.dx;
    const double r_cut2 = r_cut * r_cut;

    run_case("Minkowski", [](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, cfg,
             "rhs_Axx_slice_minkowski.csv", padding,
             [](double, double, double) { return true; });

    run_case(
        "Schwarzschild", [](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); }, cfg,
        "rhs_Axx_slice_schwarzschild.csv", padding,
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
                std::cerr << "[atilde demo] BowenYork warning: " << e.what() << '\n';
            }
        },
        cfg, "rhs_Axx_slice_bowen.csv", padding,
        [=](double x, double y, double z) {
            const double r1 = (x + 1.5) * (x + 1.5) + y * y + z * z;
            const double r2 = (x - 1.5) * (x - 1.5) + y * y + z * z;
            return r1 > r_cut2 && r2 > r_cut2;
        });

    std::cout << "[atilde demo] slices written" << std::endl;
    return 0;
}
