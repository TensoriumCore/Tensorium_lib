#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGamma.hpp"
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
    size_t padding = 4;
};

void alloc_rhs(const tensorium_RG::BSSNGridSoA<double> &grid,
               tensorium_RG::Field3D<double>            rhs[3]) {
    const size_t total = grid.tildeGamma[0].st.nx_tot * grid.tildeGamma[0].st.ny_tot *
                         grid.tildeGamma[0].st.nz_tot;
    for (int c = 0; c < 3; ++c) {
        rhs[c].st = grid.tildeGamma[c].st;
        rhs[c].data = tensorium_RG::aligned_alloc_n<double>(total);
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
                          const tensorium_RG::Field3D<double> rhs[3], size_t padding,
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

                double local_sq = 0.0;
                bool   counted = false;
                for (int c = 0; c < 3; ++c) {
                    const double val = rhs[c].ptr()[rhs[c].idx(i, j, k)];
                    const double abs_val = std::abs(val);
                    stats.max_global = std::max(stats.max_global, abs_val);
                    if (!far)
                        continue;
                    stats.max_far = std::max(stats.max_far, abs_val);
                    local_sq += val * val;
                    counted = true;
                }
                if (counted) {
                    stats.l2_far += local_sq;
                    stats.samples += 1;
                }
            }

    if (stats.samples > 0)
        stats.l2_far = std::sqrt(stats.l2_far / (stats.samples * 3.0));
    else
        stats.l2_far = 0.0;

    return stats;
}

void export_rhs_slice(const tensorium_RG::BSSNGridSoA<double> &grid,
                      const tensorium_RG::Field3D<double>      rhs[3],
                      const std::string &                      path) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ks = K0 + grid.dims.nz / 2;

    std::ofstream out(path);
    out << "x,y,rhs_Gamma_x\n";
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j) {
            const size_t id = rhs[0].idx(i, j, ks);
            double       x, y, z;
            grid.coords(i, j, ks, x, y, z);
            (void)z;
            out << x << ',' << y << ',' << rhs[0].ptr()[id] << '\n';
        }
}

template <typename InitFn, typename FarSelector>
void run_case(const char *label, InitFn init, const DemoConfig &cfg, const char *slice_path,
              FarSelector far_selector) {
    tensorium_RG::BSSNGridSoA<double> grid(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double half = 0.5 * cfg.nx * cfg.dx;
    grid.x0 = -half;
    grid.y0 = -half;
    grid.z0 = -half;

    init(grid);

    tensorium_RG::Field3D<double> rhs[3];
    alloc_rhs(grid, rhs);

    tensorium_RG::bssn::GaugeParameters<double> params;
    tensorium_RG::bssn::compute_rhs_Gamma(grid, rhs, grid.Z, grid.Theta, params, cfg.padding);

    const auto stats = far_region_stats(grid, rhs, cfg.padding, far_selector);
    std::cout << "[Gamma demo] " << label << " max_rhs_far=" << stats.max_far
              << " L2_rhs_far=" << stats.l2_far << " samples=" << stats.samples << '\n';
    std::cout << "[Gamma demo][debug] " << label << " max_rhs_full=" << stats.max_global
              << '\n';

    if (slice_path)
        export_rhs_slice(grid, rhs, slice_path);
}

} // namespace

int main() {
    DemoConfig cfg;
    const double r_cut = 8.0 * cfg.dx;
    const double r_cut2 = r_cut * r_cut;

    std::cout << "[Gamma demo] grid=" << cfg.nx << "^3 dx=" << cfg.dx << '\n';

    run_case("Minkowski", [](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, cfg, nullptr,
             [](double, double, double) { return true; });

    run_case("Schwarzschild", [](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
             cfg, "rhs_Gamma_slice_schwarzschild.csv",
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
                std::cerr << "[Gamma demo] BowenYork warning: " << e.what() << '\n';
            }
        },
        cfg, "rhs_Gamma_slice_bowen.csv",
        [=](double x, double y, double z) {
            const double r1 = (x + 1.5) * (x + 1.5) + y * y + z * z;
            const double r2 = (x - 1.5) * (x - 1.5) + y * y + z * z;
            return r1 > r_cut2 && r2 > r_cut2;
        });

    std::cout << "[Gamma demo] done" << std::endl;
    return 0;
}
