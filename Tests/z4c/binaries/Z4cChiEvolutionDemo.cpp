#include "../../framework/Assertions.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionChi.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <fstream>
#include <iostream>

namespace {

struct DemoConfig {
    size_t nx = 64;
    size_t ng = 4;
    double dx = 0.2;
};

void export_slice(const tensorium_RG::Z4cGridSoA<double> &grid,
                  const tensorium_RG::Field3D<double> &rhs, const char *path) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t js = J0 + grid.dims.ny / 2;

    std::ofstream out(path);
    out << "x,z,alpha,chi,rhs_chi\n";
    for (size_t i = I0; i < I1; ++i) {
        for (size_t k = K0; k < K1; ++k) {
            const size_t id = grid.chi.idx(i, js, k);
            double       x, y, z;
            grid.coords(i, js, k, x, y, z);
            out << x << ',' << z << ',' << grid.alpha.ptr()[id] << ',' << grid.chi.ptr()[id] << ','
                << rhs.ptr()[id] << '\n';
        }
    }
    std::cout << "[chi demo] wrote slice to " << path << '\n';
}

template <typename InitFn>
void run_one(const char *label, InitFn init, DemoConfig cfg, const char *csv_path) {
    tensorium_RG::Z4cGridSoA<double> grid(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double                      half = 0.5 * cfg.nx * cfg.dx;
    grid.x0 = -half;
    grid.y0 = -half;
    grid.z0 = -half;

    init(grid);

    tensorium_RG::Field3D<double> rhs;
    rhs.st = grid.chi.st;
    rhs.data = tensorium_RG::aligned_alloc_n<double>(rhs.st.nx_tot * rhs.st.ny_tot * rhs.st.nz_tot);

    tensorium_RG::z4c::compute_rhs_chi(grid, rhs);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    double max_rhs = 0.0;
    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k)
                max_rhs = std::max(max_rhs, std::abs(rhs.ptr()[grid.chi.idx(i, j, k)]));

    std::cout << "[chi demo] " << label << " max_rhs=" << max_rhs << '\n';
    if (csv_path)
        export_slice(grid, rhs, csv_path);
}

} // namespace

int main() {
    DemoConfig cfg;
    std::cout << "[chi demo] grid=" << cfg.nx << "^3 dx=" << cfg.dx << '\n';

    run_one(
        "Schwarzschild",
        [](tensorium_RG::Z4cGridSoA<double> &g) {
            tensorium_RG::init::schwarzschild_isotropic(g, 1.0);
        },
        cfg, "chi_rhs_slice_schwarzschild.csv");

    const double                      P1[3] = {0.0, 0.08, 0.0};
    const double                      P2[3] = {0.0, -0.08, 0.0};
    const double                      S1[3] = {0.0, 0.0, 0.15};
    const double                      S2[3] = {0.0, 0.0, -0.15};
    tensorium_RG::Z4cGridSoA<double> g(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double                      half = 0.5 * cfg.nx * cfg.dx;
    g.x0 = -half;
    g.y0 = -half;
    g.z0 = -half;
    try {
        tensorium_RG::init::binary_bowen_york_puncture_init(g, 0.5, -1.6, 0.0, 0.0, P1, S1, 0.5,
                                                            1.6, 0.0, 0.0, P2, S2, 1e-4);
    } catch (const std::exception &e) {
        std::cerr << "[chi demo] BowenYork init warning: " << e.what() << '\n';
    }
    tensorium_RG::Field3D<double> rhs;
    rhs.st = g.chi.st;
    rhs.data = tensorium_RG::aligned_alloc_n<double>(rhs.st.nx_tot * rhs.st.ny_tot * rhs.st.nz_tot);
    tensorium_RG::z4c::compute_rhs_chi(g, rhs);
    double max_rhs = 0.0;
    size_t I0, I1, J0, J1, K0, K1;
    g.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k)
                max_rhs = std::max(max_rhs, std::abs(rhs.ptr()[g.chi.idx(i, j, k)]));
    std::cout << "[chi demo] BowenYork max_rhs=" << max_rhs << '\n';
    export_slice(g, rhs, "chi_rhs_slice_bowen.csv");

    std::cout << "[chi demo] done" << std::endl;
    return 0;
}
