#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGammaTilde.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace {

struct DemoConfig {
    size_t nx = 128;
    size_t ng = 4;
    double dx = 0.25;
};

template <typename GridInit>
void run_case(const char *label, GridInit init, const DemoConfig &cfg, const std::string &gamma_csv,
              const std::string &rhs_csv) {
    tensorium_RG::BSSNGridSoA<double> grid(cfg.nx, cfg.nx, cfg.nx, cfg.ng, cfg.dx, cfg.dx, cfg.dx);
    const double                      half = 0.5 * cfg.nx * cfg.dx;
    grid.x0 = -half;
    grid.y0 = -half;
    grid.z0 = -half;

    init(grid);

    tensorium_RG::Field3D<double> rhs[6];
    for (int s = 0; s < 6; ++s) {
        rhs[s].st = grid.gamma_tilde[s].st;
        rhs[s].data = tensorium_RG::aligned_alloc_n<double>(rhs[s].st.nx_tot * rhs[s].st.ny_tot *
                                                            rhs[s].st.nz_tot);
    }

    tensorium_RG::bssn::compute_rhs_gamma_tilde(grid, rhs);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t ks = K0 + grid.dims.nz / 2;

    auto dump = [&](const std::string &path, const tensorium_RG::Field3D<double> &field,
                    const char *name) {
        std::ofstream out(path);
        out << "x,y," << name << "\n";
        for (size_t i = I0; i < I1; ++i) {
            for (size_t j = J0; j < J1; ++j) {
                const size_t id = field.idx(i, j, ks);
                double       x, y, z;
                grid.coords(i, j, ks, x, y, z);
                (void)z;
                out << x << ',' << y << ',' << field.ptr()[id] << '\n';
            }
        }
    };

    dump(gamma_csv, grid.gamma_tilde[tensorium_RG::XX], "gamma_xx");
    dump(rhs_csv, rhs[tensorium_RG::XX], "rhs_xx");

    double max_rhs = 0.0;
    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k)
                for (int s = 0; s < 6; ++s)
                    max_rhs = std::max(max_rhs, std::abs(rhs[s].ptr()[rhs[s].idx(i, j, k)]));

    std::cout << "[gamma demo] " << label << " max_rhs=" << max_rhs << '\n';
}

} // namespace

int main() {
    DemoConfig cfg;
    std::cout << "[gamma demo] grid=" << cfg.nx << "^3 dx=" << cfg.dx << '\n';

    run_case(
        "Minkowski", [](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, cfg,
        "gamma_slice_minkowski.csv", "rhs_gamma_slice_minkowski.csv");

    run_case(
        "Schwarzschild", [](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); }, cfg,
        "gamma_slice_schwarzschild.csv", "rhs_gamma_slice_schwarzschild.csv");

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
                std::cerr << "[gamma demo] BowenYork warning: " << e.what() << '\n';
            }
        },
        cfg, "gamma_slice_bowen.csv", "rhs_gamma_slice_bowen.csv");

    std::cout << "[gamma demo] done" << std::endl;
    return 0;
}
