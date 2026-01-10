#ifndef TENSORIUM_BSSN_GAMMA_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_GAMMA_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionGammaTilde.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <fstream>

namespace {

double max_rhs_gamma(const tensorium_RG::BSSNGridSoA<double> &grid, tensorium_RG::Field3D<double> rhs[6]) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    double max_val = 0.0;
    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k) {
                const size_t id = grid.gamma_tilde[0].idx(i, j, k);
                for (int s = 0; s < 6; ++s)
                    max_val = std::max(max_val, std::abs(rhs[s].ptr()[id]));
            }
    return max_val;
}

void export_gamma_slice(const tensorium_RG::BSSNGridSoA<double> &grid,
                        tensorium_RG::Field3D<double> rhs[6], const char *path) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    const size_t js = J0 + grid.dims.ny / 2;

    std::ofstream out(path);
    out << "x,z,gxx,gxy,gxz,gyy,gyz,gzz,rhs_xx,rhs_xy,rhs_xz,rhs_yy,rhs_yz,rhs_zz\n";
    for (size_t i = I0; i < I1; ++i)
        for (size_t k = K0; k < K1; ++k) {
            const size_t id = grid.gamma_tilde[0].idx(i, js, k);
            double x, y, z;
            grid.coords(i, js, k, x, y, z);
            out << x << ',' << z;
            for (int s = 0; s < 6; ++s)
                out << ',' << grid.gamma_tilde[s].ptr()[id];
            for (int s = 0; s < 6; ++s)
                out << ',' << rhs[s].ptr()[id];
            out << '\n';
        }
}

} // namespace

REGISTER_TEST("bssn.evolution.gamma_stationary", "Gamma_tilde RHS near zero for stationary data", []() {
    auto run_case = [](auto init_fn, const char *csv_path, const char *label) {
        tensorium_RG::BSSNGridSoA<double> grid(32, 32, 32, 4, 0.25, 0.25, 0.25);
        init_fn(grid);

        tensorium_RG::Field3D<double> rhs[6];
        tensorium_RG::bssn::compute_rhs_gamma_tilde(grid, rhs);

        const auto tol = tensorium_RG::bssn::compute_invariant_tolerances(grid);
        double max_val = max_rhs_gamma(grid, rhs);
        tensorium::tests::expect_le(max_val, 5.0 * tol.metric_tol,
                                    std::string("rhs gamma stationary (") + label + ")");

        if (csv_path)
            export_gamma_slice(grid, rhs, csv_path);
    };

    run_case([](auto &g) { tensorium_RG::init::minkowski(g, 0.0); }, nullptr, "minkowski");
    run_case([](auto &g) { tensorium_RG::init::schwarzschild_isotropic(g, 1.0); },
             "gamma_rhs_slice_schwarzschild.csv", "schwarzschild");
});

#endif
