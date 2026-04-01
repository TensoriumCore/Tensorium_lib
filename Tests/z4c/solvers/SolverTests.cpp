#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Solvers/BSSNConstrainSolver.hpp"

#include <limits>

REGISTER_TEST("z4c.solvers.lichnerowicz_trivial",
              "SOR solver preserves flat solution when sources vanish", []() {
    tensorium_RG::Z4cGridSoA<double> grid(16, 16, 16, 4, 0.5, 0.5, 0.5);

    const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
    for (size_t idx = 0; idx < total; ++idx) {
        grid.alpha.ptr()[idx] = 1.0;
        grid.chi.ptr()[idx] = 1.0;
        grid.A_tilde[tensorium_RG::XX].ptr()[idx] = 0.0;
        grid.A_tilde[tensorium_RG::XY].ptr()[idx] = 0.0;
        grid.A_tilde[tensorium_RG::XZ].ptr()[idx] = 0.0;
        grid.A_tilde[tensorium_RG::YY].ptr()[idx] = 0.0;
        grid.A_tilde[tensorium_RG::YZ].ptr()[idx] = 0.0;
        grid.A_tilde[tensorium_RG::ZZ].ptr()[idx] = 0.0;

        grid.gamma_tilde[tensorium_RG::XX].ptr()[idx] = 1.0;
        grid.gamma_tilde[tensorium_RG::XY].ptr()[idx] = 0.0;
        grid.gamma_tilde[tensorium_RG::XZ].ptr()[idx] = 0.0;
        grid.gamma_tilde[tensorium_RG::YY].ptr()[idx] = 1.0;
        grid.gamma_tilde[tensorium_RG::YZ].ptr()[idx] = 0.0;
        grid.gamma_tilde[tensorium_RG::ZZ].ptr()[idx] = 1.0;

        grid.gamma_tilde_inv[tensorium_RG::XX].ptr()[idx] = 1.0;
        grid.gamma_tilde_inv[tensorium_RG::XY].ptr()[idx] = 0.0;
        grid.gamma_tilde_inv[tensorium_RG::XZ].ptr()[idx] = 0.0;
        grid.gamma_tilde_inv[tensorium_RG::YY].ptr()[idx] = 1.0;
        grid.gamma_tilde_inv[tensorium_RG::YZ].ptr()[idx] = 0.0;
        grid.gamma_tilde_inv[tensorium_RG::ZZ].ptr()[idx] = 1.0;

        for (int s = 0; s < 6; ++s)
            grid.Ricci[s].ptr()[idx] = 0.0;
    }

    tensorium_RG::init::solve_lichnerowicz_u_SOR(grid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 1e-6, 8, 1e-12, 1.6);

    double min_chi = 1e9;
    double max_chi = 0.0;
    for (size_t idx = 0; idx < total; ++idx) {
        min_chi = std::min(min_chi, grid.chi.ptr()[idx]);
        max_chi = std::max(max_chi, grid.chi.ptr()[idx]);
    }
    tensorium::tests::expect_near(min_chi, 1.0, 1e-8, "chi lower bound");
    tensorium::tests::expect_near(max_chi, 1.0, 1e-8, "chi upper bound");
});

REGISTER_TEST("z4c.solvers.bowen_york",
              "Binary Bowen-York initialisation keeps chi positive", []() {
    tensorium_RG::Z4cGridSoA<double> grid(20, 20, 20, 4, 0.4, 0.4, 0.4);
    const double P1[3] = {0.0, 0.0, 0.1};
    const double P2[3] = {0.0, 0.0, -0.1};
    const double S1[3] = {0.0, 0.2, 0.0};
    const double S2[3] = {0.0, -0.2, 0.0};

    tensorium_RG::init::binary_bowen_york_puncture_init(grid, 0.5, -1.0, 0.0, 0.0, P1, S1, 0.5, 1.0,
                                                        0.0, 0.0, P2, S2, 1e-4);

    double min_chi = std::numeric_limits<double>::max();
    double max_chi = 0.0;
    const size_t total = grid.alpha.st.nx_tot * grid.alpha.st.ny_tot * grid.alpha.st.nz_tot;
    for (size_t idx = 0; idx < total; ++idx) {
        min_chi = std::min(min_chi, grid.chi.ptr()[idx]);
        max_chi = std::max(max_chi, grid.chi.ptr()[idx]);
    }

    tensorium::tests::expect_le(max_chi, 1.0, "chi upper bound after solver");
    tensorium::tests::expect_le(1e-6, min_chi, "chi positivity");
});
