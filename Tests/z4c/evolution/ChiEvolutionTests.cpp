#ifndef TENSORIUM_BSSN_CHI_EVOLUTION_TESTS_CPP
#define TENSORIUM_BSSN_CHI_EVOLUTION_TESTS_CPP

#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../Z4cTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionChi.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <fstream>

REGISTER_TEST("z4c.evolution.chi_stationary", "Chi RHS vanishes on stationary data", []() {
    auto check_grid = [](tensorium_RG::Z4cGridSoA<double> &grid, const char *label,
                         const char *slice_csv) {
        tensorium_RG::Field3D<double> rhs;
        rhs.st = grid.chi.st;
        rhs.data = tensorium_RG::aligned_alloc_n<double>(grid.chi.st.nx_tot * grid.chi.st.ny_tot *
                                                         grid.chi.st.nz_tot);

        tensorium_RG::z4c::compute_rhs_chi(grid, rhs);

        size_t I0, I1, J0, J1, K0, K1;
        grid.domain_bounds(I0, I1, J0, J1, K0, K1);

        const auto tol = tensorium_RG::z4c::compute_invariant_tolerances(grid);

        double max_rhs = 0.0;
        for (size_t i = I0 + 4; i < I1 - 4; ++i)
            for (size_t j = J0 + 4; j < J1 - 4; ++j)
                for (size_t k = K0 + 4; k < K1 - 4; ++k) {
                    const size_t id = grid.chi.idx(i, j, k);
                    max_rhs = std::max(max_rhs, std::abs(rhs.ptr()[id]));
                }

        tensorium::tests::expect_le(max_rhs, 5.0 * tol.metric_tol,
                                    std::string("rhs chi stationary (") + label + ")");

        if (slice_csv) {
            const size_t js = J0 + grid.dims.ny / 2;
            std::ofstream slice(slice_csv);
            slice << "x,z,alpha,chi,rhs_chi\n";
            for (size_t i = I0; i < I1; ++i) {
                for (size_t k = K0; k < K1; ++k) {
                    const size_t id = grid.chi.idx(i, js, k);
                    double x, y, z;
                    grid.coords(i, js, k, x, y, z);
                    slice << x << ',' << z << ',' << grid.alpha.ptr()[id] << ','
                          << grid.chi.ptr()[id] << ',' << rhs.ptr()[id] << '\n';
                }
            }
        }
    };

    tensorium_RG::Z4cGridSoA<double> grid_flat(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::minkowski(grid_flat, 0.0);
    check_grid(grid_flat, "minkowski", nullptr);

    tensorium_RG::Z4cGridSoA<double> grid_schw(32, 32, 32, 4, 0.25, 0.25, 0.25);
    tensorium_RG::init::schwarzschild_isotropic(grid_schw, 1.0);
    check_grid(grid_schw, "schwarzschild", "chi_rhs_slice.csv");
});

#endif // TENSORIUM_BSSN_CHI_EVOLUTION_TESTS_CPP
