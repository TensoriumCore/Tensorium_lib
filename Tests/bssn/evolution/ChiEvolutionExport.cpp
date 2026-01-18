#include "../../framework/Assertions.hpp"
#include "../BSSNTestUtils.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Evolution/BSSNEvolutionChi.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <fstream>
#include <iostream>

int main() {
    constexpr size_t NX = 32;
    constexpr size_t NG = 4;
    constexpr double DX = 0.25;

    tensorium_RG::BSSNGridSoA<double> grid(NX, NX, NX, NG, DX, DX, DX);
    tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

    tensorium_RG::Field3D<double> rhs;
    rhs.st = grid.chi.st;
    const size_t total = rhs.st.nx_tot * rhs.st.ny_tot * rhs.st.nz_tot;
    rhs.data = tensorium_RG::aligned_alloc_n<double>(total);

    tensorium_RG::bssn::compute_rhs_chi(grid, rhs);

    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);

    double max_rhs = 0.0;
    for (size_t i = I0 + 4; i < I1 - 4; ++i)
        for (size_t j = J0 + 4; j < J1 - 4; ++j)
            for (size_t k = K0 + 4; k < K1 - 4; ++k)
                max_rhs = std::max(max_rhs, std::abs(rhs.ptr()[grid.chi.idx(i, j, k)]));

    std::cout << "[chi] max rhs interior = " << max_rhs << "\n";

    const size_t js = J0 + grid.dims.ny / 2;
    std::ofstream slice("chi_rhs_slice.csv");
    slice << "x,z,alpha,chi,rhs_chi\n";
    for (size_t i = I0; i < I1; ++i) {
        for (size_t k = K0; k < K1; ++k) {
            const size_t id = grid.chi.idx(i, js, k);
            double x, y, z;
            grid.coords(i, js, k, x, y, z);
            slice << x << ',' << z << ',' << grid.alpha.ptr()[id] << ',' << grid.chi.ptr()[id]
                  << ',' << rhs.ptr()[id] << '\n';
        }
    }
    std::cout << "[chi] wrote chi_rhs_slice.csv (Schwarzschild slice y=0)" << std::endl;
    return 0;
}

