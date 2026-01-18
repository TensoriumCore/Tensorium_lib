#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Derivatives/BSSNGridDerivatives.hpp"
#include "../../includes/Tensorium_Grid/Grid/GridLayout.hpp"

#include <cmath>

using namespace tensorium_RG;

REGISTER_TEST("grid.derivatives.fourth_order", "Central derivatives on a sinusoid", []() {
    const size_t nx = 32;
    const size_t ng = 4;
    const double dx = 1.0 / nx;
    constexpr double two_pi = 6.28318530717958647692;

    Strides<double> st;
    st.nx_tot = nx + 2 * ng;
    st.ny_tot = nx + 2 * ng;
    st.nz_tot = nx + 2 * ng;
    st.sz = 1;
    st.sy = st.nz_tot;
    st.sx = st.sy * st.ny_tot;

    Field3D<double> field = make_field<double>(st);
    GridDims dims{nx, nx, nx, ng};

    for (size_t i = 0; i < st.nx_tot; ++i)
        for (size_t j = 0; j < st.ny_tot; ++j)
            for (size_t k = 0; k < st.nz_tot; ++k) {
                const double x = (static_cast<double>(i) - ng) * dx;
                field.ptr()[field.idx(i, j, k)] = std::sin(two_pi * x);
            }

    double max_err = 0.0;
    for (size_t i = ng + 2; i < ng + nx - 2; ++i) {
        for (size_t j = ng + 2; j < ng + nx - 2; ++j) {
            for (size_t k = ng + 2; k < ng + nx - 2; ++k) {
                const double exact = two_pi * std::cos(two_pi * (i - ng) * dx);
                const double numeric = fd::Dx(field, i, j, k, dx);
                max_err = std::max(max_err, std::abs(exact - numeric));
            }
        }
    }

    tensorium::tests::expect_le(max_err, 5e-4, "fd derivative error");
});
