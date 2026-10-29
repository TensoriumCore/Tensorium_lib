#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"
#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
std::array<double, 3> sommerfeld_errors(size_t n) {
    using namespace tensorium_RG;
    namespace bssn = tensorium_RG::bssn;
    constexpr size_t    ng = 4;
    constexpr double    extent = 10.0, frequency = 0.47;
    const double        h = extent / double(n - 1);
    BSSNGridSoA<double> grid(n, n, n, ng, h, h, h);
    grid.x0 = grid.y0 = grid.z0 = -extent / 2.0;
    const size_t total = grid.st.nx_tot * grid.st.ny_tot * grid.st.nz_tot;
    for (size_t id = 0; id < total; ++id) {
        grid.alpha.ptr()[id] = grid.chi.ptr()[id] = 1.0;
        grid.K.ptr()[id] = 0.0;
        for (int c = 0; c < 3; ++c)
            grid.beta[c].ptr()[id] = grid.B[c].ptr()[id] = grid.tildeGamma[c].ptr()[id] =
                grid.Z[c].ptr()[id] = 0.0;
        for (int s = 0; s < 6; ++s) {
            grid.gamma_tilde[s].ptr()[id] = (s == XX || s == YY || s == ZZ) ? 1.0 : 0.0;
            grid.A_tilde[s].ptr()[id] = 0.0;
        }
    }
    for (size_t i = 0; i < grid.st.nx_tot; ++i)
        for (size_t j = 0; j < grid.st.ny_tot; ++j)
            for (size_t k = 0; k < grid.st.nz_tot; ++k) {
                double x, y, z;
                grid.coords(i, j, k, x, y, z);
                const double r = std::sqrt(x * x + y * y + z * z);
                grid.Theta.ptr()[grid.Theta.idx(i, j, k)] =
                    r > 1e-14 ? -std::sin(frequency * r) / r : -frequency;
            }
    bssn::BSSNRHSWorkspace<double> rhs;
    rhs.allocate_like(grid);
    bssn::RHSBoundaryFaceMask mask{};
    for (int axis = 0; axis < 3; ++axis)
        mask.face[axis][0] = mask.face[axis][1] = true;
    bssn::apply_z4c_rhs_boundary(grid, rhs, mask, 1);
    const size_t          lo = ng, mid = ng + (n - 1) / 2;
    const size_t          points[3][3] = {{lo, mid, mid}, {lo, lo, mid}, {lo, lo, lo}};
    std::array<double, 3> errors{};
    for (int q = 0; q < 3; ++q) {
        double x, y, z;
        grid.coords(points[q][0], points[q][1], points[q][2], x, y, z);
        const double r = std::sqrt(x * x + y * y + z * z);
        const double exact = frequency * std::cos(frequency * r) / r;
        const double actual =
            rhs.Theta.ptr()[rhs.Theta.idx(points[q][0], points[q][1], points[q][2])];
        TENSORIUM_TEST_ASSERT(std::isfinite(actual));
        errors[q] = std::abs(actual - exact);
    }
    return errors;
}
} // namespace

REGISTER_TEST("z4c.bc.sommerfeld_outgoing_wave_convergence",
              "Sommerfeld RHS converges for a spherical wave on faces, edges and corners", []() {
                  const auto coarse = sommerfeld_errors(11);
                  const auto medium = sommerfeld_errors(21);
                  const auto fine = sommerfeld_errors(41);
                  for (int q = 0; q < 3; ++q) {
                      std::printf("[Sommerfeld convergence %d] errors %.9g %.9g %.9g\n", q,
                                  coarse[q], medium[q], fine[q]);
                      TENSORIUM_TEST_ASSERT(coarse[q] > 2.8 * medium[q]);
                      TENSORIUM_TEST_ASSERT(medium[q] > 2.8 * fine[q]);
                  }
              });
