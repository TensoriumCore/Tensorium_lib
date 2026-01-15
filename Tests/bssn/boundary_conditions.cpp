#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Geometry/BSSNInvariants.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/BSSNGridOperations.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <algorithm>
#include <cmath>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;

void center_grid(Grid &grid) {
    const double hx = 0.5 * (static_cast<double>(grid.dims.nx) - 1.0) * grid.dx;
    const double hy = 0.5 * (static_cast<double>(grid.dims.ny) - 1.0) * grid.dy;
    const double hz = 0.5 * (static_cast<double>(grid.dims.nz) - 1.0) * grid.dz;
    grid.x0 = -hx;
    grid.y0 = -hy;
    grid.z0 = -hz;
}

void seed_chi_shell(Grid &grid, double amplitude, size_t shell) {
    size_t i0, i1, j0, j1, k0, k1;
    grid.domain_bounds(i0, i1, j0, j1, k0, k1);
    if (shell == 0)
        return;

    for (size_t i = i0; i < i1; ++i) {
        const size_t di = std::min(i - i0, i1 - 1 - i);
        for (size_t j = j0; j < j1; ++j) {
            const size_t dj = std::min(j - j0, j1 - 1 - j);
            for (size_t k = k0; k < k1; ++k) {
                const size_t dk = std::min(k - k0, k1 - 1 - k);
                const size_t dmin = std::min({di, dj, dk});
                if (dmin >= shell)
                    continue;
                const double weight = static_cast<double>(shell - dmin) / shell;
                const size_t idx = grid.chi.idx(i, j, k);
                grid.chi.ptr()[idx] = 1.0 + amplitude * weight;
            }
        }
    }
}

double max_shell_deviation(const Grid &grid, size_t shell) {
    size_t i0, i1, j0, j1, k0, k1;
    grid.domain_bounds(i0, i1, j0, j1, k0, k1);
    double max_dev = 0.0;
    if (shell == 0)
        return max_dev;

    for (size_t i = i0; i < i1; ++i) {
        const size_t di = std::min(i - i0, i1 - 1 - i);
        for (size_t j = j0; j < j1; ++j) {
            const size_t dj = std::min(j - j0, j1 - 1 - j);
            for (size_t k = k0; k < k1; ++k) {
                const size_t dk = std::min(k - k0, k1 - 1 - k);
                const size_t dmin = std::min({di, dj, dk});
                if (dmin >= shell)
                    continue;
                const size_t idx = grid.chi.idx(i, j, k);
                const double dev = std::abs(static_cast<double>(grid.chi.ptr()[idx]) - 1.0);
                max_dev = std::max(max_dev, dev);
            }
        }
    }
    return max_dev;
}

void perturb_metric_shell(Grid &grid, double amplitude, size_t shell) {
    size_t i0, i1, j0, j1, k0, k1;
    grid.domain_bounds(i0, i1, j0, j1, k0, k1);
    if (shell == 0)
        return;

    using tensorium_RG::XX;
    using tensorium_RG::YY;
    using tensorium_RG::ZZ;

    for (size_t i = i0; i < i1; ++i) {
        const size_t di = std::min(i - i0, i1 - 1 - i);
        for (size_t j = j0; j < j1; ++j) {
            const size_t dj = std::min(j - j0, j1 - 1 - j);
            for (size_t k = k0; k < k1; ++k) {
                const size_t dk = std::min(k - k0, k1 - 1 - k);
                const size_t dmin = std::min({di, dj, dk});
                if (dmin >= shell)
                    continue;
                const double weight = static_cast<double>(shell - dmin) / shell;
                const double pulse = amplitude * weight;
                const double gxx = 1.0 + pulse;
                const double gyy = 1.0 + pulse;
                const double gzz = 1.0 - 2.0 * pulse;
                const size_t idx = grid.gamma_tilde[XX].idx(i, j, k);
                grid.gamma_tilde[XX].ptr()[idx] = gxx;
                grid.gamma_tilde[YY].ptr()[idx] = gyy;
                grid.gamma_tilde[ZZ].ptr()[idx] = gzz;
                grid.gamma_tilde_inv[XX].ptr()[idx] = 1.0 / gxx;
                grid.gamma_tilde_inv[YY].ptr()[idx] = 1.0 / gyy;
                grid.gamma_tilde_inv[ZZ].ptr()[idx] = 1.0 / gzz;
            }
        }
    }
}

REGISTER_TEST("bssn.bc.outgoing_wave", "Sponge boundary damps outgoing chi pulses", []() {
    const size_t nx = 40;
    const size_t ny = 40;
    const size_t nz = 40;
    const size_t ng = 8;
    const double spacing = 0.5;
    const size_t shell = 6;
    const double amplitude = 0.1;

    Grid clamp(nx, ny, nz, ng, spacing, spacing, spacing);
    Grid sponge(nx, ny, nz, ng, spacing, spacing, spacing);

    auto configure = [&](Grid &grid) {
        center_grid(grid);
        tensorium_RG::init::minkowski(grid, 0.0);
        seed_chi_shell(grid, amplitude, shell);
    };

    configure(clamp);
    configure(sponge);

    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryClamp>(clamp);
    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundarySponge>(sponge);

    const double clamp_dev = max_shell_deviation(clamp, shell);
    const double sponge_dev = max_shell_deviation(sponge, shell);

    TENSORIUM_TEST_ASSERT(clamp_dev > 0.5 * amplitude);
    TENSORIUM_TEST_ASSERT(sponge_dev < 0.35 * clamp_dev);
});

REGISTER_TEST("bssn.bc.by_boundary_localization",
               "Determinant drift localizes near the computational boundary", []() {
                   const size_t nx = 48;
                   const size_t ny = 48;
                   const size_t nz = 48;
                   const size_t ng = 8;
                   const double spacing = 0.5;
                   const size_t shell = 4;
                   const double amplitude = 0.2;

                   Grid grid(nx, ny, nz, ng, spacing, spacing, spacing);
                   center_grid(grid);
                   tensorium_RG::init::minkowski(grid, 0.0);
                   perturb_metric_shell(grid, amplitude, shell);

                   const auto stats = tensorium_RG::bssn::compute_invariant_stats(grid, 1);
                   TENSORIUM_TEST_ASSERT(stats.max_det_deviation > 0.1 * amplitude);

                   size_t i0, i1, j0, j1, k0, k1;
                   grid.domain_bounds(i0, i1, j0, j1, k0, k1);
                   const auto dist = [&](size_t idx, size_t lo, size_t hi) {
                       return std::min(idx - lo, hi - 1 - idx);
                   };
                   const size_t di = dist(stats.det_location.i, i0, i1);
                   const size_t dj = dist(stats.det_location.j, j0, j1);
                   const size_t dk = dist(stats.det_location.k, k0, k1);
                   const size_t min_dist = std::min({di, dj, dk});
                   TENSORIUM_TEST_ASSERT(min_dist < shell);
               });

} // namespace
