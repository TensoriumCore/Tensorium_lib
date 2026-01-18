#include "../framework/TestRegistry.hpp"
#include "stability_common.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

using tensorium::tests::center_grid;
using tensorium::tests::Grid;
using tensorium::tests::interior_bounds;
using tensorium::tests::make_constraint_scratch;

REGISTER_TEST(
    "bssn.stability.ccz4_prime", "Long-term CCZ4-prime stability check with kappa_z decoupled",
    []() {
        constexpr size_t NX = 64;
        constexpr size_t NG = 4;
        constexpr size_t padding = 4;
        const double     dx = (20.0) / static_cast<double>(NX - 1);
        const double     dt = 0.25 * dx;
        const double     t_end = 100.0;

        Grid grid(NX, NX, NX, NG, dx, dx, dx);
        center_grid(grid);
        tensorium_RG::init::schwarzschild_isotropic(grid, 1.0);

        tensorium_RG::bssn::GaugeParameters<double> params;
        params.kappa1 = 0.2;
        params.kappa_z = 0.0;
        params.kappa2 = 0.0;
        params.eta = 2.0;
        params.beta_B_coeff = 0.75;

        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, padding);
        stepper.set_gauge_parameters(params);

        auto scratch = make_constraint_scratch(grid);
        std::filesystem::create_directories("Output/tests");
        std::ofstream csv("Output/tests/stability_test_ccz4_prime.csv");
        csv << "t,L2_H,L2_M\n";

        auto write_constraints = [&](double time) {
            tensorium_RG::bssn::compute_bssn_constraints(
                grid, grid.Ricci, scratch.H, scratch.M, scratch.C, 0.0,
                std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);

            size_t i0, i1, j0, j1, k0, k1;
            interior_bounds(grid, padding, i0, i1, j0, j1, k0, k1);

            double sumH = 0.0;
            double sumM = 0.0;
            size_t count = 0;
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j)
                    for (size_t k = k0; k < k1; ++k) {
                        const size_t id = grid.alpha.idx(i, j, k);
                        const double H = scratch.H.ptr()[id];
                        const double Mx = scratch.M[0].ptr()[id];
                        const double My = scratch.M[1].ptr()[id];
                        const double Mz = scratch.M[2].ptr()[id];
                        sumH += H * H;
                        sumM += Mx * Mx + My * My + Mz * Mz;
                        ++count;
                    }

            const double normH = std::sqrt(sumH / std::max<size_t>(count, size_t(1)));
            const double normM = std::sqrt(sumM / std::max<size_t>(count, size_t(1)));
            // std::endl force le "flush" (l'écriture physique sur le disque) immédiat
            csv << time << ',' << normH << ',' << normM << std::endl;
        };

        double t = 0.0;
        size_t step = 0;
        write_constraints(t);

        while (t < t_end) {
            stepper.step(grid, dt, step);
            t += dt;
            ++step;
            if (step % 10 == 0)
                write_constraints(t);
        }

        csv.close();

        tensorium_RG::bssn::compute_bssn_constraints(
            grid, grid.Ricci, scratch.H, scratch.M, scratch.C, 0.0,
            std::numeric_limits<double>::max(), 0.0, 0.0, 0.0, false);

        auto summary = tensorium::tests::summarize_grid(grid, padding);
        TENSORIUM_TEST_ASSERT(std::isfinite(summary.max_H));
        TENSORIUM_TEST_ASSERT(std::isfinite(summary.max_gamma));
    });

} // namespace
