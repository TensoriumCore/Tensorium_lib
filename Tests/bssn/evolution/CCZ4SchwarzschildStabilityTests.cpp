#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"
#include "../stability_common.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;

struct GaugeSnapshot {
    double alpha_min = std::numeric_limits<double>::infinity();
    double chi_min = std::numeric_limits<double>::infinity();
    double beta_max = 0.0;
};

GaugeSnapshot gather_gauge_snapshot(const Grid &grid, size_t padding) {
    GaugeSnapshot snapshot;
    size_t       i0, i1, j0, j1, k0, k1;
    tensorium::tests::interior_bounds(grid, padding, i0, i1, j0, j1, k0, k1);
    if (i0 >= i1 || j0 >= j1 || k0 >= k1)
        return snapshot;

    for (size_t i = i0; i < i1; ++i)
        for (size_t j = j0; j < j1; ++j)
            for (size_t k = k0; k < k1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                snapshot.alpha_min = std::min(snapshot.alpha_min, double(grid.alpha.ptr()[idx]));
                snapshot.chi_min = std::min(snapshot.chi_min, double(grid.chi.ptr()[idx]));

                const double bx = double(grid.beta[0].ptr()[idx]);
                const double by = double(grid.beta[1].ptr()[idx]);
                const double bz = double(grid.beta[2].ptr()[idx]);
                const double beta_mag = std::sqrt(bx * bx + by * by + bz * bz);
                snapshot.beta_max = std::max(snapshot.beta_max, beta_mag);
            }

    return snapshot;
}

tensorium_RG::bssn::GaugeParameters<double> make_schwarzschild_gauge() {
    tensorium_RG::bssn::GaugeParameters<double> params;
    params.beta_B_coeff = 0.75;
    params.eta = 2.0;
    params.kappa1 = 0.1;
    params.kappa2 = 0.0;
    params.kappa_z = 1.0;
    params.use_theta_in_lapse = true;
    params.min_lapse_for_K = 1e-3;
    params.max_K_squared = 1e4;
    params.ko_sigma = 0.35;
    return params;
}

void log_progress(size_t step, const GaugeSnapshot &snapshot,
                  const tensorium_RG::bssn::ConstraintMonitorStats &stats) {
    std::printf("[ccz4.schw] step=%zu alpha_min=%.3e chi_min=%.3e beta_max=%.3e "
                "||Theta||2=%.3e ||Z||2=%.3e ||H||2=%.3e ||M||2=%.3e det_drift=%.3e trA=%.3e\n",
                step, snapshot.alpha_min, snapshot.chi_min, snapshot.beta_max, stats.l2_theta,
                stats.l2_Z, stats.l2_H, stats.l2_M, stats.max_det_drift, stats.max_trace_A);
}

} // namespace

REGISTER_TEST(
    "bssn.evolution.ccz4_schwarzschild_stability",
    "Ensure CCZ4/Z4c evolution keeps Schwarzschild data stationary", []() {
        constexpr size_t nx = 64;
        constexpr size_t ng = 4;
        constexpr size_t steps = 25;
        constexpr size_t padding = 4;
        constexpr double dx = 0.2;
        constexpr double mass = 1.0;
        constexpr size_t log_stride = 5;

        tensorium_RG::fd::set_fd_dx(dx);

        Grid grid(nx, nx, nx, ng, dx, dx, dx);
        tensorium::tests::center_grid(grid);
        tensorium_RG::init::schwarzschild_isotropic(grid, mass);

        auto constraint_fields = tensorium::tests::make_constraint_scratch(grid);
        const double domain_extent = (grid.dims.nx - 1) * grid.dx;
        const double monitor_r_min = 2.0 * grid.dx;
        const double monitor_r_max = 0.45 * domain_extent;

        auto gather_constraints = [&]() {
            tensorium_RG::bssn::compute_bssn_constraints(
                grid, grid.Ricci, constraint_fields.H, constraint_fields.M, constraint_fields.C,
                monitor_r_min, monitor_r_max, 0.0, 0.0, 0.0, false);
            auto stats = tensorium_RG::bssn::compute_constraint_monitor(grid, constraint_fields.H,
                                                                        padding);
            tensorium_RG::bssn::populate_constraint_norms(grid, constraint_fields.M, stats, padding);
            return stats;
        };

        auto stats = gather_constraints();
        auto gauge_snapshot = gather_gauge_snapshot(grid, padding);

        double theta_l2_max = stats.l2_theta;
        double z_l2_max = stats.l2_Z;
        double H_l2_max = stats.l2_H;
        double M_l2_max = stats.l2_M;
        double det_drift_max = stats.max_det_drift;
        double traceA_max = stats.max_trace_A;
        double alpha_min_record = gauge_snapshot.alpha_min;
        double chi_min_record = gauge_snapshot.chi_min;
        double beta_max_record = gauge_snapshot.beta_max;

        tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative> stepper(
            grid, padding);
        stepper.set_gauge_parameters(make_schwarzschild_gauge());

        tensorium_RG::bssn::CFLControl<double> control;
        control.cfl = 0.25;
        control.gauge_speed = 1.0;

        log_progress(0, gauge_snapshot, stats);

        for (size_t step = 1; step <= steps; ++step) {
            const double dt = tensorium_RG::bssn::compute_dt_cfl(grid, control, padding);
            stepper.step(grid, dt, step);

            stats = gather_constraints();
            gauge_snapshot = gather_gauge_snapshot(grid, padding);

            theta_l2_max = std::max(theta_l2_max, stats.l2_theta);
            z_l2_max = std::max(z_l2_max, stats.l2_Z);
            H_l2_max = std::max(H_l2_max, stats.l2_H);
            M_l2_max = std::max(M_l2_max, stats.l2_M);
            det_drift_max = std::max(det_drift_max, stats.max_det_drift);
            traceA_max = std::max(traceA_max, stats.max_trace_A);
            alpha_min_record = std::min(alpha_min_record, gauge_snapshot.alpha_min);
            chi_min_record = std::min(chi_min_record, gauge_snapshot.chi_min);
            beta_max_record = std::max(beta_max_record, gauge_snapshot.beta_max);

            if (step % log_stride == 0 || step == steps)
                log_progress(step, gauge_snapshot, stats);
        }

        tensorium::tests::expect_le(det_drift_max, 1e-8, "max |det(gamma_tilde)-1|");
        tensorium::tests::expect_le(traceA_max, 5e-11, "max |tr(A_tilde)|");
        tensorium::tests::expect_le(theta_l2_max, 3e-4, "||Theta||_2 stability");
        tensorium::tests::expect_le(z_l2_max, 4e-4, "||Z||_2 stability");
        tensorium::tests::expect_le(H_l2_max, 2e-3, "||H||_2 stability");
        tensorium::tests::expect_le(M_l2_max, 2e-3, "||M||_2 stability");

        if (!(alpha_min_record > 1e-3))
            tensorium::tests::raise_failure("alpha_min collapsed below safe floor");
        tensorium::tests::expect_le(beta_max_record, 6e-3, "|beta| remained small");
        if (!(chi_min_record > 1e-4))
            tensorium::tests::raise_failure("chi_min dropped below numerical floor");
    });
