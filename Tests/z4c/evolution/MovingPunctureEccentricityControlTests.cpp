#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEccentricityControl.hpp"

#include <cmath>
#include <vector>

namespace {

using tensorium_RG::bssn::MovingPunctureEccentricityControlConfig;
using tensorium_RG::bssn::MovingPunctureEccentricityFitResult;
using tensorium_RG::bssn::MovingPunctureTrackPoint;

std::vector<MovingPunctureTrackPoint> make_synthetic_track(double r0, double trend, double omega,
                                                           double a, double b, double tmax,
                                                           double dt) {
    std::vector<MovingPunctureTrackPoint> track;
    for (double t = 0.0; t <= tmax + 1.0e-12; t += dt) {
        const double r = r0 + trend * t + a * std::cos(omega * t) + b * std::sin(omega * t);
        const double phi = 0.3 + omega * t;
        const double x = 0.5 * r * std::cos(phi);
        const double y = 0.5 * r * std::sin(phi);
        track.push_back(MovingPunctureTrackPoint{
            t, true, true, -x, -y, x, y,
        });
    }
    return track;
}

} // namespace

REGISTER_TEST("z4c.evolution.moving_puncture_eccentricity_fit_recovers_track_parameters",
              "The moving-puncture eccentricity fit recovers the secular drift and oscillation amplitude from a synthetic puncture track",
              []() {
                  const double r0 = 6.2;
                  const double trend = -2.0e-3;
                  const double omega = 0.11;
                  const double a = 0.08;
                  const double b = -0.03;
                  const auto   track = make_synthetic_track(r0, trend, omega, a, b, 18.0, 0.05);

                  MovingPunctureEccentricityControlConfig cfg;
                  cfg.enabled = true;
                  cfg.fit_tmin = 2.0;
                  cfg.fit_tmax = 16.0;
                  cfg.min_samples = 32;

                  const MovingPunctureEccentricityFitResult fit =
                      tensorium_RG::bssn::fit_moving_puncture_eccentricity(track, cfg);

                  TENSORIUM_TEST_ASSERT(fit.valid);
                  tensorium::tests::expect_near(fit.initial_separation, r0 + a, 1.0e-12,
                                                "synthetic track initial separation");
                  tensorium::tests::expect_near(fit.secular_rdot, trend, 1.0e-4,
                                                "synthetic track secular drift");
                  tensorium::tests::expect_near(fit.orbital_frequency, omega, 5.0e-4,
                                                "synthetic track orbital frequency");
                  tensorium::tests::expect_near(fit.cos_coefficient, a, 6.0e-3,
                                                "synthetic track cosine coefficient");
                  tensorium::tests::expect_near(fit.sin_coefficient, b, 6.0e-3,
                                                "synthetic track sine coefficient");
                  tensorium::tests::expect_near(fit.eccentricity, std::hypot(a, b) / r0, 1.5e-3,
                                                "synthetic track eccentricity");
              });

REGISTER_TEST("z4c.evolution.moving_puncture_eccentricity_update_uses_fit_signs",
              "The moving-puncture eccentricity update increases tangential and radial momentum when the fit says the orbit starts too wide and too outward",
              []() {
                  MovingPunctureEccentricityFitResult fit;
                  fit.valid = true;
                  fit.mean_separation = 6.0;
                  fit.cos_coefficient = 0.12;
                  fit.sin_coefficient = 0.04;
                  fit.orbital_frequency = 0.1;
                  fit.secular_rdot = 2.0e-3;
                  fit.oscillatory_rdot0 = fit.orbital_frequency * fit.sin_coefficient;

                  MovingPunctureEccentricityControlConfig cfg;
                  cfg.tangential_gain = 1.0;
                  cfg.radial_gain = 1.0;
                  cfg.max_fractional_tangential_update = 0.2;
                  cfg.max_absolute_radial_update = 1.0e-3;

                  const auto update = tensorium_RG::bssn::suggest_moving_puncture_momentum_update(
                      fit, 0.5, 0.5, 0.10, 5.0e-4, cfg);

                  TENSORIUM_TEST_ASSERT(update.valid);
                  TENSORIUM_TEST_ASSERT(update.delta_tangential_momentum > 0.0);
                  TENSORIUM_TEST_ASSERT(update.delta_radial_momentum > 0.0);
                  TENSORIUM_TEST_ASSERT(update.new_tangential_momentum > 0.10);
                  TENSORIUM_TEST_ASSERT(update.new_radial_momentum > 5.0e-4);
              });
