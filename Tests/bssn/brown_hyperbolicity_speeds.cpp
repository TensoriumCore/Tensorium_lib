#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct ProfileSample {
    double r;
    double alpha;
    double gauge_speed;
};

std::vector<ProfileSample> build_profile(double mass, size_t nr, bool precollapsed) {
    const double dr = mass / 50.0;
    std::vector<ProfileSample> samples;
    samples.reserve(nr);
    for (size_t j = 0; j < nr; ++j) {
        const double r = (static_cast<double>(j) + 0.5) * dr;
        double alpha = 1.0;
        if (precollapsed) {
            const double psi = 1.0 + mass / (2.0 * r);
            alpha = std::pow(psi, -2.0);
        }
        const double vg = std::sqrt(2.0 / alpha);
        samples.push_back({r, alpha, vg});
    }
    return samples;
}

void print_samples(const char *label, const std::vector<ProfileSample> &samples, size_t count) {
    std::printf("%s (first %zu samples)\n", label, count);
    std::printf("r/M\talpha\tv_gauge\n");
    for (size_t i = 0; i < std::min(count, samples.size()); ++i)
        std::printf("%.3f\t%.6f\t%.6f\n", samples[i].r, samples[i].alpha, samples[i].gauge_speed);
}

} // namespace

REGISTER_TEST("bssn.analytic.brown_hyperbolicity",
              "Gauge speeds from Brown Eq. (13) bound the CFL condition", []() {
                  constexpr double mass = 1.0;
                  constexpr size_t nr = 100;
                  const auto alpha_flat = build_profile(mass, nr, false);
                  const auto alpha_collapse = build_profile(mass, nr, true);

                  print_samples("alpha=1 profile", alpha_flat, 5);
                  print_samples("precollapsed alpha profile", alpha_collapse, 5);

                  auto characterize_profile = [&](const std::vector<ProfileSample> &samples,
                                                  double expected_c) {
                      double alpha_max = 0.0;
                      double vg_max = 0.0;
                      for (const auto &s : samples) {
                          alpha_max = std::max(alpha_max, s.alpha);
                          vg_max = std::max(vg_max, s.gauge_speed);
                      }
                      const double required_c = vg_max / alpha_max;
                      if (expected_c > 0.0)
                          TENSORIUM_TEST_ASSERT(std::abs(required_c - expected_c) < 1e-12);
                      else
                          TENSORIUM_TEST_ASSERT(required_c > std::sqrt(2.0));
                  };

                  characterize_profile(alpha_flat, std::sqrt(2.0));
                  characterize_profile(alpha_collapse, -1.0);
              });
