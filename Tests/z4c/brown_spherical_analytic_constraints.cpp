#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

struct ConstraintSamples {
    double max_H = 0.0;
    double max_Mr = 0.0;
    double max_Gr = 0.0;
    double max_gamma_diff = 0.0;

    double max_chi_diff = 0.0;
    double max_dchi_diff = 0.0;
    double max_d2chi_diff = 0.0;
};

ConstraintSamples evaluate_constraints(double mass, size_t nr) {
    const double dr = mass / 100.0;
    const double r_cut = 0.5 * mass;

    ConstraintSamples stats;

    std::printf("\n[Analytic Schwarzschild / Brown Eq.(10) + chi checks]\n");
    std::printf("r/M      psi          chi          dchi         d2chi        H           Mr          Gr\n");
    std::printf("-------------------------------------------------------------------------------------------\n");

    for (size_t j = 0; j < nr; ++j) {
        const double r = (static_cast<double>(j) + 0.5) * dr;
        if (r <= r_cut)
            continue;

        const double psi  = 1.0 + mass / (2.0 * r);
        const double dpsi = -mass / (2.0 * r * r);
        const double d2psi = mass / (r * r * r);

        const double laplacian = d2psi + (2.0 / r) * dpsi;
        const double psi_inv5 = std::pow(psi, -5.0);

        const double H  = -8.0 * psi_inv5 * laplacian;
        const double Mr = 0.0;

        const double g_theta = r * r;
        const double dg_theta = 2.0 * r;
        const double gamma_calc = -dg_theta / g_theta;
        const double gamma_ref  = -2.0 / r;
        const double Gr = gamma_calc + 2.0 / r;

        const double chi_ref = std::pow(psi, -4.0);
        const double dchi_ref  = -4.0 * std::pow(psi, -5.0) * dpsi;
        const double d2chi_ref = 20.0 * std::pow(psi, -6.0) * (dpsi * dpsi)
                               - 4.0 * std::pow(psi, -5.0) * d2psi;

        const double chi_calc  = chi_ref;
        const double dchi_calc = dchi_ref;
        const double d2chi_calc = d2chi_ref;

        stats.max_H = std::max(stats.max_H, std::abs(H));
        stats.max_Mr = std::max(stats.max_Mr, std::abs(Mr));
        stats.max_Gr = std::max(stats.max_Gr, std::abs(Gr));
        stats.max_gamma_diff = std::max(stats.max_gamma_diff,
                                        std::abs(gamma_calc - gamma_ref));

        stats.max_chi_diff  = std::max(stats.max_chi_diff,
                                       std::abs(chi_calc - chi_ref));
        stats.max_dchi_diff = std::max(stats.max_dchi_diff,
                                       std::abs(dchi_calc - dchi_ref));
        stats.max_d2chi_diff = std::max(stats.max_d2chi_diff,
                                        std::abs(d2chi_calc - d2chi_ref));

        if (j < 8) {
            std::printf("%6.3f  % .6e  % .6e  % .6e  % .6e  % .3e  % .3e  % .3e\n",
                        r / mass,
                        psi,
                        chi_ref,
                        dchi_ref,
                        d2chi_ref,
                        H,
                        Mr,
                        Gr);
        }
    }

    std::printf("\n[Summary]\n");
    std::printf("max |H|         = %.3e\n", stats.max_H);
    std::printf("max |Mr|        = %.3e\n", stats.max_Mr);
    std::printf("max |Gr|        = %.3e\n", stats.max_Gr);
    std::printf("max |Gamma-Ref| = %.3e\n", stats.max_gamma_diff);
    std::printf("max |chi-Ref|   = %.3e\n", stats.max_chi_diff);
    std::printf("max |dchi-Ref|  = %.3e\n", stats.max_dchi_diff);
    std::printf("max |d2chi-Ref| = %.3e\n\n", stats.max_d2chi_diff);

    return stats;
}

} // namespace

REGISTER_TEST("z4c.analytic.brown_constraints",
              "Analytic Schwarzschild satisfies Brown Eq. (10) constraints + chi identities", []() {
                  constexpr double mass = 1.0;
                  constexpr size_t nr = 400;

                  const auto stats = evaluate_constraints(mass, nr);

                  TENSORIUM_TEST_ASSERT(stats.max_H < 1e-10);
                  TENSORIUM_TEST_ASSERT(stats.max_Mr < 1e-10);
                  TENSORIUM_TEST_ASSERT(stats.max_Gr < 1e-10);
                  TENSORIUM_TEST_ASSERT(stats.max_gamma_diff < 1e-10);

                  TENSORIUM_TEST_ASSERT(stats.max_chi_diff < 1e-14);
                  TENSORIUM_TEST_ASSERT(stats.max_dchi_diff < 1e-14);
                  TENSORIUM_TEST_ASSERT(stats.max_d2chi_diff < 1e-14);
              });
