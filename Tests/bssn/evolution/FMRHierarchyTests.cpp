#include "../../framework/Assertions.hpp"
#include "../../framework/TestRegistry.hpp"

#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/FMR/BSSNFixedMeshRefinement.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"

#include <cstdlib>
#include <cmath>
#include <string>

namespace {

using Grid = tensorium_RG::BSSNGridSoA<double>;
using Hierarchy =
    tensorium_RG::bssn::fmr::FixedMeshRefinementHierarchy<double, tensorium_RG::bssn::BoundaryRadiative>;
using FineBoundary =
    tensorium_RG::bssn::fmr::ParentInterpolationBoundary<double, tensorium_RG::bssn::BoundaryRadiative>;

template <typename Fn> void for_each_physical(Grid &grid, Fn &&fn) {
    size_t I0, I1, J0, J1, K0, K1;
    grid.domain_bounds(I0, I1, J0, J1, K0, K1);
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                const size_t idx = grid.alpha.idx(i, j, k);
                double       x, y, z;
                grid.coords(i, j, k, x, y, z);
                fn(i, j, k, idx, x, y, z);
            }
}

inline void coords_any(const Grid &grid, size_t i, size_t j, size_t k, double &x, double &y,
                       double &z) {
    const ptrdiff_t di = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(grid.dims.ng);
    const ptrdiff_t dj = static_cast<ptrdiff_t>(j) - static_cast<ptrdiff_t>(grid.dims.ng);
    const ptrdiff_t dk = static_cast<ptrdiff_t>(k) - static_cast<ptrdiff_t>(grid.dims.ng);
    x = grid.x0 + double(di) * grid.dx;
    y = grid.y0 + double(dj) * grid.dy;
    z = grid.z0 + double(dk) * grid.dz;
}

inline double alpha_linear(double x, double y, double z) {
    return 1.0 + 0.05 * x - 0.03 * y + 0.02 * z;
}

inline double theta_linear(double x, double y, double z) {
    return 0.2 - 0.04 * x + 0.01 * y - 0.03 * z;
}

inline double beta_linear(double x, double y, double z) {
    return -0.1 + 0.02 * x + 0.01 * y - 0.015 * z;
}

inline Grid make_root_grid() {
    Grid grid(24, 24, 24, 4, 0.5, 0.5, 0.5);
    grid.x0 = -6.0;
    grid.y0 = -6.0;
    grid.z0 = -6.0;
    tensorium_RG::init::minkowski(grid, 0.0);
    return grid;
}

inline tensorium_RG::bssn::fmr::PatchBox fine_patch() {
    return tensorium_RG::bssn::fmr::PatchBox{6, 14, 6, 14, 6, 14};
}

struct ScopedEnvVar {
    std::string name;
    std::string value;
    std::string previous;
    bool        had_previous = false;

    ScopedEnvVar(const char *name_in, const char *value_in) : name(name_in), value(value_in) {
        if (const char *raw = std::getenv(name.c_str())) {
            previous = raw;
            had_previous = true;
        }
#if !defined(_WIN32)
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }

    ~ScopedEnvVar() {
#if !defined(_WIN32)
        if (had_previous)
            setenv(name.c_str(), previous.c_str(), 1);
        else
            unsetenv(name.c_str());
#endif
    }
};

} // namespace

REGISTER_TEST("bssn.fmr.prolongation_is_exact_for_linear_fields",
              "FMR prolongation preserves linear fields on the refined physical cells", []() {
    Grid root = make_root_grid();
    for_each_physical(root, [&](size_t, size_t, size_t, size_t idx, double x, double y, double z) {
        root.alpha.ptr()[idx] = alpha_linear(x, y, z);
        root.Theta.ptr()[idx] = theta_linear(x, y, z);
    });
    tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root);

    Hierarchy hierarchy(root, {{fine_patch(), 2, 0}}, 4);
    const Grid &fine = hierarchy.level_grid(1);

    for_each_physical(const_cast<Grid &>(fine),
                      [&](size_t, size_t, size_t, size_t idx, double x, double y, double z) {
                          tensorium::tests::expect_near(fine.alpha.ptr()[idx], alpha_linear(x, y, z),
                                                        1e-12, "FMR alpha prolongation");
                          tensorium::tests::expect_near(fine.Theta.ptr()[idx], theta_linear(x, y, z),
                                                        1e-12, "FMR Theta prolongation");
                      });
});

REGISTER_TEST("bssn.fmr.restriction_averages_linear_fields_back_to_coarse",
              "FMR restriction averages fine linear data back to the parent cells", []() {
    Grid root = make_root_grid();
    Hierarchy hierarchy(root, {{fine_patch(), 2, 0}}, 4);
    Grid     &fine = hierarchy.level_grid(1);

    for_each_physical(fine, [&](size_t, size_t, size_t, size_t idx, double x, double y, double z) {
        fine.alpha.ptr()[idx] = alpha_linear(x, y, z);
        fine.beta[0].ptr()[idx] = beta_linear(x, y, z);
    });

    hierarchy.restrict_level_to_parent(1);
    const Grid &coarse = hierarchy.root_grid();

    const auto box = fine_patch();
    for (size_t ip = box.i0; ip < box.i1; ++ip)
        for (size_t jp = box.j0; jp < box.j1; ++jp)
            for (size_t kp = box.k0; kp < box.k1; ++kp) {
                const size_t i = coarse.dims.ng + ip;
                const size_t j = coarse.dims.ng + jp;
                const size_t k = coarse.dims.ng + kp;
                const size_t idx = coarse.alpha.idx(i, j, k);
                double       x, y, z;
                coarse.coords(i, j, k, x, y, z);
                tensorium::tests::expect_near(coarse.alpha.ptr()[idx], alpha_linear(x, y, z), 1e-12,
                                              "FMR alpha restriction");
                tensorium::tests::expect_near(coarse.beta[0].ptr()[idx], beta_linear(x, y, z), 1e-12,
                                              "FMR beta restriction");
            }
});

REGISTER_TEST("bssn.fmr.fine_boundary_evolves_physical_shell",
              "Parent-interpolated fine levels evolve the first physical cells instead of freezing them",
              []() {
                  Grid root = make_root_grid();
                  for_each_physical(root, [&](size_t, size_t, size_t, size_t idx, double, double,
                                              double) { root.K.ptr()[idx] = 0.15; });
                  tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root);

                  Hierarchy hierarchy(root, {{fine_patch(), 2, 0}}, 4);
                  Grid     &fine = hierarchy.level_grid(1);

                  const size_t I0 = fine.dims.ng;
                  const size_t J0 = fine.dims.ng + fine.dims.ny / 2;
                  const size_t K0 = fine.dims.ng + fine.dims.nz / 2;
                  const size_t shell0 = fine.alpha.idx(I0, J0, K0);
                  const size_t shell4 = fine.alpha.idx(I0 + 4, J0, K0);
                  const double alpha0_before = fine.alpha.ptr()[shell0];
                  const double alpha4_before = fine.alpha.ptr()[shell4];

                  tensorium_RG::bssn::BSSNRKStepper<double, FineBoundary> stepper(fine, 4);
                  tensorium_RG::bssn::GaugeParameters<double>             gauge;
                  stepper.set_gauge_parameters(gauge);

                  const typename FineBoundary::Context ctx{&root, &root, 0.0};
                  const tensorium_RG::bssn::fmr::ScopedParentInterpolationContext<double> scope(&ctx);
                  stepper.step(fine, 1.0e-3, 0);

                  TENSORIUM_TEST_ASSERT(fine.alpha.ptr()[shell0] < alpha0_before);
                  TENSORIUM_TEST_ASSERT(fine.alpha.ptr()[shell4] < alpha4_before);
              });

REGISTER_TEST("bssn.fmr.parent_interpolation_boundary_blends_in_time",
              "Fine halo fills use a temporal blend of the parent states", []() {
                  Grid coarse_old = make_root_grid();
                  Grid coarse_new = make_root_grid();

                  for_each_physical(coarse_old, [&](size_t, size_t, size_t, size_t idx, double x,
                                                    double y, double z) {
                      coarse_old.alpha.ptr()[idx] = 1.0 + 0.1 * x - 0.05 * y + 0.02 * z;
                  });
                  for_each_physical(coarse_new, [&](size_t, size_t, size_t, size_t idx, double x,
                                                    double y, double z) {
                      coarse_new.alpha.ptr()[idx] = 1.5 - 0.03 * x + 0.07 * y - 0.04 * z;
                  });

                  Hierarchy hierarchy(coarse_old, {{fine_patch(), 2, 0}}, 4);
                  Grid     &fine = hierarchy.level_grid(1);

                  const size_t i = fine.dims.ng - 1;
                  const size_t j = fine.dims.ng + fine.dims.ny / 2;
                  const size_t k = fine.dims.ng + fine.dims.nz / 2;
                  const size_t idx = fine.alpha.idx(i, j, k);
                  fine.alpha.ptr()[idx] = -999.0;

                  const typename FineBoundary::Context ctx{&coarse_old, &coarse_new, 0.25};
                  const tensorium_RG::bssn::fmr::ScopedParentInterpolationContext<double> scope(&ctx);
                  tensorium_RG::bssn::apply_halos_grid<FineBoundary>(fine);

                  double x, y, z;
                  coords_any(fine, i, j, k, x, y, z);
                  const double expected_old = 1.0 + 0.1 * x - 0.05 * y + 0.02 * z;
                  const double expected_new = 1.5 - 0.03 * x + 0.07 * y - 0.04 * z;
                  const double expected = 0.75 * expected_old + 0.25 * expected_new;
                  tensorium::tests::expect_near(fine.alpha.ptr()[idx], expected, 1e-12,
                                                "FMR halo time interpolation");
              });

REGISTER_TEST("bssn.fmr.parent_interpolation_boundary_tracks_stage_fraction",
              "Fine halo fills can track the RK substage time between the parent start and end states",
              []() {
                  Grid coarse_old = make_root_grid();
                  Grid coarse_new = make_root_grid();

                  for_each_physical(coarse_old, [&](size_t, size_t, size_t, size_t idx, double x,
                                                    double y, double z) {
                      coarse_old.alpha.ptr()[idx] = 1.0 + 0.1 * x - 0.05 * y + 0.02 * z;
                  });
                  for_each_physical(coarse_new, [&](size_t, size_t, size_t, size_t idx, double x,
                                                    double y, double z) {
                      coarse_new.alpha.ptr()[idx] = 1.5 - 0.03 * x + 0.07 * y - 0.04 * z;
                  });

                  Hierarchy hierarchy(coarse_old, {{fine_patch(), 2, 0}}, 4);
                  Grid     &fine = hierarchy.level_grid(1);

                  const size_t i = fine.dims.ng - 1;
                  const size_t j = fine.dims.ng + fine.dims.ny / 2;
                  const size_t k = fine.dims.ng + fine.dims.nz / 2;
                  const size_t idx = fine.alpha.idx(i, j, k);
                  fine.alpha.ptr()[idx] = -999.0;

                  const typename FineBoundary::Context ctx{&coarse_old, &coarse_new, 0.25, 0.75};
                  const tensorium_RG::bssn::fmr::ScopedParentInterpolationContext<double> scope(&ctx);
                  FineBoundary::set_stage_fraction(0.5);
                  tensorium_RG::bssn::apply_halos_grid<FineBoundary>(fine);
                  FineBoundary::set_stage_fraction(1.0);

                  double x, y, z;
                  coords_any(fine, i, j, k, x, y, z);
                  const double expected_old = 1.0 + 0.1 * x - 0.05 * y + 0.02 * z;
                  const double expected_new = 1.5 - 0.03 * x + 0.07 * y - 0.04 * z;
                  const double expected = 0.5 * expected_old + 0.5 * expected_new;
                  tensorium::tests::expect_near(fine.alpha.ptr()[idx], expected, 1e-12,
                                                "FMR halo stage-fraction interpolation");
              });

REGISTER_TEST("bssn.fmr.initial_shell_blending_matches_parent_at_interface",
              "Fine-level initialization can smoothly match the parent on a physical shell", []() {
                  Grid root = make_root_grid();
                  for_each_physical(root, [&](size_t, size_t, size_t, size_t idx, double x, double y,
                                              double z) { root.alpha.ptr()[idx] = alpha_linear(x, y, z); });
                  tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root);

                  Hierarchy hierarchy(root, {{fine_patch(), 2, 0}}, 4);
                  Grid     &fine = hierarchy.level_grid(1);

                  for_each_physical(fine, [&](size_t, size_t, size_t, size_t idx, double, double, double) {
                      fine.alpha.ptr()[idx] = 10.0;
                  });

                  hierarchy.blend_level_shell_from_parent(1, 4);

                  const size_t boundary_i = fine.dims.ng;
                  const size_t center_j = fine.dims.ng + fine.dims.ny / 2;
                  const size_t center_k = fine.dims.ng + fine.dims.nz / 2;
                  double       x, y, z;
                  fine.coords(boundary_i, center_j, center_k, x, y, z);
                  tensorium::tests::expect_near(
                      fine.alpha.ptr()[fine.alpha.idx(boundary_i, center_j, center_k)],
                      alpha_linear(x, y, z), 1e-12, "FMR shell boundary matches parent");

                  const size_t interior_i = fine.dims.ng + 2;
                  double       xi, yi, zi;
                  fine.coords(interior_i, center_j, center_k, xi, yi, zi);
                  const double interior_alpha =
                      fine.alpha.ptr()[fine.alpha.idx(interior_i, center_j, center_k)];
                  TENSORIUM_TEST_ASSERT(interior_alpha > alpha_linear(xi, yi, zi));
                  TENSORIUM_TEST_ASSERT(interior_alpha < 10.0);

                  const size_t deep_i = fine.dims.ng + 5;
                  tensorium::tests::expect_near(
                      fine.alpha.ptr()[fine.alpha.idx(deep_i, center_j, center_k)], 10.0, 1e-12,
                      "FMR shell leaves deep interior unchanged");
              });

REGISTER_TEST("bssn.fmr.centered_core_blending_keeps_parent_outer_collar",
              "Fine-level initialization can inject a centered TP core while preserving the parent prolongation near the interface",
              []() {
                  Grid root(24, 24, 24, 4, 0.5, 0.5, 0.5);
                  tensorium_RG::bssn::center_cell_centered_origin(root);
                  tensorium_RG::init::minkowski(root, 0.0);
                  for_each_physical(root, [&](size_t, size_t, size_t, size_t idx, double, double,
                                              double) { root.alpha.ptr()[idx] = 1.0; });
                  tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root);

                  const tensorium_RG::bssn::fmr::PatchBox centered_patch{8, 16, 8, 16, 8, 16};
                  Hierarchy hierarchy(root, {{centered_patch, 2, 0}}, 4);
                  hierarchy.prolongate_level_from_parent(1);
                  Grid &fine = hierarchy.level_grid(1);

                  Grid reference(fine.dims.nx, fine.dims.ny, fine.dims.nz, fine.dims.ng, fine.dx,
                                 fine.dy, fine.dz);
                  reference.x0 = fine.x0;
                  reference.y0 = fine.y0;
                  reference.z0 = fine.z0;
                  tensorium_RG::init::minkowski(reference, 0.0);
                  for_each_physical(reference, [&](size_t, size_t, size_t, size_t idx, double,
                                                   double, double) { reference.alpha.ptr()[idx] = 10.0; });

                  hierarchy.blend_level_centered_core_from_reference(1, reference, 0.5, 0.75);

                  const size_t center_j = fine.dims.ng + fine.dims.ny / 2 - 1;
                  const size_t center_k = fine.dims.ng + fine.dims.nz / 2 - 1;
                  const size_t center_i = fine.dims.ng + fine.dims.nx / 2 - 1;
                  tensorium::tests::expect_near(
                      fine.alpha.ptr()[fine.alpha.idx(center_i, center_j, center_k)], 10.0, 1e-12,
                      "FMR centered core keeps the TP reference at the center");

                  const size_t transition_i = fine.dims.ng + 11;
                  const double transition_alpha =
                      fine.alpha.ptr()[fine.alpha.idx(transition_i, center_j, center_k)];
                  TENSORIUM_TEST_ASSERT(transition_alpha > 1.0);
                  TENSORIUM_TEST_ASSERT(transition_alpha < 10.0);

                  const size_t outer_i = fine.dims.ng + fine.dims.nx - 1;
                  tensorium::tests::expect_near(
                      fine.alpha.ptr()[fine.alpha.idx(outer_i, center_j, center_k)], 1.0, 1e-12,
                      "FMR centered core leaves the outer collar on the parent prolongation");
              });

REGISTER_TEST("bssn.fmr.hierarchy_step_keeps_levels_finite",
              "The fixed hierarchy subcycles and restricts without producing non-finite states", []() {
                  Grid root = make_root_grid();
                  for_each_physical(root, [&](size_t, size_t, size_t, size_t idx, double, double,
                                              double) { root.K.ptr()[idx] = 0.1; });
                  tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(root);

                  Hierarchy hierarchy(root, {{fine_patch(), 2, 0}}, 4);
                  const auto dt = 5.0e-4;

                  const auto box = fine_patch();
                  const size_t ic = hierarchy.root_grid().dims.ng + (box.i0 + box.i1) / 2;
                  const size_t jc = hierarchy.root_grid().dims.ng + (box.j0 + box.j1) / 2;
                  const size_t kc = hierarchy.root_grid().dims.ng + (box.k0 + box.k1) / 2;
                  const double alpha_before =
                      hierarchy.root_grid().alpha.ptr()[hierarchy.root_grid().alpha.idx(ic, jc, kc)];

                  hierarchy.step(dt);

                  const Grid &coarse = hierarchy.root_grid();
                  const Grid &fine = hierarchy.level_grid(1);
                  const double alpha_after =
                      coarse.alpha.ptr()[coarse.alpha.idx(ic, jc, kc)];

                  TENSORIUM_TEST_ASSERT(std::isfinite(alpha_after));
                  TENSORIUM_TEST_ASSERT(alpha_after < alpha_before);

                  for_each_physical(const_cast<Grid &>(fine),
                                    [&](size_t, size_t, size_t, size_t idx, double, double,
                                        double) { TENSORIUM_TEST_ASSERT(std::isfinite(fine.alpha.ptr()[idx])); });
              });

REGISTER_TEST("bssn.fmr.moving_puncture_auto_boxes_cover_the_binary",
              "Moving-puncture FMR boxes are centered and remain large enough to contain both punctures",
              []() {
                  tensorium_RG::bssn::MovingPunctureEnvConfig cfg;
                  cfg.nx = 96;
                  cfg.ny = 96;
                  cfg.nz = 96;
                  cfg.spacing = 62.4 / 96.0;
                  cfg.separation = 6.10679;
                  cfg.fmr.enabled = true;
                  cfg.fmr.levels = 2;
                  cfg.fmr.refinement_ratio = 2;

                  Grid root(cfg.nx, cfg.ny, cfg.nz, 4, cfg.spacing, cfg.spacing, cfg.spacing);
                  tensorium_RG::bssn::center_cell_centered_origin(root);

                  const auto levels = tensorium_RG::bssn::build_moving_puncture_fmr_levels(root, cfg);
                  TENSORIUM_TEST_ASSERT(levels.size() == 2);

                  Hierarchy hierarchy(root, levels, 4);
                  const Grid &fine = hierarchy.level_grid(hierarchy.num_levels() - 1);
                  const double x_min = fine.x0 - 0.5 * fine.dx;
                  const double x_max = fine.x0 + (double(fine.dims.nx) - 0.5) * fine.dx;
                  const double y_min = fine.y0 - 0.5 * fine.dy;
                  const double y_max = fine.y0 + (double(fine.dims.ny) - 0.5) * fine.dy;

                  TENSORIUM_TEST_ASSERT(x_min < -cfg.separation);
                  TENSORIUM_TEST_ASSERT(x_max > cfg.separation);
                  TENSORIUM_TEST_ASSERT(y_min < -cfg.separation);
                  TENSORIUM_TEST_ASSERT(y_max > cfg.separation);
                  tensorium::tests::expect_near(fine.dx, root.dx / 4.0, 1e-15,
                                                "Moving-puncture finest spacing");
              });

REGISTER_TEST("bssn.fmr.moving_puncture_outer_box_half_width_controls_nested_levels",
              "An explicit outer refined half-width builds a geometric nested hierarchy toward the center",
              []() {
                  tensorium_RG::bssn::MovingPunctureEnvConfig cfg;
                  cfg.nx = 64;
                  cfg.ny = 64;
                  cfg.nz = 64;
                  cfg.spacing = 0.5;
                  cfg.separation = 3.053395;
                  cfg.fmr.enabled = true;
                  cfg.fmr.levels = 3;
                  cfg.fmr.refinement_ratio = 2;
                  cfg.fmr.outer_box_half_width = 8.0;

                  Grid root(cfg.nx, cfg.ny, cfg.nz, 4, cfg.spacing, cfg.spacing, cfg.spacing);
                  tensorium_RG::bssn::center_cell_centered_origin(root);

                  const auto levels = tensorium_RG::bssn::build_moving_puncture_fmr_levels(root, cfg);
                  TENSORIUM_TEST_ASSERT(levels.size() == 3);

                  Hierarchy hierarchy(root, levels, 4);
                  const Grid &level1 = hierarchy.level_grid(1);
                  const Grid &level2 = hierarchy.level_grid(2);
                  const Grid &level3 = hierarchy.level_grid(3);

                  const double h1 = 0.5 * level1.dims.nx * level1.dx;
                  const double h2 = 0.5 * level2.dims.nx * level2.dx;
                  const double h3 = 0.5 * level3.dims.nx * level3.dx;

                  tensorium::tests::expect_near(h1, 8.0, root.dx, "Outer refined half-width");
                  tensorium::tests::expect_near(h2, 4.0, level1.dx, "Intermediate refined half-width");
                  tensorium::tests::expect_near(h3, 2.0, level2.dx, "Finest refined half-width");
                  tensorium::tests::expect_near(level3.dx, root.dx / 8.0, 1e-15,
                                                "Three-level finest spacing");
              });

REGISTER_TEST("bssn.fmr.moving_puncture_centered_levels_follow_requested_center",
              "An explicit FMR center shifts the refined hierarchy away from the origin",
              []() {
                  tensorium_RG::bssn::MovingPunctureEnvConfig cfg;
                  cfg.nx = 96;
                  cfg.ny = 96;
                  cfg.nz = 96;
                  cfg.spacing = 0.25;
                  cfg.separation = 3.053395;
                  cfg.fmr.enabled = true;
                  cfg.fmr.levels = 2;
                  cfg.fmr.refinement_ratio = 2;
                  cfg.fmr.outer_box_half_width = 6.0;

                  Grid root(cfg.nx, cfg.ny, cfg.nz, 4, cfg.spacing, cfg.spacing, cfg.spacing);
                  tensorium_RG::bssn::center_cell_centered_origin(root);

                  const std::array<double, 3> center{2.5, -1.0, 0.0};
                  const auto levels =
                      tensorium_RG::bssn::build_moving_puncture_fmr_levels(root, cfg, center);
                  TENSORIUM_TEST_ASSERT(levels.size() == 2);

                  Hierarchy hierarchy(root, levels, 4);
                  const Grid &level1 = hierarchy.level_grid(1);
                  const Grid &level2 = hierarchy.level_grid(2);

                  const auto center1 =
                      std::array<double, 3>{level1.x0 + 0.5 * double(level1.dims.nx - 1) * level1.dx,
                                            level1.y0 + 0.5 * double(level1.dims.ny - 1) * level1.dy,
                                            level1.z0 + 0.5 * double(level1.dims.nz - 1) * level1.dz};
                  const auto center2 =
                      std::array<double, 3>{level2.x0 + 0.5 * double(level2.dims.nx - 1) * level2.dx,
                                            level2.y0 + 0.5 * double(level2.dims.ny - 1) * level2.dy,
                                            level2.z0 + 0.5 * double(level2.dims.nz - 1) * level2.dz};

                  tensorium::tests::expect_near(center1[0], center[0], level1.dx,
                                                "Outer refined level follows requested x center");
                  tensorium::tests::expect_near(center1[1], center[1], level1.dy,
                                                "Outer refined level follows requested y center");
                  tensorium::tests::expect_near(center2[0], center[0], level2.dx,
                                                "Finest refined level follows requested x center");
                  tensorium::tests::expect_near(center2[1], center[1], level2.dy,
                                                "Finest refined level follows requested y center");
              });

#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
REGISTER_TEST("bssn.fmr.twopunctures_repeated_init_reuses_cached_solve",
              "Repeated TwoPunctures initializations with identical binary parameters stay finite",
              []() {
                  ScopedEnvVar tp_a("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_A", "8");
                  ScopedEnvVar tp_b("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_B", "8");
                  ScopedEnvVar tp_phi("TENSORIUM_MOVING_PUNCTURE_TP_NPOINTS_PHI", "4");
                  ScopedEnvVar tp_newton_tol("TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_TOL", "1e-8");
                  ScopedEnvVar tp_newton_maxit("TENSORIUM_MOVING_PUNCTURE_TP_NEWTON_MAXIT", "2");
                  ScopedEnvVar tp_eps("TENSORIUM_MOVING_PUNCTURE_TP_EPSILON", "1e-6");
                  ScopedEnvVar tp_verbose("TENSORIUM_MOVING_PUNCTURE_TP_VERBOSE", "0");

                  Grid coarse(16, 16, 16, 4, 0.5, 0.5, 0.5);
                  Grid fine(12, 12, 12, 4, 0.25, 0.25, 0.25);
                  tensorium_RG::bssn::center_cell_centered_origin(coarse);
                  tensorium_RG::bssn::center_cell_centered_origin(fine);

                  const double P1[3] = {0.0, 0.05, 0.0};
                  const double P2[3] = {0.0, -0.05, 0.0};
                  const double S1[3] = {0.0, 0.0, 0.0};
                  const double S2[3] = {0.0, 0.0, 0.0};

                  tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init(
                      coarse, 0.5, -1.0, 0.0, 0.0, P1, S1, 0.5, 1.0, 0.0, 0.0, P2, S2);
                  tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init(
                      fine, 0.5, -1.0, 0.0, 0.0, P1, S1, 0.5, 1.0, 0.0, 0.0, P2, S2);

                  const size_t coarse_center =
                      coarse.alpha.idx(coarse.dims.ng + coarse.dims.nx / 2,
                                       coarse.dims.ng + coarse.dims.ny / 2,
                                       coarse.dims.ng + coarse.dims.nz / 2);
                  const size_t fine_center =
                      fine.alpha.idx(fine.dims.ng + fine.dims.nx / 2, fine.dims.ng + fine.dims.ny / 2,
                                     fine.dims.ng + fine.dims.nz / 2);

                  TENSORIUM_TEST_ASSERT(std::isfinite(coarse.alpha.ptr()[coarse_center]));
                  TENSORIUM_TEST_ASSERT(std::isfinite(fine.alpha.ptr()[fine_center]));
                  TENSORIUM_TEST_ASSERT(coarse.alpha.ptr()[coarse_center] > 0.0);
                  TENSORIUM_TEST_ASSERT(fine.alpha.ptr()[fine_center] > 0.0);
              });
#endif
