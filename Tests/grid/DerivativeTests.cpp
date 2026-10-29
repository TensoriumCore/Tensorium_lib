#include "../framework/Assertions.hpp"
#include "../framework/TestRegistry.hpp"

#include "../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Derivatives/BSSNGridDerivatives.hpp"
#include "../../includes/Tensorium_Grid/Grid/GridLayout.hpp"

#include <array>
#include <cmath>

using namespace tensorium_RG;

namespace {
struct DerivativeErrors {
    double first = 0.0;
    double second = 0.0;
    double mixed = 0.0;
    double advection = 0.0;
};

DerivativeErrors derivative_errors(double h, size_t position) {
    constexpr size_t ng = 4, n = 12;
    constexpr double wave[3] = {1.1, 0.9, 1.3};
    const double     spacing[3] = {h, 0.8 * h, 1.2 * h};
    Strides<double>  st;
    st.nx_tot = st.ny_tot = st.nz_tot = n + 2 * ng;
    st.sz = 1;
    st.sy = st.nz_tot;
    st.sx = st.ny_tot * st.sy;
    auto field = make_field<double>(st);
    // The analytic function also supplies the ghosts, isolating stencil accuracy from BCs.
    for (size_t i = 0; i < st.nx_tot; ++i)
        for (size_t j = 0; j < st.ny_tot; ++j)
            for (size_t k = 0; k < st.nz_tot; ++k) {
                const double phase = 0.61 + wave[0] * (double(i) - double(position)) * spacing[0] +
                                     wave[1] * (double(j) - double(position)) * spacing[1] +
                                     wave[2] * (double(k) - double(position)) * spacing[2];
                field.ptr()[field.idx(i, j, k)] = std::sin(phase);
            }
    const double *p = field.ptr() + field.idx(position, position, position);
    const double  first[3] = {fd::Dx_ptr(p, st.sx, 1.0 / (60 * spacing[0])),
                              fd::Dy_ptr(p, st.sy, 1.0 / (60 * spacing[1])),
                              fd::Dz_ptr(p, 1.0 / (60 * spacing[2]))};
    const double  second[3] = {fd::Dxx_ptr(p, st.sx, 1.0 / (180 * spacing[0] * spacing[0])),
                               fd::Dyy_ptr(p, st.sy, 1.0 / (180 * spacing[1] * spacing[1])),
                               fd::Dzz_ptr(p, 1.0 / (180 * spacing[2] * spacing[2]))};
    const double  mixed[3] = {fd::Dxy4_ptr(p, st.sx, st.sy, 1.0 / (144 * spacing[0] * spacing[1])),
                              fd::Dxz4_ptr(p, st.sx, 1.0 / (144 * spacing[0] * spacing[2])),
                              fd::Dyz4_ptr(p, st.sy, 1.0 / (144 * spacing[1] * spacing[2]))};
    DerivativeErrors err;
    for (int axis = 0; axis < 3; ++axis) {
        err.first = std::max(err.first, std::abs(first[axis] - wave[axis] * std::cos(0.61)));
        err.second =
            std::max(err.second, std::abs(second[axis] + wave[axis] * wave[axis] * std::sin(0.61)));
        for (double beta : {-1.0, 1.0}) {
            const double upwind =
                axis == 0   ? fd::Dx_upwind_ptr(p, st.sx, 1.0 / (2 * spacing[0]), beta)
                : axis == 1 ? fd::Dy_upwind_ptr(p, st.sy, 1.0 / (2 * spacing[1]), beta)
                            : fd::Dz_upwind_ptr(p, 1.0 / (2 * spacing[2]), beta);
            err.advection = std::max(err.advection, std::abs(upwind - wave[axis] * std::cos(0.61)));
        }
    }
    const int a[3] = {0, 0, 1}, b[3] = {1, 2, 2};
    for (int pair = 0; pair < 3; ++pair)
        err.mixed = std::max(
            err.mixed, std::abs(mixed[pair] + wave[a[pair]] * wave[b[pair]] * std::sin(0.61)));
    return err;
}
} // namespace

REGISTER_TEST("grid.derivatives.convergence_3d",
              "Centered, mixed and upwind stencils converge in the bulk and at analytic ghosts",
              []() {
                  const int saved = fd::max_spatial_derivative_order();
                  struct Restore {
                      int order;
                      ~Restore() { fd::set_max_spatial_derivative_order(order); }
                  } restore{saved};
                  for (int order : {4, 6}) {
                      fd::set_max_spatial_derivative_order(order);
                      for (size_t position : {size_t(4), size_t(9), size_t(15)}) {
                          const auto   coarse = derivative_errors(0.4, position);
                          const auto   medium = derivative_errors(0.2, position);
                          const auto   fine = derivative_errors(0.1, position);
                          const double centered_ratio = order == 4 ? 12.0 : 45.0;
                          TENSORIUM_TEST_ASSERT(coarse.first > centered_ratio * medium.first);
                          TENSORIUM_TEST_ASSERT(medium.first > centered_ratio * fine.first);
                          TENSORIUM_TEST_ASSERT(coarse.second > centered_ratio * medium.second);
                          TENSORIUM_TEST_ASSERT(medium.second > centered_ratio * fine.second);
                          TENSORIUM_TEST_ASSERT(coarse.mixed > 12.0 * medium.mixed);
                          TENSORIUM_TEST_ASSERT(medium.mixed > 12.0 * fine.mixed);
                          TENSORIUM_TEST_ASSERT(coarse.advection > 45.0 * medium.advection);
                          TENSORIUM_TEST_ASSERT(medium.advection > 45.0 * fine.advection);
                      }
                  }
              });

REGISTER_TEST("grid.derivatives.ko_fourier_symbol", "KO6 damps Fourier modes on all axes", []() {
    constexpr size_t n = 16;
    constexpr double q[3] = {0.4, 0.7, 1.1};
    Strides<double>  st;
    st.nx_tot = st.ny_tot = st.nz_tot = n;
    st.sz = 1;
    st.sy = n;
    st.sx = n * n;
    auto field = make_field<double>(st);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                field.ptr()[field.idx(i, j, k)] = std::cos(q[0] * i + q[1] * j + q[2] * k);
    const double   *p = field.ptr() + field.idx(7, 7, 7);
    const ptrdiff_t stride[3] = {static_cast<ptrdiff_t>(st.sx), static_cast<ptrdiff_t>(st.sy), 1};
    for (int a = 0; a < 3; ++a) {
        const double exact = -std::pow(std::sin(q[a] / 2.0), 6) * *p;
        const double actual = fd::KO6_axis_ptr(p, stride[a]);
        TENSORIUM_TEST_ASSERT(std::isfinite(actual));
        tensorium::tests::expect_near(actual, exact, 1e-14, "KO6 Fourier symbol");
    }
});

REGISTER_TEST("grid.derivatives.fourth_order", "Central derivatives on a sinusoid", []() {
    const size_t     nx = 32;
    const size_t     ng = 4;
    const double     dx = 1.0 / nx;
    constexpr double two_pi = 6.28318530717958647692;

    Strides<double> st;
    st.nx_tot = nx + 2 * ng;
    st.ny_tot = nx + 2 * ng;
    st.nz_tot = nx + 2 * ng;
    st.sz = 1;
    st.sy = st.nz_tot;
    st.sx = st.sy * st.ny_tot;

    Field3D<double> field = make_field<double>(st);
    GridDims        dims{nx, nx, nx, ng};

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
