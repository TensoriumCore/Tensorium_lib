#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"
#include "../../../includes/Tensorium/Utils/IO/export.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

namespace {

using Grid = tensorium_RG::Z4cGridSoA<double>;
using MovingPunctureHierarchy =
    tensorium_RG::z4c::fmr::FixedMeshRefinementHierarchy<double, tensorium_RG::z4c::BoundaryRadiative>;
using BinaryMovingPunctureHierarchy =
    tensorium_RG::z4c::fmr::BinaryPunctureFixedMeshRefinementHierarchy<double,
                                                                       tensorium_RG::z4c::BoundaryRadiative>;
using FineBoundary = MovingPunctureHierarchy::FineBoundary;

template <typename HierarchyType>
void export_slice_boxes_csv(const Grid &grid, const HierarchyType *hierarchy, size_t step,
                            const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/slice_boxes_" << std::setw(4) << std::setfill('0') << step << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "level,x_min,x_max,y_min,y_max,z_min,z_max,center_x,center_y,center_z,dx,dy,dz,nx,ny,nz\n";

    auto write_level = [&](size_t level, const Grid &level_grid) {
        const double x_min = level_grid.x0 - 0.5 * level_grid.dx;
        const double y_min = level_grid.y0 - 0.5 * level_grid.dy;
        const double z_min = level_grid.z0 - 0.5 * level_grid.dz;
        const double x_max = level_grid.x0 + (double(level_grid.dims.nx) - 0.5) * level_grid.dx;
        const double y_max = level_grid.y0 + (double(level_grid.dims.ny) - 0.5) * level_grid.dy;
        const double z_max = level_grid.z0 + (double(level_grid.dims.nz) - 0.5) * level_grid.dz;
        const std::array<double, 3> center{
            level_grid.x0 + 0.5 * double(level_grid.dims.nx - 1) * level_grid.dx,
            level_grid.y0 + 0.5 * double(level_grid.dims.ny - 1) * level_grid.dy,
            level_grid.z0 + 0.5 * double(level_grid.dims.nz - 1) * level_grid.dz};
        file << level << "," << x_min << "," << x_max << "," << y_min << "," << y_max << ","
             << z_min << "," << z_max << "," << center[0] << "," << center[1] << "," << center[2]
             << "," << level_grid.dx << "," << level_grid.dy << "," << level_grid.dz << ","
             << level_grid.dims.nx << "," << level_grid.dims.ny << "," << level_grid.dims.nz
             << "\n";
    };

    if (hierarchy == nullptr) {
        write_level(0, grid);
        return;
    }

    for (size_t level = 0; level < hierarchy->num_levels(); ++level)
        write_level(level, hierarchy->level_grid(level));
}

template <typename HierarchyType>
void export_slice_csv(const Grid &grid, const HierarchyType *hierarchy, size_t step,
                      const std::string &output_dir) {
    std::stringstream ss;
    ss << output_dir << "/slice_" << std::setw(4) << std::setfill('0') << step << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,i,j,cell_xmin,cell_xmax,cell_ymin,cell_ymax,dx,dy,alpha,chi,mask\n";
    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;
            const double cell_xmin = x - 0.5 * grid.dx;
            const double cell_xmax = x + 0.5 * grid.dx;
            const double cell_ymin = y - 0.5 * grid.dy;
            const double cell_ymax = y + 0.5 * grid.dy;
            const size_t idx = grid.alpha.idx(i, j, k);
            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            const double mask = (alpha < 0.1) ? 1.0 : 0.0;
            file << x << "," << y << "," << (i - ng) << "," << (j - ng) << "," << cell_xmin << ","
                 << cell_xmax << "," << cell_ymin << "," << cell_ymax << "," << grid.dx << ","
                 << grid.dy << "," << alpha << "," << chi << "," << mask << "\n";
        }
    }

    export_slice_boxes_csv(grid, hierarchy, step, output_dir);
}

void export_constraint_slice_csv(const Grid &grid, const tensorium_RG::Field3D<double> &H,
                                 const tensorium_RG::Field3D<double> M[3],
                                 const tensorium_RG::Field3D<double> C[3], size_t step,
                                 const std::string &output_dir) {
    constexpr size_t kConstraintGuard = 4;

    std::stringstream ss;
    ss << output_dir << "/constraint_slice_" << std::setw(4) << std::setfill('0') << step
       << ".csv";

    std::ofstream file(ss.str());
    if (!file.is_open())
        return;

    file << "x,y,alpha,chi,H,absH,Mx,My,Mz,Mnorm,Cx,Cy,Cz,Cnorm,Theta,Zx,Zy,Zz,Znorm\n";

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;

            const size_t idx = grid.alpha.idx(i, j, k);
            const size_t cidx = H.idx(i, j, k);
            const bool constraint_valid =
                (i >= ng + kConstraintGuard && i + kConstraintGuard < nx + ng &&
                 j >= ng + kConstraintGuard && j + kConstraintGuard < ny + ng);

            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double h = constraint_valid ? H.ptr()[cidx] : nan;
            const double mx = constraint_valid ? M[0].ptr()[cidx] : nan;
            const double my = constraint_valid ? M[1].ptr()[cidx] : nan;
            const double mz = constraint_valid ? M[2].ptr()[cidx] : nan;
            const double mnorm = std::sqrt(mx * mx + my * my + mz * mz);
            const double cx = constraint_valid ? C[0].ptr()[cidx] : nan;
            const double cy = constraint_valid ? C[1].ptr()[cidx] : nan;
            const double cz = constraint_valid ? C[2].ptr()[cidx] : nan;
            const double cnorm = std::sqrt(cx * cx + cy * cy + cz * cz);

            const double theta = grid.Theta.ptr()[idx];
            const double zx = grid.Z[0].ptr()[idx];
            const double zy = grid.Z[1].ptr()[idx];
            const double zz = grid.Z[2].ptr()[idx];
            const double znorm = std::sqrt(zx * zx + zy * zy + zz * zz);

            file << x << "," << y << "," << alpha << "," << chi << "," << h << ","
                 << std::abs(h) << "," << mx << "," << my << "," << mz << "," << mnorm << ","
                 << cx << "," << cy << "," << cz << "," << cnorm << "," << theta << "," << zx
                 << "," << zy << "," << zz << "," << znorm << "\n";
        }
    }
}

template <typename HierarchyType>
bool export_slice_hdf5(const Grid &grid, const HierarchyType *hierarchy, size_t step, double time,
                       const std::string &output_dir) {
    if (!tensorium::io::hdf5_available())
        return false;

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;
    std::vector<double> x(nx);
    std::vector<double> y(ny);
    std::vector<double> alpha(nx * ny);
    std::vector<double> chi(nx * ny);
    std::vector<double> mask(nx * ny);

    for (size_t i = 0; i < nx; ++i)
        x[i] = grid.x0 + double(i) * grid.dx;
    for (size_t j = 0; j < ny; ++j)
        y[j] = grid.y0 + double(j) * grid.dy;

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            const size_t ii = ng + i;
            const size_t jj = ng + j;
            const size_t idx = grid.alpha.idx(ii, jj, k);
            const size_t flat = i * ny + j;
            alpha[flat] = grid.alpha.ptr()[idx];
            chi[flat] = grid.chi.ptr()[idx];
            mask[flat] = (alpha[flat] < 0.1) ? 1.0 : 0.0;
        }
    }

    std::stringstream ss;
    ss << output_dir << "/slice_" << std::setw(4) << std::setfill('0') << step << ".h5";
    tensorium::io::HDF5File file(ss.str());
    if (!file.is_open())
        return false;

    (void)tensorium::io::write_string_attribute(file.id(), "tensorium_kind",
                                                "moving_puncture_slice_v1");
    (void)tensorium::io::write_scalar_attribute(file.id(), "step", static_cast<std::uint64_t>(step));
    (void)tensorium::io::write_scalar_attribute(file.id(), "time", time);
    (void)tensorium::io::write_scalar_attribute(file.id(), "nx", static_cast<std::uint64_t>(nx));
    (void)tensorium::io::write_scalar_attribute(file.id(), "ny", static_cast<std::uint64_t>(ny));
    (void)tensorium::io::write_scalar_attribute(file.id(), "dx", grid.dx);
    (void)tensorium::io::write_scalar_attribute(file.id(), "dy", grid.dy);
    (void)tensorium::io::write_scalar_attribute(file.id(), "z", grid.z0 + double(grid.dims.nz / 2) * grid.dz);
    (void)tensorium::io::write_scalar_attribute(
        file.id(), "level_count",
        static_cast<std::uint64_t>(hierarchy ? hierarchy->num_levels() : 1));
    return tensorium::io::write_vector_dataset(file.id(), "x", x) &&
           tensorium::io::write_vector_dataset(file.id(), "y", y) &&
           tensorium::io::write_matrix_dataset(file.id(), "alpha", nx, ny, alpha) &&
           tensorium::io::write_matrix_dataset(file.id(), "chi", nx, ny, chi) &&
           tensorium::io::write_matrix_dataset(file.id(), "mask", nx, ny, mask);
}

bool export_constraint_slice_hdf5(const Grid &grid, const tensorium_RG::Field3D<double> &H,
                                  const tensorium_RG::Field3D<double> M[3],
                                  const tensorium_RG::Field3D<double> C[3], size_t step,
                                  double time, const std::string &output_dir) {
    if (!tensorium::io::hdf5_available())
        return false;

    constexpr size_t kConstraintGuard = 4;
    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    std::vector<double> alpha(nx * ny);
    std::vector<double> chi(nx * ny);
    std::vector<double> h(nx * ny);
    std::vector<double> abs_h(nx * ny);
    std::vector<double> mx(nx * ny);
    std::vector<double> my(nx * ny);
    std::vector<double> mz(nx * ny);
    std::vector<double> mnorm(nx * ny);
    std::vector<double> cx(nx * ny);
    std::vector<double> cy(nx * ny);
    std::vector<double> cz(nx * ny);
    std::vector<double> cnorm(nx * ny);
    std::vector<double> theta(nx * ny);
    std::vector<double> zx(nx * ny);
    std::vector<double> zy(nx * ny);
    std::vector<double> zz(nx * ny);
    std::vector<double> znorm(nx * ny);

    for (size_t i = 0; i < nx; ++i)
        x[i] = grid.x0 + double(i) * grid.dx;
    for (size_t j = 0; j < ny; ++j)
        y[j] = grid.y0 + double(j) * grid.dy;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            const size_t ii = ng + i;
            const size_t jj = ng + j;
            const size_t idx = grid.alpha.idx(ii, jj, k);
            const size_t cidx = H.idx(ii, jj, k);
            const size_t flat = i * ny + j;
            const bool constraint_valid =
                (ii >= ng + kConstraintGuard && ii + kConstraintGuard < nx + ng &&
                 jj >= ng + kConstraintGuard && jj + kConstraintGuard < ny + ng);

            alpha[flat] = grid.alpha.ptr()[idx];
            chi[flat] = grid.chi.ptr()[idx];
            h[flat] = constraint_valid ? H.ptr()[cidx] : nan;
            abs_h[flat] = constraint_valid ? std::abs(H.ptr()[cidx]) : nan;
            mx[flat] = constraint_valid ? M[0].ptr()[cidx] : nan;
            my[flat] = constraint_valid ? M[1].ptr()[cidx] : nan;
            mz[flat] = constraint_valid ? M[2].ptr()[cidx] : nan;
            mnorm[flat] = constraint_valid ? std::sqrt(mx[flat] * mx[flat] + my[flat] * my[flat] +
                                                       mz[flat] * mz[flat])
                                           : nan;
            cx[flat] = constraint_valid ? C[0].ptr()[cidx] : nan;
            cy[flat] = constraint_valid ? C[1].ptr()[cidx] : nan;
            cz[flat] = constraint_valid ? C[2].ptr()[cidx] : nan;
            cnorm[flat] = constraint_valid ? std::sqrt(cx[flat] * cx[flat] + cy[flat] * cy[flat] +
                                                       cz[flat] * cz[flat])
                                           : nan;
            theta[flat] = grid.Theta.ptr()[idx];
            zx[flat] = grid.Z[0].ptr()[idx];
            zy[flat] = grid.Z[1].ptr()[idx];
            zz[flat] = grid.Z[2].ptr()[idx];
            znorm[flat] = std::sqrt(zx[flat] * zx[flat] + zy[flat] * zy[flat] + zz[flat] * zz[flat]);
        }
    }

    std::stringstream ss;
    ss << output_dir << "/constraint_slice_" << std::setw(4) << std::setfill('0') << step
       << ".h5";
    tensorium::io::HDF5File file(ss.str());
    if (!file.is_open())
        return false;

    (void)tensorium::io::write_string_attribute(file.id(), "tensorium_kind",
                                                "moving_puncture_constraint_slice_v1");
    (void)tensorium::io::write_scalar_attribute(file.id(), "step", static_cast<std::uint64_t>(step));
    (void)tensorium::io::write_scalar_attribute(file.id(), "time", time);
    (void)tensorium::io::write_scalar_attribute(file.id(), "nx", static_cast<std::uint64_t>(nx));
    (void)tensorium::io::write_scalar_attribute(file.id(), "ny", static_cast<std::uint64_t>(ny));
    (void)tensorium::io::write_scalar_attribute(file.id(), "dx", grid.dx);
    (void)tensorium::io::write_scalar_attribute(file.id(), "dy", grid.dy);
    return tensorium::io::write_vector_dataset(file.id(), "x", x) &&
           tensorium::io::write_vector_dataset(file.id(), "y", y) &&
           tensorium::io::write_matrix_dataset(file.id(), "alpha", nx, ny, alpha) &&
           tensorium::io::write_matrix_dataset(file.id(), "chi", nx, ny, chi) &&
           tensorium::io::write_matrix_dataset(file.id(), "H", nx, ny, h) &&
           tensorium::io::write_matrix_dataset(file.id(), "abs_H", nx, ny, abs_h) &&
           tensorium::io::write_matrix_dataset(file.id(), "Mx", nx, ny, mx) &&
           tensorium::io::write_matrix_dataset(file.id(), "My", nx, ny, my) &&
           tensorium::io::write_matrix_dataset(file.id(), "Mz", nx, ny, mz) &&
           tensorium::io::write_matrix_dataset(file.id(), "Mnorm", nx, ny, mnorm) &&
           tensorium::io::write_matrix_dataset(file.id(), "Cx", nx, ny, cx) &&
           tensorium::io::write_matrix_dataset(file.id(), "Cy", nx, ny, cy) &&
           tensorium::io::write_matrix_dataset(file.id(), "Cz", nx, ny, cz) &&
           tensorium::io::write_matrix_dataset(file.id(), "Cnorm", nx, ny, cnorm) &&
           tensorium::io::write_matrix_dataset(file.id(), "Theta", nx, ny, theta) &&
           tensorium::io::write_matrix_dataset(file.id(), "Zx", nx, ny, zx) &&
           tensorium::io::write_matrix_dataset(file.id(), "Zy", nx, ny, zy) &&
           tensorium::io::write_matrix_dataset(file.id(), "Zz", nx, ny, zz) &&
           tensorium::io::write_matrix_dataset(file.id(), "Znorm", nx, ny, znorm);
}

size_t parse_env_stride_or(const char *name, size_t fallback) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && parsed > 0)
            return static_cast<size_t>(parsed);
    }
    return fallback;
}

bool parse_env_bool_or(const char *name, bool fallback) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw)
            return parsed != 0;
    }
    return fallback;
}

std::array<double, 3> grid_center(const Grid &grid) {
    return {grid.x0 + 0.5 * double(grid.dims.nx - 1) * grid.dx,
            grid.y0 + 0.5 * double(grid.dims.ny - 1) * grid.dy,
            grid.z0 + 0.5 * double(grid.dims.nz - 1) * grid.dz};
}

double grid_max_spacing(const Grid &grid) {
    return std::max({double(grid.dx), double(grid.dy), double(grid.dz)});
}

bool point_in_grid_physical_interior(const Grid &grid, double x, double y, double z,
                                     size_t guard_cells) {
    if (grid.dims.nx <= 2 * guard_cells || grid.dims.ny <= 2 * guard_cells ||
        grid.dims.nz <= 2 * guard_cells) {
        return false;
    }

    const double x_min = grid.x0 + double(guard_cells) * grid.dx;
    const double y_min = grid.y0 + double(guard_cells) * grid.dy;
    const double z_min = grid.z0 + double(guard_cells) * grid.dz;
    const double x_max = grid.x0 + double(grid.dims.nx - 1 - guard_cells) * grid.dx;
    const double y_max = grid.y0 + double(grid.dims.ny - 1 - guard_cells) * grid.dy;
    const double z_max = grid.z0 + double(grid.dims.nz - 1 - guard_cells) * grid.dz;
    return x >= x_min && x <= x_max && y >= y_min && y <= y_max && z >= z_min && z <= z_max;
}

template <typename FieldLike>
bool sample_scalar_tracker_interp(const Grid &grid, const FieldLike &field, double x, double y,
                                  double z, double &value) {
    if (!point_in_grid_physical_interior(grid, x, y, z, 0))
        return false;
    value = tensorium_RG::z4c::fmr::detail::sample_trilinear_field(grid, field, x, y, z);
    return std::isfinite(value);
}

bool sample_vector3_tracker_interp(const Grid &grid, const tensorium_RG::Field3D<double> field[3],
                                   const std::array<double, 3> &p, std::array<double, 3> &out) {
    for (int d = 0; d < 3; ++d) {
        if (!sample_scalar_tracker_interp(grid, field[d], p[0], p[1], p[2], out[d]))
            return false;
    }
    return true;
}

void clamp_tracker_to_domain(const Grid &grid, std::array<double, 3> &p) {
    const double x_min = grid.x0;
    const double y_min = grid.y0;
    const double z_min = grid.z0;
    const double x_max = grid.x0 + double(grid.dims.nx - 1) * grid.dx;
    const double y_max = grid.y0 + double(grid.dims.ny - 1) * grid.dy;
    const double z_max = grid.z0 + double(grid.dims.nz - 1) * grid.dz;
    p[0] = std::clamp(p[0], x_min, x_max);
    p[1] = std::clamp(p[1], y_min, y_max);
    p[2] = std::clamp(p[2], z_min, z_max);
}

struct PuncturePlaneSample {
    double x_left = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double chi_left = std::numeric_limits<double>::quiet_NaN();
    double alpha_left = std::numeric_limits<double>::quiet_NaN();
    bool   has_left = false;

    double x_right = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
    double chi_right = std::numeric_limits<double>::quiet_NaN();
    double alpha_right = std::numeric_limits<double>::quiet_NaN();
    bool   has_right = false;
};

struct ShiftPunctureTracker {
    std::array<double, 3> p1{0.0, 0.0, 0.0};
    std::array<double, 3> p2{0.0, 0.0, 0.0};
    std::array<double, 3> beta1_prev{0.0, 0.0, 0.0};
    std::array<double, 3> beta2_prev{0.0, 0.0, 0.0};
    bool                  initialized = false;
};

void refresh_shift_puncture_tracker_beta(const Grid &grid, ShiftPunctureTracker &tracker) {
    if (!tracker.initialized)
        return;
    std::array<double, 3> b1{0.0, 0.0, 0.0};
    std::array<double, 3> b2{0.0, 0.0, 0.0};
    if (sample_vector3_tracker_interp(grid, grid.beta, tracker.p1, b1))
        tracker.beta1_prev = b1;
    if (sample_vector3_tracker_interp(grid, grid.beta, tracker.p2, b2))
        tracker.beta2_prev = b2;
}

void initialize_shift_puncture_tracker(const Grid &grid, ShiftPunctureTracker &tracker,
                                       const PuncturePlaneSample &sample) {
    if (!(sample.has_left && sample.has_right))
        return;
    tracker = ShiftPunctureTracker{};
    tracker.p1 = {sample.x_left, sample.y_left, 0.0};
    tracker.p2 = {sample.x_right, sample.y_right, 0.0};
    clamp_tracker_to_domain(grid, tracker.p1);
    clamp_tracker_to_domain(grid, tracker.p2);
    tracker.initialized = true;
    refresh_shift_puncture_tracker_beta(grid, tracker);
}

void advance_shift_puncture_tracker(const Grid &grid, ShiftPunctureTracker &tracker, double dt) {
    if (!tracker.initialized || !std::isfinite(dt) || dt <= 0.0)
        return;

    std::array<double, 3> b1_new = tracker.beta1_prev;
    std::array<double, 3> b2_new = tracker.beta2_prev;
    (void)sample_vector3_tracker_interp(grid, grid.beta, tracker.p1, b1_new);
    (void)sample_vector3_tracker_interp(grid, grid.beta, tracker.p2, b2_new);

    for (int d = 0; d < 3; ++d) {
        tracker.p1[d] += -0.5 * dt * (tracker.beta1_prev[d] + b1_new[d]);
        tracker.p2[d] += -0.5 * dt * (tracker.beta2_prev[d] + b2_new[d]);
    }
    clamp_tracker_to_domain(grid, tracker.p1);
    clamp_tracker_to_domain(grid, tracker.p2);
    tracker.beta1_prev = b1_new;
    tracker.beta2_prev = b2_new;
}

PuncturePlaneSample make_tracker_sample(const Grid &grid, const ShiftPunctureTracker &tracker) {
    PuncturePlaneSample sample;
    if (!tracker.initialized)
        return sample;

    auto fill = [&](const std::array<double, 3> &p, bool &has_point, double &x_out, double &y_out,
                    double &chi_out, double &alpha_out) {
        double alpha = std::numeric_limits<double>::quiet_NaN();
        double chi = std::numeric_limits<double>::quiet_NaN();
        if (!sample_scalar_tracker_interp(grid, grid.alpha, p[0], p[1], p[2], alpha))
            return;
        if (!sample_scalar_tracker_interp(grid, grid.chi, p[0], p[1], p[2], chi))
            return;
        has_point = true;
        x_out = p[0];
        y_out = p[1];
        chi_out = chi;
        alpha_out = alpha;
    };

    fill(tracker.p1, sample.has_left, sample.x_left, sample.y_left, sample.chi_left, sample.alpha_left);
    fill(tracker.p2, sample.has_right, sample.x_right, sample.y_right, sample.chi_right,
         sample.alpha_right);
    return sample;
}

void refresh_shift_puncture_tracker_beta_split(const Grid &left_grid, const Grid &right_grid,
                                               ShiftPunctureTracker &tracker) {
    if (!tracker.initialized)
        return;
    std::array<double, 3> b1{0.0, 0.0, 0.0};
    std::array<double, 3> b2{0.0, 0.0, 0.0};
    if (sample_vector3_tracker_interp(left_grid, left_grid.beta, tracker.p1, b1))
        tracker.beta1_prev = b1;
    if (sample_vector3_tracker_interp(right_grid, right_grid.beta, tracker.p2, b2))
        tracker.beta2_prev = b2;
}

void initialize_shift_puncture_tracker_split(const Grid &root_grid, const Grid &left_grid,
                                             const Grid &right_grid, ShiftPunctureTracker &tracker,
                                             const std::array<double, 3> &p1,
                                             const std::array<double, 3> &p2) {
    tracker = ShiftPunctureTracker{};
    tracker.p1 = p1;
    tracker.p2 = p2;
    clamp_tracker_to_domain(root_grid, tracker.p1);
    clamp_tracker_to_domain(root_grid, tracker.p2);
    tracker.initialized = true;
    refresh_shift_puncture_tracker_beta_split(left_grid, right_grid, tracker);
}

void advance_shift_puncture_tracker_split(const Grid &root_grid, const Grid &left_grid,
                                          const Grid &right_grid, ShiftPunctureTracker &tracker,
                                          double dt) {
    if (!tracker.initialized || !std::isfinite(dt) || dt <= 0.0)
        return;

    std::array<double, 3> b1_new = tracker.beta1_prev;
    std::array<double, 3> b2_new = tracker.beta2_prev;
    (void)sample_vector3_tracker_interp(left_grid, left_grid.beta, tracker.p1, b1_new);
    (void)sample_vector3_tracker_interp(right_grid, right_grid.beta, tracker.p2, b2_new);

    for (int d = 0; d < 3; ++d) {
        tracker.p1[d] += -0.5 * dt * (tracker.beta1_prev[d] + b1_new[d]);
        tracker.p2[d] += -0.5 * dt * (tracker.beta2_prev[d] + b2_new[d]);
    }
    clamp_tracker_to_domain(root_grid, tracker.p1);
    clamp_tracker_to_domain(root_grid, tracker.p2);
    tracker.beta1_prev = b1_new;
    tracker.beta2_prev = b2_new;
}

PuncturePlaneSample make_tracker_sample_split(const Grid &left_grid, const Grid &right_grid,
                                              const ShiftPunctureTracker &tracker) {
    PuncturePlaneSample sample;
    if (!tracker.initialized)
        return sample;

    sample.has_left =
        sample_scalar_tracker_interp(left_grid, left_grid.alpha, tracker.p1[0], tracker.p1[1],
                                     tracker.p1[2], sample.alpha_left) &&
        sample_scalar_tracker_interp(left_grid, left_grid.chi, tracker.p1[0], tracker.p1[1],
                                     tracker.p1[2], sample.chi_left);
    if (sample.has_left) {
        sample.x_left = tracker.p1[0];
        sample.y_left = tracker.p1[1];
    }

    sample.has_right =
        sample_scalar_tracker_interp(right_grid, right_grid.alpha, tracker.p2[0], tracker.p2[1],
                                     tracker.p2[2], sample.alpha_right) &&
        sample_scalar_tracker_interp(right_grid, right_grid.chi, tracker.p2[0], tracker.p2[1],
                                     tracker.p2[2], sample.chi_right);
    if (sample.has_right) {
        sample.x_right = tracker.p2[0];
        sample.y_right = tracker.p2[1];
    }

    return sample;
}

bool compute_tracker_drift(const PuncturePlaneSample &tracked, const PuncturePlaneSample &minima,
                           double &drift_left, double &drift_right) {
    if (!(tracked.has_left && tracked.has_right && minima.has_left && minima.has_right))
        return false;
    drift_left = std::hypot(tracked.x_left - minima.x_left, tracked.y_left - minima.y_left);
    drift_right = std::hypot(tracked.x_right - minima.x_right, tracked.y_right - minima.y_right);
    return std::isfinite(drift_left) && std::isfinite(drift_right);
}

struct PunctureMinimumSample {
    double x = std::numeric_limits<double>::quiet_NaN();
    double y = std::numeric_limits<double>::quiet_NaN();
    double chi = std::numeric_limits<double>::quiet_NaN();
    double alpha = std::numeric_limits<double>::quiet_NaN();
    bool   valid = false;
};

PunctureMinimumSample sample_grid_minimum_alpha(const Grid &grid) {
    PunctureMinimumSample sample;
    const size_t          nx = grid.dims.nx;
    const size_t          ny = grid.dims.ny;
    const size_t          ng = grid.dims.ng;
    const size_t          k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const size_t idx = grid.alpha.idx(i, j, k);
            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];
            if (!std::isfinite(alpha) || !std::isfinite(chi))
                continue;
            if (!sample.valid || alpha < sample.alpha) {
                sample.valid = true;
                sample.x = grid.x0 + (double(i) - double(ng)) * grid.dx;
                sample.y = grid.y0 + (double(j) - double(ng)) * grid.dy;
                sample.chi = chi;
                sample.alpha = alpha;
            }
        }
    }

    return sample;
}

PuncturePlaneSample sample_puncture_minima_split(const Grid &left_grid, const Grid &right_grid) {
    PuncturePlaneSample sample;
    const auto left = sample_grid_minimum_alpha(left_grid);
    const auto right = sample_grid_minimum_alpha(right_grid);
    if (left.valid) {
        sample.has_left = true;
        sample.x_left = left.x;
        sample.y_left = left.y;
        sample.chi_left = left.chi;
        sample.alpha_left = left.alpha;
    }
    if (right.valid) {
        sample.has_right = true;
        sample.x_right = right.x;
        sample.y_right = right.y;
        sample.chi_right = right.chi;
        sample.alpha_right = right.alpha;
    }
    return sample;
}

struct ConstraintScratch {
    tensorium_RG::Field3D<double> H;
    tensorium_RG::Field3D<double> M[3];
    tensorium_RG::Field3D<double> C[3];

    explicit ConstraintScratch(const Grid &grid) : H(tensorium_RG::make_field(grid.alpha.st)) {
        for (int q = 0; q < 3; ++q) {
            M[q] = tensorium_RG::make_field(grid.alpha.st);
            C[q] = tensorium_RG::make_field(grid.alpha.st);
        }
    }
};

tensorium_RG::z4c::ConstraintMonitorStats
compute_constraint_stats(Grid &grid, ConstraintScratch &scratch, size_t padding) {
    tensorium_RG::fd::set_fd_dx(grid.dx);
    const double min_extent =
        std::min({(grid.dims.nx - 1) * grid.dx, (grid.dims.ny - 1) * grid.dy,
                  (grid.dims.nz - 1) * grid.dz});
    const double r_min = 2.0 * std::min({grid.dx, grid.dy, grid.dz});
    const double r_max = 0.45 * min_extent;
    tensorium_RG::z4c::compute_z4c_constraints(grid, grid.Ricci, scratch.H, scratch.M, scratch.C,
                                                 r_min, r_max, 0.0, 0.0, 0.0);
    auto stats = tensorium_RG::z4c::compute_constraint_monitor(grid, scratch.H, padding);
    tensorium_RG::z4c::populate_constraint_norms(grid, scratch.M, stats, padding);
    return stats;
}

void compute_constraint_slice_fields(Grid &grid, ConstraintScratch &scratch) {
    tensorium_RG::fd::set_fd_dx(grid.dx);
    tensorium_RG::z4c::compute_z4c_constraints(grid, grid.Ricci, scratch.H, scratch.M, scratch.C,
                                               0.0, std::numeric_limits<double>::max(), 0.0, 0.0,
                                               0.0, false);
}

void initialize_moving_puncture_level(Grid &grid, const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
                                      const tensorium_RG::z4c::ProjectionConfig &proj_cfg,
                                      bool zero_z4c = true) {
    tensorium_RG::z4c::initialize_moving_puncture_data(grid, cfg);
    tensorium_RG::z4c::project_z4c_state(grid, proj_cfg);
    if (zero_z4c)
        tensorium_RG::init::zero_z4c_fields(grid);
}

Grid make_grid_like(const Grid &grid) {
    Grid out(grid.dims.nx, grid.dims.ny, grid.dims.nz, grid.dims.ng, grid.dx, grid.dy, grid.dz);
    out.x0 = grid.x0;
    out.y0 = grid.y0;
    out.z0 = grid.z0;
    return out;
}

void remap_field_overlap_from_reference(const Grid &reference, Grid &target, size_t guard_cells,
                                        tensorium_RG::z4c::BoundaryField which, int component) {
    const auto &src = tensorium_RG::z4c::fmr::detail::select_field(reference, which, component);
    auto       &dst = tensorium_RG::z4c::fmr::detail::select_field(target, which, component);

    size_t I0, I1, J0, J1, K0, K1;
    target.domain_bounds(I0, I1, J0, J1, K0, K1);

#pragma omp parallel for collapse(3)
    for (size_t i = I0; i < I1; ++i)
        for (size_t j = J0; j < J1; ++j)
            for (size_t k = K0; k < K1; ++k) {
                double x, y, z;
                target.coords(i, j, k, x, y, z);
                if (!point_in_grid_physical_interior(reference, x, y, z, guard_cells))
                    continue;
                dst.ptr()[dst.idx(i, j, k)] =
                    tensorium_RG::z4c::fmr::detail::sample_trilinear_field(reference, src, x, y, z);
            }
}

void inject_overlap_from_reference(const Grid &reference, Grid &target, size_t guard_cells) {
    remap_field_overlap_from_reference(reference, target, guard_cells,
                                       tensorium_RG::z4c::BoundaryField::Alpha, 0);
    remap_field_overlap_from_reference(reference, target, guard_cells,
                                       tensorium_RG::z4c::BoundaryField::Chi, 0);
    remap_field_overlap_from_reference(reference, target, guard_cells,
                                       tensorium_RG::z4c::BoundaryField::K, 0);
    remap_field_overlap_from_reference(reference, target, guard_cells,
                                       tensorium_RG::z4c::BoundaryField::Theta, 0);
    for (int a = 0; a < 3; ++a) {
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::Beta, a);
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::B, a);
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::TildeGamma, a);
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::Z, a);
    }
    for (int s = 0; s < 6; ++s) {
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::GammaTilde, s);
        remap_field_overlap_from_reference(reference, target, guard_cells,
                                           tensorium_RG::z4c::BoundaryField::ATilde, s);
    }
    tensorium_RG::z4c::enforce_algebraic_constraints(target);
}

double moving_puncture_grid_half_width(const Grid &grid) {
    const auto axis_half_width = [](double origin, double spacing, size_t cells) {
        const double xmin = std::abs(origin - 0.5 * spacing);
        const double xmax = std::abs(origin + (double(cells) - 0.5) * spacing);
        return std::min(xmin, xmax);
    };

    return std::min({axis_half_width(grid.x0, grid.dx, grid.dims.nx),
                     axis_half_width(grid.y0, grid.dy, grid.dims.ny),
                     axis_half_width(grid.z0, grid.dz, grid.dims.nz)});
}

struct MovingPunctureInitBlendRegion {
    double core_half_width = 0.0;
    double transition_width = 0.0;
    double tp_outer_half_width = 0.0;
};

MovingPunctureInitBlendRegion
moving_puncture_init_blend_region(const Grid &grid,
                                  const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg) {
    const double max_half_width =
        std::max(0.0, moving_puncture_grid_half_width(grid) - 0.5 * std::max({grid.dx, grid.dy, grid.dz}));
    const double physical_buffer =
        (cfg.fmr.puncture_buffer > 0.0) ? cfg.fmr.puncture_buffer : std::max(2.0, 4.0 * cfg.spacing);
    const double requested_transition = (cfg.fmr.init_transition_width > 0.0)
                                            ? cfg.fmr.init_transition_width
                                            : std::max(0.5 * physical_buffer, 6.0 * std::max({grid.dx, grid.dy, grid.dz}));

    const double desired_outer_half_width =
        (cfg.fmr.init_core_half_width > 0.0) ? (cfg.fmr.init_core_half_width + requested_transition)
                                             : (cfg.separation + physical_buffer);
    const double tp_outer_half_width = std::clamp(desired_outer_half_width, 0.0, max_half_width);
    const double requested_core_half_width =
        (cfg.fmr.init_core_half_width > 0.0) ? cfg.fmr.init_core_half_width
                                             : std::max(0.0, tp_outer_half_width - requested_transition);
    const double core_half_width = std::clamp(requested_core_half_width, 0.0, tp_outer_half_width);

    MovingPunctureInitBlendRegion region;
    region.core_half_width = core_half_width;
    region.transition_width = std::max(0.0, tp_outer_half_width - core_half_width);
    region.tp_outer_half_width = tp_outer_half_width;
    return region;
}

void initialize_moving_puncture_hierarchy(MovingPunctureHierarchy &hierarchy,
                                          const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
                                          const tensorium_RG::z4c::ProjectionConfig &proj_cfg) {
    initialize_moving_puncture_level(hierarchy.root_grid(), cfg, proj_cfg, true);
    tensorium_RG::fd::set_fd_dx(hierarchy.root_grid().dx);
    tensorium_RG::z4c::apply_halos_grid<tensorium_RG::z4c::BoundaryRadiative>(hierarchy.root_grid());

    for (size_t level = 1; level < hierarchy.num_levels(); ++level) {
        hierarchy.prolongate_level_from_parent(level);
        tensorium_RG::z4c::project_z4c_state(hierarchy.level_grid(level), proj_cfg);

        if (cfg.fmr.fine_levels_use_parent_init) {
            hierarchy.apply_level_boundaries();
            std::cout << "[fmr.init] level=" << level << " source=parent_prolongation"
                      << std::endl;
            continue;
        }

        Grid tp_reference = make_grid_like(hierarchy.level_grid(level));
        initialize_moving_puncture_level(tp_reference, cfg, proj_cfg, true);

        const auto region = moving_puncture_init_blend_region(hierarchy.level_grid(level), cfg);
        hierarchy.blend_level_centered_core_from_reference(level, tp_reference,
                                                           region.core_half_width,
                                                           region.transition_width);
        tensorium_RG::z4c::project_z4c_state(hierarchy.level_grid(level), proj_cfg);
        hierarchy.apply_level_boundaries();

        std::cout << "[fmr.init] level=" << level << " core_half_width=" << region.core_half_width
                  << " transition_width=" << region.transition_width
                  << " tp_outer_half_width=" << region.tp_outer_half_width << std::endl;
    }

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

void project_hierarchy_levels(MovingPunctureHierarchy &hierarchy,
                              const tensorium_RG::z4c::ProjectionConfig &proj_cfg) {
    for (size_t level = 0; level < hierarchy.num_levels(); ++level)
        tensorium_RG::z4c::project_z4c_state(hierarchy.level_grid(level), proj_cfg);

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

std::unique_ptr<MovingPunctureHierarchy>
rebuild_moving_puncture_hierarchy(const MovingPunctureHierarchy *previous,
                                  const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
                                  const tensorium_RG::z4c::ProjectionConfig &proj_cfg,
                                  const std::array<double, 3> &center) {
    Grid root_copy = make_grid_like(previous->root_grid());
    tensorium_RG::z4c::fmr::detail::copy_evolved_state(previous->root_grid(), root_copy);

    const auto level_cfgs = tensorium_RG::z4c::build_moving_puncture_fmr_levels(root_copy, cfg, center);
    auto hierarchy = std::make_unique<MovingPunctureHierarchy>(root_copy, level_cfgs, cfg.padding);

    for (size_t level = 1; level < hierarchy->num_levels(); ++level) {
        hierarchy->prolongate_level_from_parent(level);
        if (previous != nullptr && level < previous->num_levels())
            inject_overlap_from_reference(previous->level_grid(level), hierarchy->level_grid(level), 2);
        tensorium_RG::z4c::project_z4c_state(hierarchy->level_grid(level), proj_cfg);
    }
    hierarchy->apply_level_boundaries();
    return hierarchy;
}

void initialize_moving_puncture_binary_hierarchy(
    BinaryMovingPunctureHierarchy &hierarchy,
    const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
    const tensorium_RG::z4c::ProjectionConfig &proj_cfg) {
    initialize_moving_puncture_level(hierarchy.root_grid(), cfg, proj_cfg, true);
    tensorium_RG::fd::set_fd_dx(hierarchy.root_grid().dx);
    tensorium_RG::z4c::apply_halos_grid<tensorium_RG::z4c::BoundaryRadiative>(hierarchy.root_grid());

    for (size_t level = 1; level < hierarchy.num_shared_levels(); ++level) {
        hierarchy.prolongate_shared_level_from_parent(level);
        tensorium_RG::z4c::project_z4c_state(hierarchy.shared_level_grid(level), proj_cfg);

        if (cfg.fmr.fine_levels_use_parent_init) {
            hierarchy.apply_level_boundaries();
            std::cout << "[fmr.init] level=" << level << " source=parent_prolongation" << std::endl;
            continue;
        }

        Grid tp_reference = make_grid_like(hierarchy.shared_level_grid(level));
        initialize_moving_puncture_level(tp_reference, cfg, proj_cfg, true);
        const auto region = moving_puncture_init_blend_region(hierarchy.shared_level_grid(level), cfg);
        hierarchy.blend_shared_level_centered_core_from_reference(level, tp_reference,
                                                                  region.core_half_width,
                                                                  region.transition_width);
        tensorium_RG::z4c::project_z4c_state(hierarchy.shared_level_grid(level), proj_cfg);
        hierarchy.apply_level_boundaries();
        std::cout << "[fmr.init] level=" << level << " core_half_width=" << region.core_half_width
                  << " transition_width=" << region.transition_width
                  << " tp_outer_half_width=" << region.tp_outer_half_width << std::endl;
    }

    for (size_t leaf = 0; leaf < hierarchy.num_leaves(); ++leaf) {
        hierarchy.prolongate_leaf_from_parent(leaf);
        tensorium_RG::z4c::project_z4c_state(hierarchy.leaf_grid(leaf), proj_cfg);

        const size_t flat_level = hierarchy.num_shared_levels() + leaf;
        if (cfg.fmr.fine_levels_use_parent_init) {
            hierarchy.apply_level_boundaries();
            std::cout << "[fmr.init] level=" << flat_level << " source=parent_prolongation"
                      << std::endl;
            continue;
        }

        Grid tp_reference = make_grid_like(hierarchy.leaf_grid(leaf));
        initialize_moving_puncture_level(tp_reference, cfg, proj_cfg, true);
        const auto region = moving_puncture_init_blend_region(hierarchy.leaf_grid(leaf), cfg);
        hierarchy.blend_leaf_centered_core_from_reference(leaf, tp_reference,
                                                          region.core_half_width,
                                                          region.transition_width);
        tensorium_RG::z4c::project_z4c_state(hierarchy.leaf_grid(leaf), proj_cfg);
        hierarchy.apply_level_boundaries();
        std::cout << "[fmr.init] level=" << flat_level << " core_half_width=" << region.core_half_width
                  << " transition_width=" << region.transition_width
                  << " tp_outer_half_width=" << region.tp_outer_half_width << std::endl;
    }

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

void project_binary_hierarchy_levels(BinaryMovingPunctureHierarchy &hierarchy,
                                     const tensorium_RG::z4c::ProjectionConfig &proj_cfg) {
    for (size_t level = 0; level < hierarchy.num_levels(); ++level)
        tensorium_RG::z4c::project_z4c_state(hierarchy.level_grid(level), proj_cfg);

    hierarchy.restrict_all_levels_to_root();
    hierarchy.apply_level_boundaries();
}

std::unique_ptr<BinaryMovingPunctureHierarchy>
rebuild_moving_puncture_binary_hierarchy(
    const BinaryMovingPunctureHierarchy *previous,
    const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
    const tensorium_RG::z4c::ProjectionConfig &proj_cfg, const std::array<double, 3> &p1,
    const std::array<double, 3> &p2) {
    Grid root_copy = make_grid_like(previous->root_grid());
    tensorium_RG::z4c::fmr::detail::copy_evolved_state(previous->root_grid(), root_copy);

    const auto hierarchy_cfg =
        tensorium_RG::z4c::build_moving_puncture_bbh_split_hierarchy_config(root_copy, cfg, p1, p2);
    auto hierarchy =
        std::make_unique<BinaryMovingPunctureHierarchy>(root_copy, hierarchy_cfg, cfg.padding);

    for (size_t level = 1; level < hierarchy->num_shared_levels(); ++level) {
        hierarchy->prolongate_shared_level_from_parent(level);
        if (previous != nullptr && level < previous->num_shared_levels()) {
            inject_overlap_from_reference(previous->shared_level_grid(level),
                                          hierarchy->shared_level_grid(level), 2);
        }
        tensorium_RG::z4c::project_z4c_state(hierarchy->shared_level_grid(level), proj_cfg);
    }

    for (size_t leaf = 0; leaf < hierarchy->num_leaves(); ++leaf) {
        hierarchy->prolongate_leaf_from_parent(leaf);
        if (previous != nullptr && leaf < previous->num_leaves())
            inject_overlap_from_reference(previous->leaf_grid(leaf), hierarchy->leaf_grid(leaf), 2);
        tensorium_RG::z4c::project_z4c_state(hierarchy->leaf_grid(leaf), proj_cfg);
    }

    hierarchy->apply_level_boundaries();
    return hierarchy;
}

PuncturePlaneSample sample_puncture_minima(const Grid &grid) {
    PuncturePlaneSample sample;

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;
            const size_t idx = grid.alpha.idx(i, j, k);
            const double alpha = grid.alpha.ptr()[idx];
            const double chi = grid.chi.ptr()[idx];

            if (!std::isfinite(alpha) || !std::isfinite(chi))
                continue;

            if (x <= 0.0) {
                if (!sample.has_left || alpha < sample.alpha_left) {
                    sample.has_left = true;
                    sample.x_left = x;
                    sample.y_left = y;
                    sample.chi_left = chi;
                    sample.alpha_left = alpha;
                }
            } else {
                if (!sample.has_right || alpha < sample.alpha_right) {
                    sample.has_right = true;
                    sample.x_right = x;
                    sample.y_right = y;
                    sample.chi_right = chi;
                    sample.alpha_right = alpha;
                }
            }
        }
    }

    return sample;
}

} // namespace

int main(int argc, char **argv) {
    const auto cfg = tensorium_RG::z4c::load_moving_puncture_env();
    size_t output_stride = cfg.state_log_stride;
    size_t slice_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 10);
    size_t constraint_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_EXPORT_STRIDE", 5);
    bool export_constraint_slices =
        parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EXPORT_CONSTRAINT_SLICES", true);
    const bool export_hdf5 =
        parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EXPORT_HDF5", false);
    size_t constraint_slice_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_SLICE_STRIDE",
                            constraint_export_stride);
    size_t fmr_profile_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_FMR_PROFILE_STRIDE", 0);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output-stride") == 0 && i + 1 < argc) {
            output_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--slice-export-stride") == 0 && i + 1 < argc) {
            slice_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--constraint-export-stride") == 0 && i + 1 < argc) {
            constraint_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--constraint-slice-stride") == 0 && i + 1 < argc) {
            constraint_slice_stride = static_cast<size_t>(std::stoul(argv[++i]));
        }
    }

    std::cout << "[mesh] nx=" << cfg.nx << " ny=" << cfg.ny << " nz=" << cfg.nz
              << " spacing=" << cfg.spacing
              << " box=(" << cfg.nx * cfg.spacing << ", " << cfg.ny * cfg.spacing << ", "
              << cfg.nz * cfg.spacing << ")\n";

    tensorium_RG::fd::set_max_spatial_derivative_order(cfg.spatial_derivative_order);
    std::cout << "[num] spatial_derivative_order=" << cfg.spatial_derivative_order << std::endl;
    tensorium_RG::fd::set_fd_dx(cfg.spacing);

    std::filesystem::create_directories("Output/viz");

    Grid root_grid(cfg.nx, cfg.ny, cfg.nz, cfg.ng, cfg.spacing, cfg.spacing, cfg.spacing);
    tensorium_RG::z4c::center_cell_centered_origin(root_grid);
    const bool use_bbh_split =
        cfg.fmr.enabled && cfg.fmr.layout == tensorium_RG::z4c::MovingPunctureFMRLayout::BBHSplit;
    const auto initial_punctures = tensorium_RG::z4c::moving_puncture_initial_puncture_positions(cfg);
    std::vector<tensorium_RG::z4c::fmr::LevelConfig> level_cfgs;
    tensorium_RG::z4c::fmr::BinaryPunctureHierarchyConfig binary_hierarchy_cfg;
    bool use_fmr = false;
    if (cfg.fmr.enabled) {
        if (use_bbh_split) {
            binary_hierarchy_cfg = tensorium_RG::z4c::build_moving_puncture_bbh_split_hierarchy_config(
                root_grid, cfg, initial_punctures[0], initial_punctures[1]);
            use_fmr = !binary_hierarchy_cfg.shared_levels.empty();
        } else {
            level_cfgs = tensorium_RG::z4c::build_moving_puncture_fmr_levels(root_grid, cfg);
            use_fmr = !level_cfgs.empty();
        }
    }

    if (cfg.print_suggested_momentum) {
        if (cfg.circular_hint.valid) {
            std::cout << "[Z4c.Init] Physics Diagnostic for d = " << cfg.circular_hint.d
                      << "\n[Z4c.Init] > Suggested P_tang (Newtonian): "
                      << cfg.circular_hint.p_newtonian
                      << "\n[Z4c.Init] > Suggested P_tang (Post-Newtonian): "
                      << cfg.circular_hint.p_pn
                      << "\n[Z4c.Init] Current P_tang (User): " << cfg.user_tangential_momentum
                      << std::endl;
        } else {
            std::cout << "[Z4c.Init] Physics Diagnostic unavailable: invalid masses/separation."
                      << std::endl;
        }
    }
    if (cfg.auto_circular) {
        if (cfg.circular_hint.valid) {
            std::cout << "[Z4c.Init] AUTO_CIRCULAR enabled: overriding P_tang from "
                      << cfg.user_tangential_momentum << " to " << cfg.tangential_momentum
                      << " (Post-Newtonian)." << std::endl;
        } else {
            std::cout << "[Z4c.Init] AUTO_CIRCULAR requested but suggestion is invalid; keeping "
                         "user momentum."
                      << std::endl;
        }
    }

    std::cout << "[init] puncture separation=" << cfg.separation
              << " momentum_tan=" << cfg.tangential_momentum
              << " momentum_rad=" << cfg.radial_momentum
              << " m1=" << cfg.mass1
              << " m2=" << cfg.mass2 << std::endl;
    std::cout << "[init] mode="
              << (cfg.use_interpolated_init ? tensorium_RG::z4c::moving_puncture_interpolated_mode_name()
                                            : "bowen_york")
              << " seed_n=" << cfg.interp_seed_n << std::endl;

    tensorium_RG::z4c::apply_boundary_configuration(cfg);

    tensorium_RG::z4c::ProjectionConfig proj_cfg;
    proj_cfg.padding = 0;
    proj_cfg.renormalize_metric = true;
    proj_cfg.project_A_tilde = true;
    proj_cfg.recompute_inverse = true;

    std::unique_ptr<MovingPunctureHierarchy>       hierarchy;
    std::unique_ptr<BinaryMovingPunctureHierarchy> binary_hierarchy;
    Grid                                          *root_state = &root_grid;
    Grid                                          *puncture_state = &root_grid;
    Grid                                          *left_puncture_state = nullptr;
    Grid                                          *right_puncture_state = nullptr;
    if (use_fmr) {
        if (use_bbh_split) {
            binary_hierarchy =
                std::make_unique<BinaryMovingPunctureHierarchy>(root_grid, binary_hierarchy_cfg,
                                                                cfg.padding);
            initialize_moving_puncture_binary_hierarchy(*binary_hierarchy, cfg, proj_cfg);
            root_state = &binary_hierarchy->root_grid();
            left_puncture_state = &binary_hierarchy->leaf_grid(0);
            right_puncture_state = &binary_hierarchy->leaf_grid(1);
            puncture_state = left_puncture_state;
        } else {
            hierarchy = std::make_unique<MovingPunctureHierarchy>(root_grid, level_cfgs, cfg.padding);
            initialize_moving_puncture_hierarchy(*hierarchy, cfg, proj_cfg);
            root_state = &hierarchy->root_grid();
            puncture_state = &hierarchy->level_grid(hierarchy->num_levels() - 1);
        }
    } else {
        initialize_moving_puncture_level(root_grid, cfg, proj_cfg, true);
        tensorium_RG::fd::set_fd_dx(root_grid.dx);
        tensorium_RG::z4c::apply_halos_grid<tensorium_RG::z4c::BoundaryRadiative>(root_grid);
    }

    auto params = cfg.gauge_params;
    const auto bc_char =
        tensorium_RG::z4c::configure_boundary_characteristics_from_state(*root_state, cfg, params);

    std::cout << "[bc] allow_reflective=" << cfg.allow_reflective_bc
              << " fail_on_gauge_bc_mismatch=" << cfg.fail_on_gauge_bc_mismatch
              << " sponge_enable=" << cfg.sponge.enabled
              << " sponge_width=" << cfg.sponge.width
              << " sponge_strength=" << cfg.sponge.strength
              << " sponge_exponent=" << cfg.sponge.exponent
              << " radiative_collar_width=" << cfg.radiative_collar_width
              << " ko_boundary_width=" << cfg.ko_boundary_width
              << " ko_boundary_floor=" << cfg.ko_boundary_floor
              << " ko_boundary_boost=" << cfg.ko_boundary_boost
              << " ko_edge_corner_boost=" << cfg.ko_edge_corner_boost << std::endl;
    for (int axis = 0; axis < 3; ++axis) {
        std::cout << "[bc] "
                  << tensorium_RG::z4c::describe_boundary_face_mode(
                         cfg.boundary_faces, cfg.sponge, axis, false)
                  << std::endl;
        std::cout << "[bc] "
                  << tensorium_RG::z4c::describe_boundary_face_mode(
                         cfg.boundary_faces, cfg.sponge, axis, true)
                  << std::endl;
    }
    tensorium_RG::z4c::report_boundary_characteristics(
        params, cfg.fail_on_gauge_bc_mismatch, bc_char);

    if (use_fmr) {
        const Grid &outer_refined =
            use_bbh_split ? binary_hierarchy->shared_level_grid(1) : hierarchy->level_grid(1);
        const Grid &finest_refined =
            use_bbh_split ? binary_hierarchy->leaf_grid(0) : hierarchy->level_grid(hierarchy->num_levels() - 1);
        const double outer_half_width =
            0.5 * std::min({double(outer_refined.dims.nx) * outer_refined.dx,
                            double(outer_refined.dims.ny) * outer_refined.dy,
                            double(outer_refined.dims.nz) * outer_refined.dz});
        const double finest_half_width =
            0.5 * std::min({double(finest_refined.dims.nx) * finest_refined.dx,
                            double(finest_refined.dims.ny) * finest_refined.dy,
                            double(finest_refined.dims.nz) * finest_refined.dz});
        const char *sizing_mode = (cfg.fmr.outer_box_half_width > 0.0)
                                      ? "outer_half_width"
                                      : ((cfg.fmr.finest_box_half_width > 0.0)
                                             ? "finest_half_width"
                                             : "auto");
        const char *fine_init_mode =
            cfg.fmr.fine_levels_use_parent_init ? "parent_prolongation" : "local_tp_blend";
        std::cout << "[fmr] enabled=1 levels="
                  << (use_bbh_split ? (binary_hierarchy->num_levels() - 1) : (hierarchy->num_levels() - 1))
                  << " ratio=" << cfg.fmr.refinement_ratio
                  << " layout=" << tensorium_RG::z4c::moving_puncture_fmr_layout_name(cfg.fmr.layout)
                  << " sizing=" << sizing_mode
                  << " fine_init=" << fine_init_mode
                  << " outer_half_width=" << outer_half_width
                  << " finest_half_width=" << finest_half_width
                  << " finest_dx=" << finest_refined.dx
                  << " profile_stride=" << fmr_profile_stride << std::endl;
        const auto bytes_to_gib = [](double bytes) {
            return bytes / (1024.0 * 1024.0 * 1024.0);
        };
        const size_t level_count = use_bbh_split ? binary_hierarchy->num_levels() : hierarchy->num_levels();
        for (size_t level = 1; level < level_count; ++level) {
            const Grid &g = use_bbh_split ? binary_hierarchy->level_grid(level) : hierarchy->level_grid(level);
            const double half_width =
                0.5 * std::min({double(g.dims.nx) * g.dx, double(g.dims.ny) * g.dy,
                                double(g.dims.nz) * g.dz});
            std::cout << "[fmr] level=" << level << " nx=" << g.dims.nx << " ny=" << g.dims.ny
                      << " nz=" << g.dims.nz << " spacing=" << g.dx
                      << " half_width=" << half_width << " box=("
                      << g.dims.nx * g.dx << ", " << g.dims.ny * g.dy << ", "
                      << g.dims.nz * g.dz << ")\n";
        }
        for (size_t level = 0; level < level_count; ++level) {
            const Grid &g = use_bbh_split ? binary_hierarchy->level_grid(level) : hierarchy->level_grid(level);
            const size_t allocated_fields = use_bbh_split ? binary_hierarchy->level_allocated_field_count(level)
                                                          : hierarchy->level_allocated_field_count(level);
            const size_t snapshot_fields =
                use_bbh_split ? binary_hierarchy->level_snapshot_allocated_field_count(level)
                              : hierarchy->level_snapshot_allocated_field_count(level);
            const size_t snapshot_cells =
                use_bbh_split ? binary_hierarchy->level_snapshot_total_cells(level)
                              : hierarchy->level_snapshot_total_cells(level);
            const double estimated_bytes = use_bbh_split ? binary_hierarchy->level_estimated_bytes(level)
                                                         : hierarchy->level_estimated_bytes(level);
            std::cout << "[fmr.mem] level=" << level
                      << " allocated_fields=" << allocated_fields
                      << " total_cells=" << g.total_cells()
                      << " snapshot_fields=" << snapshot_fields
                      << " snapshot_cells=" << snapshot_cells
                      << " estimated_gib=" << bytes_to_gib(estimated_bytes)
                      << std::endl;
        }
        std::cout << "[fmr.mem] total_estimated_gib="
                  << bytes_to_gib(use_bbh_split ? binary_hierarchy->total_estimated_bytes()
                                                : hierarchy->total_estimated_bytes())
                  << std::endl;
    } else {
        std::cout << "[fmr] enabled=0" << std::endl;
    }

    std::cout << "[log] state_log_stride=" << output_stride << std::endl;
    std::cout << "[viz] slice_export_stride=" << slice_export_stride << std::endl;
    std::cout << "[constraints] norms_stride=" << constraint_export_stride
              << " slice_export=" << (export_constraint_slices ? 1 : 0)
              << " slice_stride=" << constraint_slice_stride << std::endl;
    std::cout << "[hdf5] requested=" << (export_hdf5 ? 1 : 0)
              << " available=" << (tensorium::io::hdf5_available() ? 1 : 0) << std::endl;

    (void)std::remove("Output/viz/constraints_norms.csv");
    std::ofstream constraint_log("Output/viz/constraints_norms.csv",
                                 std::ios::out | std::ios::trunc);
    if (constraint_log.is_open()) {
        constraint_log.setf(std::ios::unitbuf);
        constraint_log
            << "step,t,dt,l2_theta,l2_Z,l2_H,l2_M,max_H,max_det_drift,max_trace_A,samples\n";
    } else {
        std::cout << "[warn] could not open Output/viz/constraints_norms.csv for writing"
                  << std::endl;
    }

    (void)std::remove("Output/viz/puncture_track.csv");
    std::ofstream puncture_track("Output/viz/puncture_track.csv", std::ios::out | std::ios::trunc);
    if (puncture_track.is_open()) {
        puncture_track.setf(std::ios::unitbuf);
        puncture_track << "step,t,"
                       << "x_left,y_left,chi_left,alpha_left,"
                       << "x_right,y_right,chi_right,alpha_right\n";
    } else {
        std::cout << "[warn] could not open Output/viz/puncture_track.csv for writing"
                  << std::endl;
    }

    (void)std::remove("Output/viz/puncture_track_minima.csv");
    std::ofstream puncture_track_minima("Output/viz/puncture_track_minima.csv",
                                        std::ios::out | std::ios::trunc);
    if (puncture_track_minima.is_open()) {
        puncture_track_minima.setf(std::ios::unitbuf);
        puncture_track_minima << "step,t,"
                              << "x_left,y_left,chi_left,alpha_left,"
                              << "x_right,y_right,chi_right,alpha_right\n";
    } else {
        std::cout << "[warn] could not open Output/viz/puncture_track_minima.csv for writing"
                  << std::endl;
    }

    tensorium::io::HDF5AppendTable constraint_log_h5;
    tensorium::io::HDF5AppendTable puncture_track_h5;
    tensorium::io::HDF5AppendTable puncture_track_minima_h5;
    if (export_hdf5) {
        if (tensorium::io::hdf5_available()) {
            if (!constraint_log_h5.open(
                    "Output/viz/constraints_norms.h5",
                    {"step", "t", "dt", "l2_theta", "l2_Z", "l2_H", "l2_M", "max_H",
                     "max_det_drift", "max_trace_A", "samples"})) {
                std::cout << "[warn] could not open Output/viz/constraints_norms.h5 for writing"
                          << std::endl;
            }
            if (!puncture_track_h5.open("Output/viz/puncture_track.h5",
                                        {"step", "t", "x_left", "y_left", "chi_left",
                                         "alpha_left", "x_right", "y_right", "chi_right",
                                         "alpha_right"})) {
                std::cout << "[warn] could not open Output/viz/puncture_track.h5 for writing"
                          << std::endl;
            }
            if (!puncture_track_minima_h5.open(
                    "Output/viz/puncture_track_minima.h5",
                    {"step", "t", "x_left", "y_left", "chi_left", "alpha_left", "x_right",
                     "y_right", "chi_right", "alpha_right"})) {
                std::cout << "[warn] could not open Output/viz/puncture_track_minima.h5 for writing"
                          << std::endl;
            }
        } else {
            std::cout << "[warn] HDF5 export requested but this build has no HDF5 support"
                      << std::endl;
        }
    }

    auto write_puncture_row = [&](std::ofstream &file, size_t step, double time,
                                  const PuncturePlaneSample &sample) {
        if (!file.is_open())
            return;
        file << step << "," << time << ",";
        if (sample.has_left) {
            file << sample.x_left << "," << sample.y_left << "," << sample.chi_left << ","
                 << sample.alpha_left << ",";
        } else {
            file << "nan,nan,nan,nan,";
        }
        if (sample.has_right) {
            file << sample.x_right << "," << sample.y_right << "," << sample.chi_right << ","
                 << sample.alpha_right << "\n";
        } else {
            file << "nan,nan,nan,nan\n";
        }
    };
    auto write_puncture_row_hdf5 = [&](tensorium::io::HDF5AppendTable &file, size_t step,
                                       double time, const PuncturePlaneSample &sample) {
        if (!file.is_open())
            return;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::vector<double> row = {
            static_cast<double>(step),
            time,
            sample.has_left ? sample.x_left : nan,
            sample.has_left ? sample.y_left : nan,
            sample.has_left ? sample.chi_left : nan,
            sample.has_left ? sample.alpha_left : nan,
            sample.has_right ? sample.x_right : nan,
            sample.has_right ? sample.y_right : nan,
            sample.has_right ? sample.chi_right : nan,
            sample.has_right ? sample.alpha_right : nan};
        (void)file.append_row(row);
    };

    ShiftPunctureTracker puncture_tracker;
    if (use_fmr && cfg.fmr.move_with_punctures) {
        if (use_bbh_split) {
            initialize_shift_puncture_tracker_split(*root_state, *left_puncture_state,
                                                    *right_puncture_state, puncture_tracker,
                                                    initial_punctures[0], initial_punctures[1]);
        } else {
            const auto initial_minima = sample_puncture_minima(*puncture_state);
            initialize_shift_puncture_tracker(*puncture_state, puncture_tracker, initial_minima);
        }
        std::cout << "[tracker] enabled=" << (puncture_tracker.initialized ? 1 : 0)
                  << " regrid_interval=" << cfg.fmr.regrid_interval
                  << " regrid_threshold_cells=" << cfg.fmr.regrid_threshold_cells
                  << " recenter_on_drift=" << (cfg.fmr.tracker_recenter_on_drift ? 1 : 0)
                  << " recenter_cells=" << cfg.fmr.tracker_recenter_cells
                  << " drift_warn_cells=" << cfg.fmr.tracker_drift_warn_cells << std::endl;
        if (!puncture_tracker.initialized) {
            std::cout << "[tracker][warn] could not initialize puncture tracker from the finest slice minima"
                      << std::endl;
        }
    }

    size_t current_step = 0;
    double current_time = 0.0;
    double current_dt = 0.0;

    tensorium_RG::z4c::CFLControl<double> control;
    control.cfl = cfg.cfl;
    control.gauge_speed = cfg.gauge_speed;

    double t = 0.0;
    if (use_fmr && use_bbh_split) {
        binary_hierarchy->set_gauge_parameters(params);
        binary_hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
        ConstraintScratch constraint_scratch(*root_state);
        const auto elapsed_seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };

        for (size_t n = 0; n < cfg.steps; ++n) {
            const auto wall_start = std::chrono::steady_clock::now();
            const double dt = binary_hierarchy->compute_dt(control);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            const auto evolve_start = std::chrono::steady_clock::now();
            binary_hierarchy->step(dt);
            const auto evolve_stop = std::chrono::steady_clock::now();
            t += dt;
            const double evolve_seconds = elapsed_seconds(evolve_start, evolve_stop);
            if (fmr_profile_stride > 0 && ((n + 1) % fmr_profile_stride) == 0) {
                const auto &perf = binary_hierarchy->last_step_performance();
                for (size_t level = 0; level < perf.size(); ++level) {
                    const auto &stats = perf[level];
                    const double total_seconds =
                        stats.snapshot_seconds + stats.evolve_seconds +
                        stats.child_subcycling_seconds + stats.restriction_seconds;
                    std::cout << "[fmr.perf] step=" << (n + 1)
                              << " level=" << level
                              << " calls=" << stats.calls
                              << " child_substeps=" << stats.child_substeps
                              << " snapshot=" << stats.snapshot_seconds
                              << " evolve=" << stats.evolve_seconds
                              << " children=" << stats.child_subcycling_seconds
                              << " restrict=" << stats.restriction_seconds
                              << " total=" << total_seconds << std::endl;
                }
            }

            PuncturePlaneSample puncture_sample =
                sample_puncture_minima_split(*left_puncture_state, *right_puncture_state);
            PuncturePlaneSample puncture_minima = puncture_sample;

            if (puncture_tracker.initialized) {
                advance_shift_puncture_tracker_split(*root_state, *left_puncture_state,
                                                     *right_puncture_state, puncture_tracker, dt);
                puncture_sample = make_tracker_sample_split(*left_puncture_state,
                                                           *right_puncture_state, puncture_tracker);

                double drift_left = std::numeric_limits<double>::quiet_NaN();
                double drift_right = std::numeric_limits<double>::quiet_NaN();
                const bool have_drift = compute_tracker_drift(puncture_sample, puncture_minima,
                                                              drift_left, drift_right);
                const double finest_spacing =
                    std::max(grid_max_spacing(*left_puncture_state), grid_max_spacing(*right_puncture_state));
                const double drift_warn_radius = cfg.fmr.tracker_drift_warn_cells * finest_spacing;
                const double recenter_radius = cfg.fmr.tracker_recenter_cells * finest_spacing;

                if (cfg.fmr.tracker_recenter_on_drift && have_drift &&
                    (drift_left > recenter_radius || drift_right > recenter_radius)) {
                    puncture_tracker.p1 = {puncture_minima.x_left, puncture_minima.y_left, 0.0};
                    puncture_tracker.p2 = {puncture_minima.x_right, puncture_minima.y_right, 0.0};
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p1);
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p2);
                    refresh_shift_puncture_tracker_beta_split(*left_puncture_state,
                                                              *right_puncture_state,
                                                              puncture_tracker);
                    puncture_sample = make_tracker_sample_split(*left_puncture_state,
                                                               *right_puncture_state,
                                                               puncture_tracker);
                } else if (have_drift &&
                           (drift_left > drift_warn_radius || drift_right > drift_warn_radius) &&
                           (n % std::max<size_t>(output_stride, size_t(1)) == 0)) {
                    std::cout << "[tracker][warn] step=" << n << " drift_left=" << drift_left
                              << " drift_right=" << drift_right
                              << " warn_radius=" << drift_warn_radius << std::endl;
                }

                const bool regrid_due =
                    cfg.fmr.regrid_interval > 0 && ((n + 1) % cfg.fmr.regrid_interval) == 0;
                if (regrid_due) {
                    const auto current_center = grid_center(binary_hierarchy->shared_level_grid(1));
                    const std::array<double, 3> target_center{
                        0.5 * (puncture_tracker.p1[0] + puncture_tracker.p2[0]),
                        0.5 * (puncture_tracker.p1[1] + puncture_tracker.p2[1]),
                        0.5 * (puncture_tracker.p1[2] + puncture_tracker.p2[2])};
                    const auto current_left_center = grid_center(*left_puncture_state);
                    const auto current_right_center = grid_center(*right_puncture_state);
                    const double center_shift = std::hypot(target_center[0] - current_center[0],
                                                           target_center[1] - current_center[1]);
                    const double left_shift = std::hypot(puncture_tracker.p1[0] - current_left_center[0],
                                                         puncture_tracker.p1[1] - current_left_center[1]);
                    const double right_shift =
                        std::hypot(puncture_tracker.p2[0] - current_right_center[0],
                                   puncture_tracker.p2[1] - current_right_center[1]);
                    const double regrid_threshold =
                        cfg.fmr.regrid_threshold_cells *
                        std::max(left_puncture_state->dx, right_puncture_state->dx);
                    if (center_shift > regrid_threshold || left_shift > regrid_threshold ||
                        right_shift > regrid_threshold) {
                        auto next_hierarchy = rebuild_moving_puncture_binary_hierarchy(
                            binary_hierarchy.get(), cfg, proj_cfg, puncture_tracker.p1,
                            puncture_tracker.p2);
                        next_hierarchy->set_gauge_parameters(params);
                        next_hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
                        binary_hierarchy = std::move(next_hierarchy);
                        root_state = &binary_hierarchy->root_grid();
                        left_puncture_state = &binary_hierarchy->leaf_grid(0);
                        right_puncture_state = &binary_hierarchy->leaf_grid(1);
                        puncture_state = left_puncture_state;
                        refresh_shift_puncture_tracker_beta_split(*left_puncture_state,
                                                                  *right_puncture_state,
                                                                  puncture_tracker);
                        puncture_minima =
                            sample_puncture_minima_split(*left_puncture_state, *right_puncture_state);
                        puncture_sample = make_tracker_sample_split(*left_puncture_state,
                                                                   *right_puncture_state,
                                                                   puncture_tracker);
                        std::cout << "[fmr.regrid] step=" << (n + 1)
                                  << " center_old=(" << current_center[0] << ", " << current_center[1]
                                  << ", " << current_center[2] << ")"
                                  << " center_new=(" << target_center[0] << ", " << target_center[1]
                                  << ", " << target_center[2] << ")"
                                  << " shift=" << center_shift
                                  << " left_shift=" << left_shift
                                  << " right_shift=" << right_shift << std::endl;
                    }
                }
            }

            double constraint_seconds = 0.0;
            const bool need_constraint_export =
                constraint_export_stride > 0 && constraint_log.is_open() &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(*root_state, constraint_scratch, cfg.padding);
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                constraint_log << current_step << "," << current_time << "," << current_dt << ","
                               << stats.l2_theta << "," << stats.l2_Z << "," << stats.l2_H << ","
                               << stats.l2_M << "," << stats.max_H << "," << stats.max_det_drift
                               << "," << stats.max_trace_A << "," << stats.samples << "\n";
                if (constraint_log_h5.is_open()) {
                    (void)constraint_log_h5.append_row(
                        {static_cast<double>(current_step), current_time, current_dt,
                         stats.l2_theta, stats.l2_Z, stats.l2_H, stats.l2_M, stats.max_H,
                         stats.max_det_drift, stats.max_trace_A, static_cast<double>(stats.samples)});
                }
            }

            double export_seconds = 0.0;
            if (slice_export_stride > 0 && (n % slice_export_stride) == 0) {
                std::cout << ">> Exporting domain slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                export_slice_csv(*root_state, binary_hierarchy.get(), n, "Output/viz");
                if (export_hdf5)
                    (void)export_slice_hdf5(*root_state, binary_hierarchy.get(), n, t, "Output/viz");
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                export_constraint_slice_csv(*root_state, constraint_scratch.H, constraint_scratch.M,
                                            constraint_scratch.C, n, "Output/viz");
                if (export_hdf5) {
                    (void)export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                       constraint_scratch.M, constraint_scratch.C,
                                                       n, t, "Output/viz");
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }

            write_puncture_row(puncture_track, n, t, puncture_sample);
            write_puncture_row(puncture_track_minima, n, t, puncture_minima);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_minima);

            double projection_seconds = 0.0;
            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0)) {
                const auto projection_start = std::chrono::steady_clock::now();
                project_binary_hierarchy_levels(*binary_hierarchy, proj_cfg);
                const auto projection_stop = std::chrono::steady_clock::now();
                projection_seconds = elapsed_seconds(projection_start, projection_stop);
            }

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            std::printf(
                "dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds, constraint_seconds,
                export_seconds, projection_seconds);
        }
    } else if (use_fmr) {
        hierarchy->set_gauge_parameters(params);
        hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
        ConstraintScratch constraint_scratch(*root_state);
        const auto elapsed_seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };

        for (size_t n = 0; n < cfg.steps; ++n) {
            const auto wall_start = std::chrono::steady_clock::now();
            const double dt = hierarchy->compute_dt(control);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            const auto evolve_start = std::chrono::steady_clock::now();
            hierarchy->step(dt);
            const auto evolve_stop = std::chrono::steady_clock::now();
            t += dt;
            const double evolve_seconds = elapsed_seconds(evolve_start, evolve_stop);
            if (fmr_profile_stride > 0 && ((n + 1) % fmr_profile_stride) == 0) {
                const auto &perf = hierarchy->last_step_performance();
                for (size_t level = 0; level < perf.size(); ++level) {
                    const auto &stats = perf[level];
                    const double total_seconds =
                        stats.snapshot_seconds + stats.evolve_seconds +
                        stats.child_subcycling_seconds + stats.restriction_seconds;
                    std::cout << "[fmr.perf] step=" << (n + 1)
                              << " level=" << level
                              << " calls=" << stats.calls
                              << " child_substeps=" << stats.child_substeps
                              << " snapshot=" << stats.snapshot_seconds
                              << " evolve=" << stats.evolve_seconds
                              << " children=" << stats.child_subcycling_seconds
                              << " restrict=" << stats.restriction_seconds
                              << " total=" << total_seconds << std::endl;
                }
            }

            PuncturePlaneSample puncture_sample = sample_puncture_minima(*puncture_state);
            PuncturePlaneSample puncture_minima = puncture_sample;

            if (puncture_tracker.initialized) {
                advance_shift_puncture_tracker(*puncture_state, puncture_tracker, dt);
                puncture_sample = make_tracker_sample(*puncture_state, puncture_tracker);

                double drift_left = std::numeric_limits<double>::quiet_NaN();
                double drift_right = std::numeric_limits<double>::quiet_NaN();
                const bool have_drift = compute_tracker_drift(puncture_sample, puncture_minima,
                                                              drift_left, drift_right);
                const double drift_warn_radius =
                    cfg.fmr.tracker_drift_warn_cells * grid_max_spacing(*puncture_state);
                const double recenter_radius =
                    cfg.fmr.tracker_recenter_cells * grid_max_spacing(*puncture_state);

                if (cfg.fmr.tracker_recenter_on_drift && have_drift &&
                    (drift_left > recenter_radius || drift_right > recenter_radius)) {
                    puncture_tracker.p1 = {puncture_minima.x_left, puncture_minima.y_left, 0.0};
                    puncture_tracker.p2 = {puncture_minima.x_right, puncture_minima.y_right, 0.0};
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p1);
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p2);
                    refresh_shift_puncture_tracker_beta(*puncture_state, puncture_tracker);
                    puncture_sample = make_tracker_sample(*puncture_state, puncture_tracker);
                } else if (have_drift &&
                           (drift_left > drift_warn_radius || drift_right > drift_warn_radius) &&
                           (n % std::max<size_t>(output_stride, size_t(1)) == 0)) {
                    std::cout << "[tracker][warn] step=" << n << " drift_left=" << drift_left
                              << " drift_right=" << drift_right
                              << " warn_radius=" << drift_warn_radius << std::endl;
                }

                const bool regrid_due =
                    cfg.fmr.regrid_interval > 0 && ((n + 1) % cfg.fmr.regrid_interval) == 0;
                if (regrid_due) {
                    const auto current_center = grid_center(hierarchy->level_grid(1));
                    const std::array<double, 3> target_center{
                        0.5 * (puncture_tracker.p1[0] + puncture_tracker.p2[0]),
                        0.5 * (puncture_tracker.p1[1] + puncture_tracker.p2[1]),
                        0.5 * (puncture_tracker.p1[2] + puncture_tracker.p2[2])};
                    const double center_shift = std::hypot(target_center[0] - current_center[0],
                                                           target_center[1] - current_center[1]);
                    const double regrid_threshold =
                        cfg.fmr.regrid_threshold_cells * hierarchy->level_grid(hierarchy->num_levels() - 1).dx;
                    if (center_shift > regrid_threshold) {
                        auto next_hierarchy =
                            rebuild_moving_puncture_hierarchy(hierarchy.get(), cfg, proj_cfg, target_center);
                        next_hierarchy->set_gauge_parameters(params);
                        next_hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
                        hierarchy = std::move(next_hierarchy);
                        root_state = &hierarchy->root_grid();
                        puncture_state = &hierarchy->level_grid(hierarchy->num_levels() - 1);
                        refresh_shift_puncture_tracker_beta(*puncture_state, puncture_tracker);
                        puncture_minima = sample_puncture_minima(*puncture_state);
                        puncture_sample = make_tracker_sample(*puncture_state, puncture_tracker);
                        std::cout << "[fmr.regrid] step=" << (n + 1)
                                  << " center_old=(" << current_center[0] << ", " << current_center[1]
                                  << ", " << current_center[2] << ")"
                                  << " center_new=(" << target_center[0] << ", " << target_center[1]
                                  << ", " << target_center[2] << ")"
                                  << " shift=" << center_shift << std::endl;
                    }
                }
            }

            double constraint_seconds = 0.0;
            const bool need_constraint_export =
                constraint_export_stride > 0 && constraint_log.is_open() &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(*root_state, constraint_scratch, cfg.padding);
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                constraint_log << current_step << "," << current_time << "," << current_dt << ","
                               << stats.l2_theta << "," << stats.l2_Z << "," << stats.l2_H << ","
                               << stats.l2_M << "," << stats.max_H << "," << stats.max_det_drift
                               << "," << stats.max_trace_A << "," << stats.samples << "\n";
                if (constraint_log_h5.is_open()) {
                    (void)constraint_log_h5.append_row(
                        {static_cast<double>(current_step), current_time, current_dt,
                         stats.l2_theta, stats.l2_Z, stats.l2_H, stats.l2_M, stats.max_H,
                         stats.max_det_drift, stats.max_trace_A, static_cast<double>(stats.samples)});
                }
            }

            double export_seconds = 0.0;
            if (slice_export_stride > 0 && (n % slice_export_stride) == 0) {
                std::cout << ">> Exporting domain slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                export_slice_csv(*root_state, hierarchy.get(), n, "Output/viz");
                if (export_hdf5)
                    (void)export_slice_hdf5(*root_state, hierarchy.get(), n, t, "Output/viz");
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                export_constraint_slice_csv(*root_state, constraint_scratch.H, constraint_scratch.M,
                                            constraint_scratch.C, n, "Output/viz");
                if (export_hdf5) {
                    (void)export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                       constraint_scratch.M, constraint_scratch.C,
                                                       n, t, "Output/viz");
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }

            write_puncture_row(puncture_track, n, t, puncture_sample);
            write_puncture_row(puncture_track_minima, n, t, puncture_minima);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_minima);

            double projection_seconds = 0.0;
            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
            {
                const auto projection_start = std::chrono::steady_clock::now();
                project_hierarchy_levels(*hierarchy, proj_cfg);
                const auto projection_stop = std::chrono::steady_clock::now();
                projection_seconds = elapsed_seconds(projection_start, projection_stop);
            }

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                        dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds,
                        constraint_seconds, export_seconds, projection_seconds);
        }
    } else {
        tensorium_RG::z4c::Z4cRKStepper<double, tensorium_RG::z4c::BoundaryRadiative> stepper(
            *root_state, cfg.padding);
        stepper.set_gauge_parameters(params);
        stepper.set_state_log_stride(output_stride);
        ConstraintScratch constraint_scratch(*root_state);
        const auto elapsed_seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };

        for (size_t n = 0; n < cfg.steps; ++n) {
            const auto wall_start = std::chrono::steady_clock::now();
            const double dt = tensorium_RG::z4c::compute_dt_cfl(*root_state, control, cfg.padding);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            const auto evolve_start = std::chrono::steady_clock::now();
            stepper.step(*root_state, dt, n);
            const auto evolve_stop = std::chrono::steady_clock::now();
            t += dt;

            const auto puncture_sample = sample_puncture_minima(*root_state);
            write_puncture_row(puncture_track, n, t, puncture_sample);
            write_puncture_row(puncture_track_minima, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_sample);

            double constraint_seconds = 0.0;
            const bool need_constraint_export =
                constraint_export_stride > 0 && constraint_log.is_open() &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(*root_state, constraint_scratch, cfg.padding);
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                constraint_log << current_step << "," << current_time << "," << current_dt << ","
                               << stats.l2_theta << "," << stats.l2_Z << "," << stats.l2_H << ","
                               << stats.l2_M << "," << stats.max_H << "," << stats.max_det_drift
                               << "," << stats.max_trace_A << "," << stats.samples << "\n";
                if (constraint_log_h5.is_open()) {
                    (void)constraint_log_h5.append_row(
                        {static_cast<double>(current_step), current_time, current_dt,
                         stats.l2_theta, stats.l2_Z, stats.l2_H, stats.l2_M, stats.max_H,
                         stats.max_det_drift, stats.max_trace_A, static_cast<double>(stats.samples)});
                }
            }

            double export_seconds = 0.0;
            if (slice_export_stride > 0 && (n % slice_export_stride) == 0) {
                std::cout << ">> Exporting slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                export_slice_csv<MovingPunctureHierarchy>(*root_state, nullptr, n, "Output/viz");
                if (export_hdf5)
                    (void)export_slice_hdf5<MovingPunctureHierarchy>(*root_state, nullptr, n, t,
                                                                     "Output/viz");
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                export_constraint_slice_csv(*root_state, constraint_scratch.H, constraint_scratch.M,
                                            constraint_scratch.C, n, "Output/viz");
                if (export_hdf5) {
                    (void)export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                       constraint_scratch.M, constraint_scratch.C,
                                                       n, t, "Output/viz");
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }

            double projection_seconds = 0.0;
            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
            {
                const auto projection_start = std::chrono::steady_clock::now();
                tensorium_RG::z4c::project_z4c_state(*root_state, proj_cfg);
                const auto projection_stop = std::chrono::steady_clock::now();
                projection_seconds = elapsed_seconds(projection_start, projection_stop);
            }

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            const double evolve_seconds = elapsed_seconds(evolve_start, evolve_stop);
            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                        dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds,
                        constraint_seconds, export_seconds, projection_seconds);
        }
    }

    std::cout << "[done] steps=" << cfg.steps << " t_final=" << t << std::endl;
    return 0;
}
