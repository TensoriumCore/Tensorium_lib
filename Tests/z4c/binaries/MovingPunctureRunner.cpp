#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintMonitoring.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Constraints/BSSNConstraintsGrid.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/MovingPunctureEnv.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"
#include "../../../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/FMR/FMRMemoryManager.hpp"
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
#include <string>

namespace {

using Grid = tensorium_RG::Z4cGridSoA<double>;
using MovingPunctureHierarchy =
    tensorium_RG::z4c::fmr::FixedMeshRefinementHierarchy<double, tensorium_RG::z4c::BoundaryRadiative>;
using BinaryMovingPunctureHierarchy =
    tensorium_RG::z4c::fmr::BinaryPunctureFixedMeshRefinementHierarchy<double,
                                                                       tensorium_RG::z4c::BoundaryRadiative>;
using FineBoundary = MovingPunctureHierarchy::FineBoundary;

bool point_in_grid_physical_interior(const Grid &grid, double x, double y, double z,
                                     size_t guard_cells);

template <typename HierarchyType>
bool sample_composite_scalar(const Grid &root_grid, const HierarchyType *hierarchy,
                             tensorium_RG::z4c::BoundaryField which, int component, double x,
                             double y, double z, double &value, size_t guard_cells = 0) {
    const auto sample_from_grid = [&](const Grid &grid) {
        if (!point_in_grid_physical_interior(grid, x, y, z, guard_cells))
            return false;
        const auto &field = tensorium_RG::z4c::fmr::detail::select_field(grid, which, component);
        value = tensorium_RG::z4c::fmr::detail::sample_trilinear_field(grid, field, x, y, z);
        return std::isfinite(value);
    };

    if (hierarchy != nullptr) {
        for (size_t level = hierarchy->num_levels(); level-- > 0;) {
            if (sample_from_grid(hierarchy->level_grid(level)))
                return true;
        }
    }
    return sample_from_grid(root_grid);
}

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
            double alpha = std::numeric_limits<double>::quiet_NaN();
            double chi = std::numeric_limits<double>::quiet_NaN();
            if (!sample_composite_scalar(grid, hierarchy, tensorium_RG::z4c::BoundaryField::Alpha,
                                         0, x, y, grid.z0 + double(grid.dims.nz / 2) * grid.dz,
                                         alpha)) {
                alpha = grid.alpha.ptr()[grid.alpha.idx(i, j, k)];
            }
            if (!sample_composite_scalar(grid, hierarchy, tensorium_RG::z4c::BoundaryField::Chi, 0,
                                         x, y, grid.z0 + double(grid.dims.nz / 2) * grid.dz, chi)) {
                chi = grid.chi.ptr()[grid.chi.idx(i, j, k)];
            }
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
                                 const std::string &output_dir,
                                 size_t constraint_guard_cells) {
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
    const size_t constraint_guard =
        std::min(constraint_guard_cells, std::min(nx, ny) / size_t(2));

    for (size_t i = ng; i < nx + ng; ++i) {
        for (size_t j = ng; j < ny + ng; ++j) {
            const double x = grid.x0 + (double(i) - double(ng)) * grid.dx;
            const double y = grid.y0 + (double(j) - double(ng)) * grid.dy;

            const size_t idx = grid.alpha.idx(i, j, k);
            const size_t cidx = H.idx(i, j, k);
            const bool constraint_valid =
                (i >= ng + constraint_guard && i + constraint_guard < nx + ng &&
                 j >= ng + constraint_guard && j + constraint_guard < ny + ng);

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
            const size_t flat = i * ny + j;
            const double x_ij = x[i];
            const double y_ij = y[j];
            if (!sample_composite_scalar(grid, hierarchy, tensorium_RG::z4c::BoundaryField::Alpha,
                                         0, x_ij, y_ij, grid.z0 + double(grid.dims.nz / 2) * grid.dz,
                                         alpha[flat])) {
                alpha[flat] = grid.alpha.ptr()[grid.alpha.idx(ii, jj, k)];
            }
            if (!sample_composite_scalar(grid, hierarchy, tensorium_RG::z4c::BoundaryField::Chi, 0,
                                         x_ij, y_ij, grid.z0 + double(grid.dims.nz / 2) * grid.dz,
                                         chi[flat])) {
                chi[flat] = grid.chi.ptr()[grid.chi.idx(ii, jj, k)];
            }
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
                                  double time, const std::string &output_dir,
                                  size_t constraint_guard_cells) {
    if (!tensorium::io::hdf5_available())
        return false;

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t ng = grid.dims.ng;
    const size_t k = ng + grid.dims.nz / 2;
    const size_t constraint_guard =
        std::min(constraint_guard_cells, std::min(nx, ny) / size_t(2));

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
                (ii >= ng + constraint_guard && ii + constraint_guard < nx + ng &&
                 jj >= ng + constraint_guard && jj + constraint_guard < ny + ng);

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
    (void)tensorium::io::write_scalar_attribute(file.id(), "constraint_guard",
                                                static_cast<std::uint64_t>(constraint_guard));
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

bool write_volume_xdmf(const std::string &xmf_path, const std::string &h5_name, size_t nx,
                       size_t ny, size_t nz, double time) {
    std::ofstream file(xmf_path, std::ios::out | std::ios::trunc);
    if (!file.is_open())
        return false;

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\" ?>\n";
    file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    file << "<Xdmf Version=\"3.0\">\n";
    file << "  <Domain>\n";
    file << "    <Grid Name=\"moving_puncture_volume\" GridType=\"Uniform\">\n";
    file << "      <Time Value=\"" << time << "\" />\n";
    file << "      <Topology TopologyType=\"3DRectMesh\" Dimensions=\"" << (nz + 1) << " "
         << (ny + 1) << " " << (nx + 1) << "\"/>\n";
    file << "      <Geometry GeometryType=\"VXVYVZ\">\n";
    file << "        <DataItem Dimensions=\"" << (nx + 1)
         << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << h5_name
         << ":/x_nodes</DataItem>\n";
    file << "        <DataItem Dimensions=\"" << (ny + 1)
         << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << h5_name
         << ":/y_nodes</DataItem>\n";
    file << "        <DataItem Dimensions=\"" << (nz + 1)
         << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << h5_name
         << ":/z_nodes</DataItem>\n";
    file << "      </Geometry>\n";

    const auto write_attr = [&](const char *name) {
        file << "      <Attribute Name=\"" << name
             << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        file << "        <DataItem Dimensions=\"" << nz << " " << ny << " " << nx
             << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" << h5_name << ":/"
             << name << "</DataItem>\n";
        file << "      </Attribute>\n";
    };

    write_attr("alpha");
    write_attr("chi");
    write_attr("W");
    write_attr("mask");

    file << "    </Grid>\n";
    file << "  </Domain>\n";
    file << "</Xdmf>\n";
    return true;
}

bool export_volume_hdf5_xdmf(const Grid &grid, size_t step, double time,
                             const std::string &output_dir) {
    if (!tensorium::io::hdf5_available())
        return false;

    const size_t nx = grid.dims.nx;
    const size_t ny = grid.dims.ny;
    const size_t nz = grid.dims.nz;
    const size_t ng = grid.dims.ng;

    std::vector<double> x_nodes(nx + 1);
    std::vector<double> y_nodes(ny + 1);
    std::vector<double> z_nodes(nz + 1);
    std::vector<double> alpha(nx * ny * nz);
    std::vector<double> chi(nx * ny * nz);
    std::vector<double> W(nx * ny * nz);
    std::vector<double> mask(nx * ny * nz);

    for (size_t i = 0; i <= nx; ++i)
        x_nodes[i] = grid.x0 - 0.5 * grid.dx + double(i) * grid.dx;
    for (size_t j = 0; j <= ny; ++j)
        y_nodes[j] = grid.y0 - 0.5 * grid.dy + double(j) * grid.dy;
    for (size_t k = 0; k <= nz; ++k)
        z_nodes[k] = grid.z0 - 0.5 * grid.dz + double(k) * grid.dz;

    for (size_t k = 0; k < nz; ++k) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                const size_t ii = ng + i;
                const size_t jj = ng + j;
                const size_t kk = ng + k;
                const size_t idx = grid.alpha.idx(ii, jj, kk);
                const size_t flat = (k * ny + j) * nx + i;
                alpha[flat] = grid.alpha.ptr()[idx];
                chi[flat] = grid.chi.ptr()[idx];
                W[flat] = std::sqrt(std::max(chi[flat], 0.0));
                mask[flat] = (alpha[flat] < 0.1) ? 1.0 : 0.0;
            }
        }
    }

    std::stringstream h5_ss;
    h5_ss << output_dir << "/volume_" << std::setw(4) << std::setfill('0') << step << ".h5";
    tensorium::io::HDF5File h5_file(h5_ss.str());
    if (!h5_file.is_open())
        return false;

    (void)tensorium::io::write_string_attribute(h5_file.id(), "tensorium_kind",
                                                "moving_puncture_volume_v1");
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "step",
                                                static_cast<std::uint64_t>(step));
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "time", time);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "nx", static_cast<std::uint64_t>(nx));
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "ny", static_cast<std::uint64_t>(ny));
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "nz", static_cast<std::uint64_t>(nz));
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "dx", grid.dx);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "dy", grid.dy);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "dz", grid.dz);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "x0", grid.x0);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "y0", grid.y0);
    (void)tensorium::io::write_scalar_attribute(h5_file.id(), "z0", grid.z0);

    const bool ok = tensorium::io::write_vector_dataset(h5_file.id(), "x_nodes", x_nodes) &&
                    tensorium::io::write_vector_dataset(h5_file.id(), "y_nodes", y_nodes) &&
                    tensorium::io::write_vector_dataset(h5_file.id(), "z_nodes", z_nodes) &&
                    tensorium::io::write_tensor3_dataset(h5_file.id(), "alpha", nz, ny, nx, alpha) &&
                    tensorium::io::write_tensor3_dataset(h5_file.id(), "chi", nz, ny, nx, chi) &&
                    tensorium::io::write_tensor3_dataset(h5_file.id(), "W", nz, ny, nx, W) &&
                    tensorium::io::write_tensor3_dataset(h5_file.id(), "mask", nz, ny, nx, mask);
    if (!ok)
        return false;

    std::filesystem::path h5_path(h5_ss.str());
    std::stringstream xmf_ss;
    xmf_ss << output_dir << "/volume_" << std::setw(4) << std::setfill('0') << step << ".xmf";
    return write_volume_xdmf(xmf_ss.str(), h5_path.filename().string(), nx, ny, nz, time);
}

size_t parse_env_stride_or(const char *name, size_t fallback) {
    if (const char *raw = std::getenv(name)) {
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && parsed >= 0)
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

PunctureMinimumSample sample_grid_minimum_alpha_near(const Grid &grid,
                                                     const std::array<double, 3> &center,
                                                     size_t radius_cells) {
    PunctureMinimumSample sample;
    const size_t          nx = grid.dims.nx;
    const size_t          ny = grid.dims.ny;
    const size_t          ng = grid.dims.ng;
    const size_t          k = ng + grid.dims.nz / 2;

    const auto coord_to_physical_index = [&](double coord, double origin, double spacing,
                                             size_t n) -> size_t {
        if (n == 0 || !(std::isfinite(coord) && std::isfinite(origin) && spacing > 0.0))
            return size_t(0);
        const double rel = std::round((coord - origin) / spacing);
        const double clamped = std::clamp(rel, 0.0, double(n - 1));
        return static_cast<size_t>(clamped);
    };

    const size_t ic = coord_to_physical_index(center[0], double(grid.x0), double(grid.dx), nx);
    const size_t jc = coord_to_physical_index(center[1], double(grid.y0), double(grid.dy), ny);
    const size_t r = std::max<size_t>(radius_cells, size_t(1));
    const size_t i_begin = ng + ((ic > r) ? (ic - r) : size_t(0));
    const size_t j_begin = ng + ((jc > r) ? (jc - r) : size_t(0));
    const size_t i_end = ng + std::min(nx, ic + r + size_t(1));
    const size_t j_end = ng + std::min(ny, jc + r + size_t(1));

    for (size_t i = i_begin; i < i_end; ++i) {
        for (size_t j = j_begin; j < j_end; ++j) {
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

PuncturePlaneSample sample_puncture_minima_near_tracker(const Grid &grid,
                                                        const ShiftPunctureTracker &tracker,
                                                        size_t radius_cells) {
    PuncturePlaneSample sample;
    if (!tracker.initialized)
        return sample;

    const auto left = sample_grid_minimum_alpha_near(grid, tracker.p1, radius_cells);
    const auto right = sample_grid_minimum_alpha_near(grid, tracker.p2, radius_cells);
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

double moving_puncture_grid_half_width(const Grid &grid,
                                       const std::array<double, 3> &center = {0.0, 0.0, 0.0}) {
    const auto axis_half_width = [](double origin, double spacing, size_t cells, double center_coord) {
        const double xmin = origin - 0.5 * spacing;
        const double xmax = origin + (double(cells) - 0.5) * spacing;
        return std::min(std::abs(center_coord - xmin), std::abs(xmax - center_coord));
    };

    return std::min({axis_half_width(grid.x0, grid.dx, grid.dims.nx, center[0]),
                     axis_half_width(grid.y0, grid.dy, grid.dims.ny, center[1]),
                     axis_half_width(grid.z0, grid.dz, grid.dims.nz, center[2])});
}

struct MovingPunctureInitBlendRegion {
    double core_half_width = 0.0;
    double transition_width = 0.0;
    double tp_outer_half_width = 0.0;
};

MovingPunctureInitBlendRegion
moving_puncture_init_blend_region(const Grid &grid,
                                  const tensorium_RG::z4c::MovingPunctureEnvConfig &cfg,
                                  const std::array<double, 3> &center = {0.0, 0.0, 0.0}) {
    const double max_half_width =
        std::max(0.0, moving_puncture_grid_half_width(grid, center) -
                          0.5 * std::max({grid.dx, grid.dy, grid.dz}));
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
    const auto initial_punctures = tensorium_RG::z4c::moving_puncture_initial_puncture_positions(cfg);

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
        const auto &leaf_center = initial_punctures[leaf];
        const auto region = moving_puncture_init_blend_region(hierarchy.leaf_grid(leaf), cfg,
                                                              leaf_center);
        hierarchy.blend_leaf_core_from_reference(leaf, tp_reference, leaf_center,
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

bool is_recoverable_bbh_split_regrid_geometry_error(const std::exception &ex) {
    const std::string message = ex.what();
    return message.find("finest leaves overlap") != std::string::npos ||
           message.find("leaf patches overlap") != std::string::npos ||
           message.find("too small to contain both puncture leaves") != std::string::npos ||
           message.find("cannot be separated") != std::string::npos ||
           message.find("cannot retain ordered split boxes") != std::string::npos ||
           message.find("adaptive clipping would exclude a puncture") != std::string::npos ||
           message.find("does not fit on the grid") != std::string::npos ||
           message.find("exceed the root domain") != std::string::npos;
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
    auto cfg = tensorium_RG::z4c::load_moving_puncture_env();
    const size_t production_steps = cfg.steps;
    size_t base_output_stride = cfg.state_log_stride;
    size_t base_slice_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_SLICE_EXPORT_STRIDE", 10);
    size_t base_constraint_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_EXPORT_STRIDE", 5);
    bool base_export_constraint_slices =
        parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EXPORT_CONSTRAINT_SLICES", true);
    const bool base_request_hdf5_export =
        parse_env_bool_or("TENSORIUM_MOVING_PUNCTURE_EXPORT_HDF5", false);
    const bool base_export_hdf5 =
        base_request_hdf5_export && tensorium::io::hdf5_available();
    const bool base_export_csv = !base_export_hdf5;
    size_t base_volume_export_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_VOLUME_EXPORT_STRIDE", 0);
    size_t base_constraint_slice_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_SLICE_STRIDE",
                            base_constraint_export_stride);
    size_t base_constraint_guard_cells =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_CONSTRAINT_GUARD_CELLS", 4);
    size_t base_fmr_profile_stride =
        parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_FMR_PROFILE_STRIDE", 0);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output-stride") == 0 && i + 1 < argc) {
            base_output_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--slice-export-stride") == 0 && i + 1 < argc) {
            base_slice_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--constraint-export-stride") == 0 && i + 1 < argc) {
            base_constraint_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--constraint-slice-stride") == 0 && i + 1 < argc) {
            base_constraint_slice_stride = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--volume-export-stride") == 0 && i + 1 < argc) {
            base_volume_export_stride = static_cast<size_t>(std::stoul(argv[++i]));
        }
    }

    const bool ecc_tuning_enabled = cfg.ecc_control.enabled && cfg.ecc_control.iterations > 0;
    if (ecc_tuning_enabled && cfg.auto_circular) {
        std::cout << "[ecc] AUTO_CIRCULAR provided only the initial guess; eccentricity control "
                     "takes ownership of momentum updates."
                  << std::endl;
        cfg.auto_circular = false;
        cfg.user_tangential_momentum = cfg.tangential_momentum;
    }

    const size_t total_passes =
        ecc_tuning_enabled ? (cfg.ecc_control.iterations + (cfg.ecc_control.tune_only ? 0u : 1u))
                           : 1u;

    for (size_t pass = 0; pass < total_passes; ++pass) {
        const bool ecc_tuning_pass = ecc_tuning_enabled && pass < cfg.ecc_control.iterations;
        const bool suppress_trial_exports = ecc_tuning_pass && !cfg.ecc_control.export_trials;
        cfg.steps = ecc_tuning_pass ? cfg.ecc_control.trial_steps : production_steps;

        size_t output_stride = base_output_stride;
        size_t slice_export_stride = base_slice_export_stride;
        size_t constraint_export_stride = base_constraint_export_stride;
        bool export_constraint_slices = base_export_constraint_slices;
        const bool request_hdf5_export =
            suppress_trial_exports ? false : base_request_hdf5_export;
        bool export_hdf5 = suppress_trial_exports ? false : base_export_hdf5;
        bool export_csv = suppress_trial_exports ? false : base_export_csv;
        size_t volume_export_stride = suppress_trial_exports ? 0 : base_volume_export_stride;
        size_t constraint_slice_stride =
            suppress_trial_exports ? 0 : base_constraint_slice_stride;
        size_t constraint_guard_cells = base_constraint_guard_cells;
        size_t fmr_profile_stride = suppress_trial_exports ? 0 : base_fmr_profile_stride;

        if (suppress_trial_exports) {
            output_stride = std::max<size_t>(cfg.steps + 1, size_t(1));
            slice_export_stride = 0;
            constraint_export_stride = 0;
            export_constraint_slices = false;
            volume_export_stride = 0;
            constraint_slice_stride = 0;
            std::cout << "[ecc] iteration " << (pass + 1) << "/" << cfg.ecc_control.iterations
                      << " trial_steps=" << cfg.steps
                      << " p_tang=" << cfg.tangential_momentum
                      << " p_rad=" << cfg.radial_momentum << std::endl;
        } else if (ecc_tuning_pass) {
            std::cout << "[ecc] iteration " << (pass + 1) << "/" << cfg.ecc_control.iterations
                      << " trial_steps=" << cfg.steps
                      << " p_tang=" << cfg.tangential_momentum
                      << " p_rad=" << cfg.radial_momentum
                      << " export_trials=1" << std::endl;
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
    const size_t effective_radiative_collar =
        cfg.gauge_params.apply_rhs_sommerfeld ? std::max<size_t>(cfg.radiative_collar_width, 4)
                                              : size_t(0);
    std::cout << "[gauge] apply_rhs_sommerfeld="
              << (cfg.gauge_params.apply_rhs_sommerfeld ? 1 : 0)
              << " rhs_collar_width=" << cfg.radiative_collar_width
              << " effective_rhs_collar=" << effective_radiative_collar << std::endl;
    std::cout << "[gauge] strict_tp_gauge=" << (cfg.strict_tp_gauge ? 1 : 0)
              << " shift_eta=" << cfg.gauge_params.shift_eta
              << " eta=" << cfg.gauge_params.eta
              << " beta_B_coeff=" << cfg.gauge_params.beta_B_coeff
              << " use_shift_advection=" << (cfg.gauge_params.use_shift_advection ? 1 : 0)
              << " shift_advect=" << cfg.gauge_params.shift_advect
              << " lapse_advect=" << cfg.gauge_params.lapse_advect
              << " ko_sigma=" << cfg.gauge_params.ko_sigma
              << " kappa1=" << cfg.gauge_params.kappa1
              << " kappa2=" << cfg.gauge_params.kappa2
              << " kappa3=" << cfg.gauge_params.kappa3
              << " kappa_z=" << cfg.gauge_params.kappa_z
              << " christoffel_lapse_damping=" << cfg.gauge_params.christoffel_lapse_damping
              << " metric_christoffel_damping=" << cfg.gauge_params.metric_christoffel_damping
              << " gamma_driver_filtered_rhs="
              << (cfg.gauge_params.gamma_driver_uses_filtered_gamma_rhs ? 1 : 0)
              << " slow_start_lapse=" << (cfg.gauge_params.slow_start_lapse_article ? 1 : 0)
              << " slow_start_h=" << cfg.gauge_params.slow_start_lapse_h
              << " slow_start_sigma=" << cfg.gauge_params.slow_start_lapse_sigma
              << " covariant_z4=" << (cfg.gauge_params.covariant_z4 ? 1 : 0)
              << " evolve_Z=" << (cfg.gauge_params.evolve_Z ? 1 : 0) << std::endl;
    if (cfg.use_interpolated_init && cfg.strict_tp_gauge) {
        std::cout << "[gauge][warn] STRICT_TP_GAUGE is active for interpolated init; it "
                     "overrides SHIFT_ETA, KO_SIGMA, SHIFT_GAMMA, and shift-advection defaults. "
                     "Set TENSORIUM_MOVING_PUNCTURE_STRICT_TP_GAUGE=0 for NR102-like runs."
                  << std::endl;
    }
    if (cfg.gauge_params.apply_rhs_sommerfeld && cfg.radiative_collar_width < 4) {
        std::cout << "[bc][warn] RADIATIVE_COLLAR_WIDTH < 4; using effective_rhs_collar=4 "
                     "to keep high-order RHS stencils off extrapolated ghosts."
                  << std::endl;
    }
    if (cfg.use_interpolated_init) {
        std::cout << "[init] twopunctures_mass_mode="
                  << (cfg.tp_calculate_target_masses ? "target" : "bare")
                  << " adm_tol=" << cfg.tp_adm_tol << std::endl;
        if (cfg.auto_circular && !cfg.tp_calculate_target_masses) {
            std::cout << "[warn] AUTO_CIRCULAR with fixed bare masses is not the GRChombo "
                         "TwoPunctures setup; when separation changes it can generate expanding "
                         "trajectories. For GRChombo-like TP data, enable "
                         "TENSORIUM_MOVING_PUNCTURE_TP_CALCULATE_TARGET_MASSES=1 and pass target "
                         "masses via MASS/MASS1/MASS2."
                      << std::endl;
        }
    }
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
              << " ko_edge_corner_boost=" << cfg.ko_edge_corner_boost
              << " cako=" << (cfg.enable_cako ? 1 : 0)
              << " ko_curvature_floor=" << cfg.ko_curvature_floor << std::endl;
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

    // Initialize FMR memory pool only when refinement levels are actually active.
    tensorium_RG::z4c::fmr::FMRMemoryManager<double> memory_manager;
    if (use_fmr) {
        tensorium_RG::z4c::fmr::FMRMemoryConfig pool_config;
        pool_config.root_nx = cfg.nx;
        pool_config.root_ny = cfg.ny;
        pool_config.root_nz = cfg.nz;
        pool_config.ghost_cells = cfg.ng;
        pool_config.num_levels = use_bbh_split ? binary_hierarchy_cfg.shared_levels.size() + 2
                                               : level_cfgs.size() + 1;
        pool_config.refinement_ratio = cfg.fmr.refinement_ratio;
        pool_config.grids_per_level = 5;  // Current + 4 RK4 stages
        memory_manager.configure(pool_config);
        std::cout << "[memory] pool_initialized=1 expected_mb="
                  << (pool_config.total_memory_bytes<double>() / (1024.0 * 1024.0))
                  << " levels=" << pool_config.num_levels << std::endl;
    } else {
        std::cout << "[memory] pool_initialized=0 reason=no_fmr" << std::endl;
    }

    std::cout << "[log] state_log_stride=" << output_stride << std::endl;
    std::cout << "[viz] slice_export_stride=" << slice_export_stride
              << " volume_export_stride=" << volume_export_stride << std::endl;
    std::cout << "[constraints] norms_stride=" << constraint_export_stride
              << " slice_export=" << (export_constraint_slices ? 1 : 0)
              << " slice_stride=" << constraint_slice_stride
              << " guard_cells=" << constraint_guard_cells << std::endl;
    std::cout << "[hdf5] requested=" << (request_hdf5_export ? 1 : 0)
              << " available=" << (tensorium::io::hdf5_available() ? 1 : 0)
              << " active=" << (export_hdf5 ? 1 : 0)
              << " csv_active=" << (export_csv ? 1 : 0) << std::endl;
    if (volume_export_stride > 0 && !export_hdf5) {
        std::cout << "[warn] volume export requested but HDF5 export is inactive; disabling volume export"
                  << std::endl;
        volume_export_stride = 0;
    }

    std::ofstream constraint_log;
    std::ofstream puncture_track;
    std::ofstream puncture_track_minima;
    bool csv_outputs_initialized = false;
    auto initialize_csv_outputs = [&](const char *reason) {
        if (csv_outputs_initialized)
            return;
        if (reason != nullptr) {
            std::cout << "[warn] " << reason << std::endl;
        }
        csv_outputs_initialized = true;
        export_csv = true;
        (void)std::remove("Output/viz/constraints_norms.csv");
        constraint_log.open("Output/viz/constraints_norms.csv", std::ios::out | std::ios::trunc);
        if (constraint_log.is_open()) {
            constraint_log.setf(std::ios::unitbuf);
            constraint_log
                << "step,t,dt,l2_theta,l2_Z,l2_H,l2_M,max_H,max_det_drift,max_trace_A,samples\n";
        } else {
            std::cout << "[warn] could not open Output/viz/constraints_norms.csv for writing"
                      << std::endl;
        }

        (void)std::remove("Output/viz/puncture_track.csv");
        puncture_track.open("Output/viz/puncture_track.csv", std::ios::out | std::ios::trunc);
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
        puncture_track_minima.open("Output/viz/puncture_track_minima.csv",
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
    };
    if (export_csv) {
        initialize_csv_outputs(
            request_hdf5_export && !tensorium::io::hdf5_available()
                ? "HDF5 export requested but this build has no HDF5 support; falling back to CSV"
                : nullptr);
    }

    tensorium::io::HDF5AppendTable constraint_log_h5;
    tensorium::io::HDF5AppendTable puncture_track_h5;
    tensorium::io::HDF5AppendTable puncture_track_minima_h5;
    constexpr const char *constraint_log_h5_path = "Output/viz/constraints_norms.h5";
    constexpr const char *puncture_track_h5_path = "Output/viz/puncture_track.h5";
    constexpr const char *puncture_track_minima_h5_path = "Output/viz/puncture_track_minima.h5";
    bool hdf5_log_init_failed = false;
    if (export_hdf5) {
        (void)std::remove(constraint_log_h5_path);
        (void)std::remove(puncture_track_h5_path);
        (void)std::remove(puncture_track_minima_h5_path);

        if (!constraint_log_h5.open(constraint_log_h5_path,
                                    {"step", "t", "dt", "l2_theta", "l2_Z", "l2_H", "l2_M",
                                     "max_H", "max_det_drift", "max_trace_A", "samples"})) {
            std::cout << "[warn] could not open " << constraint_log_h5_path << " for writing"
                      << std::endl;
            (void)std::remove(constraint_log_h5_path);
            hdf5_log_init_failed = true;
        }
        if (!puncture_track_h5.open(puncture_track_h5_path,
                                    {"step", "t", "x_left", "y_left", "chi_left", "alpha_left",
                                     "x_right", "y_right", "chi_right", "alpha_right"})) {
            std::cout << "[warn] could not open " << puncture_track_h5_path << " for writing"
                      << std::endl;
            (void)std::remove(puncture_track_h5_path);
            hdf5_log_init_failed = true;
        }
        if (!puncture_track_minima_h5.open(puncture_track_minima_h5_path,
                                           {"step", "t", "x_left", "y_left", "chi_left",
                                            "alpha_left", "x_right", "y_right", "chi_right",
                                            "alpha_right"})) {
            std::cout << "[warn] could not open " << puncture_track_minima_h5_path
                      << " for writing" << std::endl;
            (void)std::remove(puncture_track_minima_h5_path);
            hdf5_log_init_failed = true;
        }
    }
    if (export_hdf5 && hdf5_log_init_failed) {
        initialize_csv_outputs(
            "HDF5 append-table export initialization failed; enabling CSV diagnostics fallback");
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

    std::vector<tensorium_RG::z4c::MovingPunctureTrackPoint> ecc_track;
    if (cfg.ecc_control.enabled)
        ecc_track.reserve(cfg.steps + 1);
    auto append_ecc_track = [&](double time, const PuncturePlaneSample &sample) {
        if (!cfg.ecc_control.enabled)
            return;
        ecc_track.push_back(tensorium_RG::z4c::MovingPunctureTrackPoint{
            time,
            sample.has_left,
            sample.has_right,
            sample.x_left,
            sample.y_left,
            sample.x_right,
            sample.y_right,
        });
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
    } else if (!use_fmr) {
        initialize_shift_puncture_tracker_split(*root_state, *root_state, *root_state,
                                                puncture_tracker, initial_punctures[0],
                                                initial_punctures[1]);
        std::cout << "[tracker] enabled=" << (puncture_tracker.initialized ? 1 : 0)
                  << " mode=shift_no_fmr"
                  << " recenter_on_drift=" << (cfg.fmr.tracker_recenter_on_drift ? 1 : 0)
                  << " recenter_cells=" << cfg.fmr.tracker_recenter_cells
                  << " drift_warn_cells=" << cfg.fmr.tracker_drift_warn_cells << std::endl;
    }

    {
        PuncturePlaneSample initial_sample;
        if (use_fmr && use_bbh_split) {
            initial_sample = sample_puncture_minima_split(*left_puncture_state, *right_puncture_state);
            if (puncture_tracker.initialized)
                initial_sample = make_tracker_sample_split(*left_puncture_state,
                                                           *right_puncture_state,
                                                           puncture_tracker);
        } else if (use_fmr) {
            initial_sample = sample_puncture_minima(*puncture_state);
            if (puncture_tracker.initialized)
                initial_sample = make_tracker_sample(*puncture_state, puncture_tracker);
        } else {
            initial_sample = sample_puncture_minima(*root_state);
            if (puncture_tracker.initialized)
                initial_sample = make_tracker_sample(*root_state, puncture_tracker);
        }
        append_ecc_track(0.0, initial_sample);
    }

    size_t current_step = 0;
    double current_time = 0.0;
    double current_dt = 0.0;

    tensorium_RG::z4c::CFLControl<double> control;
    control.cfl = cfg.cfl;
    control.gauge_speed = cfg.gauge_speed;

    double t = 0.0;
    size_t memory_log_stride = parse_env_stride_or("TENSORIUM_MOVING_PUNCTURE_MEMORY_LOG_STRIDE", 100);
    if (use_fmr && use_bbh_split) {
        binary_hierarchy->set_gauge_parameters(params);
        binary_hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
        ConstraintScratch constraint_scratch(*root_state);
        const auto elapsed_seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };

        for (size_t n = 0; n < cfg.steps; ++n) {
            memory_manager.begin_epoch();
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

                    // Check if only leaves need regridding (punctures moved but still within shared level)
                    const bool leaves_need_regrid = (left_shift > regrid_threshold || right_shift > regrid_threshold);
                    const bool shared_needs_regrid = (center_shift > regrid_threshold);
                    bool need_full_rebuild = shared_needs_regrid || leaves_need_regrid;

                    if (leaves_need_regrid && !shared_needs_regrid) {
                        try {
                            // Fast path: only regrid leaves using tricubic interpolation.
                            const auto &parent_grid = binary_hierarchy->shared_level_grid(
                                binary_hierarchy->num_shared_levels() - 1);
                            constexpr size_t clearance = 2;
                            const double leaf_half_width =
                                (cfg.fmr.finest_box_half_width > 0.0)
                                    ? cfg.fmr.finest_box_half_width
                                    : std::max(cfg.fmr.puncture_buffer > 0.0 ? cfg.fmr.puncture_buffer
                                                                              : 4.0 * cfg.spacing,
                                               4.0 * cfg.spacing);

                            const auto [new_left_box, new_right_box] =
                                tensorium_RG::z4c::detail::moving_puncture_disjoint_split_leaf_boxes(
                                    parent_grid, puncture_tracker.p1, puncture_tracker.p2,
                                    leaf_half_width, clearance);

                            if (tensorium_RG::z4c::fmr::detail::patch_boxes_overlap(new_left_box,
                                                                                    new_right_box)) {
                                need_full_rebuild = false;
                                std::cout << "[fmr.regrid.fast][warn] step=" << (n + 1)
                                          << " split leaf boxes overlap; keeping current hierarchy"
                                          << " (reduce finest width or switch layout)" << std::endl;
                            } else if (binary_hierarchy->regrid_leaves(new_left_box, new_right_box)) {
                                left_puncture_state = &binary_hierarchy->leaf_grid(0);
                                right_puncture_state = &binary_hierarchy->leaf_grid(1);
                                puncture_state = left_puncture_state;
                                refresh_shift_puncture_tracker_beta_split(*left_puncture_state,
                                                                          *right_puncture_state,
                                                                          puncture_tracker);
                                puncture_minima = sample_puncture_minima_split(*left_puncture_state,
                                                                               *right_puncture_state);
                                puncture_sample = make_tracker_sample_split(*left_puncture_state,
                                                                           *right_puncture_state,
                                                                           puncture_tracker);
                                need_full_rebuild = false;
                                std::cout << "[fmr.regrid.fast] step=" << (n + 1)
                                          << " left_shift=" << left_shift
                                          << " right_shift=" << right_shift
                                          << " (leaves only, tricubic transfer)" << std::endl;
                            } else {
                                std::cout << "[fmr.regrid.fast][warn] step=" << (n + 1)
                                          << " fast leaf regrid was rejected; falling back to full rebuild"
                                          << std::endl;
                            }
                        } catch (const std::exception &ex) {
                            std::cout << "[fmr.regrid.fast][warn] step=" << (n + 1)
                                      << " fast leaf regrid failed: " << ex.what()
                                      << "; falling back to full rebuild" << std::endl;
                        }
                    }
                    if (need_full_rebuild) {
                        // Full rebuild when shared levels need to move or the fast path cannot be used.
                        try {
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
                            std::cout << "[fmr.regrid.full] step=" << (n + 1)
                                      << " center_old=(" << current_center[0] << ", " << current_center[1]
                                      << ", " << current_center[2] << ")"
                                      << " center_new=(" << target_center[0] << ", " << target_center[1]
                                      << ", " << target_center[2] << ")"
                                      << " shift=" << center_shift
                                      << " left_shift=" << left_shift
                                      << " right_shift=" << right_shift
                                      << " (full hierarchy rebuild)" << std::endl;
                        } catch (const std::exception &ex) {
                            if (!is_recoverable_bbh_split_regrid_geometry_error(ex))
                                throw;
                            std::cout << "[fmr.regrid.full][warn] step=" << (n + 1)
                                      << " regrid skipped: " << ex.what()
                                      << " ; keeping current hierarchy" << std::endl;
                        }
                    }
                }
            }

            double constraint_seconds = 0.0;
            const bool need_constraint_export =
                constraint_export_stride > 0 &&
                (constraint_log.is_open() || constraint_log_h5.is_open()) &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(
                    *root_state, constraint_scratch,
                    std::max(cfg.padding, constraint_guard_cells));
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                if (constraint_log.is_open()) {
                    constraint_log << current_step << "," << current_time << "," << current_dt
                                   << "," << stats.l2_theta << "," << stats.l2_Z << ","
                                   << stats.l2_H << "," << stats.l2_M << "," << stats.max_H
                                   << "," << stats.max_det_drift << "," << stats.max_trace_A
                                   << "," << stats.samples << "\n";
                }
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
                bool wrote_hdf5_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_slice =
                        export_slice_hdf5(*root_state, binary_hierarchy.get(), n, t, "Output/viz");
                    if (!wrote_hdf5_slice) {
                        std::cout << "[warn] HDF5 slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_slice))
                    export_slice_csv(*root_state, binary_hierarchy.get(), n, "Output/viz");
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (export_hdf5 && volume_export_stride > 0 && (n % volume_export_stride) == 0) {
                std::cout << ">> Exporting domain volume " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                if (!export_volume_hdf5_xdmf(*root_state, n, t, "Output/viz")) {
                    std::cout << "[warn] HDF5 volume export failed at step " << n << std::endl;
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                bool wrote_hdf5_constraint_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_constraint_slice =
                        export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                     constraint_scratch.M, constraint_scratch.C,
                                                     n, t, "Output/viz", constraint_guard_cells);
                    if (!wrote_hdf5_constraint_slice) {
                        std::cout << "[warn] HDF5 constraint-slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_constraint_slice)) {
                    export_constraint_slice_csv(*root_state, constraint_scratch.H,
                                                constraint_scratch.M, constraint_scratch.C, n,
                                                "Output/viz", constraint_guard_cells);
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }

            if (puncture_track.is_open())
                write_puncture_row(puncture_track, n, t, puncture_sample);
            if (puncture_track_minima.is_open())
                write_puncture_row(puncture_track_minima, n, t, puncture_minima);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_minima);
            append_ecc_track(t, puncture_sample);

            double projection_seconds = 0.0;
            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0)) {
                const auto projection_start = std::chrono::steady_clock::now();
                project_binary_hierarchy_levels(*binary_hierarchy, proj_cfg);
                const auto projection_stop = std::chrono::steady_clock::now();
                projection_seconds = elapsed_seconds(projection_start, projection_stop);
            }

            memory_manager.end_epoch();

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            std::printf(
                "dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds, constraint_seconds,
                export_seconds, projection_seconds);

            // Periodic memory pool statistics
            if (memory_log_stride > 0 && ((n + 1) % memory_log_stride) == 0) {
                const auto& stats = memory_manager.pool().stats();
                std::cout << "[memory.epoch] step=" << (n + 1)
                          << " allocs=" << stats.allocation_count.load()
                          << " reuses=" << stats.reuse_count.load()
                          << " in_use_mb=" << (stats.total_in_use.load() / (1024.0 * 1024.0))
                          << " peak_mb=" << (stats.peak_in_use.load() / (1024.0 * 1024.0))
                          << std::endl;
            }
        }
    } else if (use_fmr) {
        hierarchy->set_gauge_parameters(params);
        hierarchy->set_state_log_stride(std::numeric_limits<size_t>::max());
        ConstraintScratch constraint_scratch(*root_state);
        const auto elapsed_seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };

        for (size_t n = 0; n < cfg.steps; ++n) {
            memory_manager.begin_epoch();
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
                constraint_export_stride > 0 &&
                (constraint_log.is_open() || constraint_log_h5.is_open()) &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(
                    *root_state, constraint_scratch,
                    std::max(cfg.padding, constraint_guard_cells));
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                if (constraint_log.is_open()) {
                    constraint_log << current_step << "," << current_time << "," << current_dt
                                   << "," << stats.l2_theta << "," << stats.l2_Z << ","
                                   << stats.l2_H << "," << stats.l2_M << "," << stats.max_H
                                   << "," << stats.max_det_drift << "," << stats.max_trace_A
                                   << "," << stats.samples << "\n";
                }
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
                bool wrote_hdf5_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_slice =
                        export_slice_hdf5(*root_state, hierarchy.get(), n, t, "Output/viz");
                    if (!wrote_hdf5_slice) {
                        std::cout << "[warn] HDF5 slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_slice))
                    export_slice_csv(*root_state, hierarchy.get(), n, "Output/viz");
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (export_hdf5 && volume_export_stride > 0 && (n % volume_export_stride) == 0) {
                std::cout << ">> Exporting domain volume " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                if (!export_volume_hdf5_xdmf(*root_state, n, t, "Output/viz")) {
                    std::cout << "[warn] HDF5 volume export failed at step " << n << std::endl;
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                bool wrote_hdf5_constraint_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_constraint_slice =
                        export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                     constraint_scratch.M, constraint_scratch.C,
                                                     n, t, "Output/viz", constraint_guard_cells);
                    if (!wrote_hdf5_constraint_slice) {
                        std::cout << "[warn] HDF5 constraint-slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_constraint_slice)) {
                    export_constraint_slice_csv(*root_state, constraint_scratch.H,
                                                constraint_scratch.M, constraint_scratch.C, n,
                                                "Output/viz", constraint_guard_cells);
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }

            if (puncture_track.is_open())
                write_puncture_row(puncture_track, n, t, puncture_sample);
            if (puncture_track_minima.is_open())
                write_puncture_row(puncture_track_minima, n, t, puncture_minima);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_minima);
            append_ecc_track(t, puncture_sample);

            double projection_seconds = 0.0;
            if (cfg.projection_stride > 0 && ((n + 1) % cfg.projection_stride == 0))
            {
                const auto projection_start = std::chrono::steady_clock::now();
                project_hierarchy_levels(*hierarchy, proj_cfg);
                const auto projection_stop = std::chrono::steady_clock::now();
                projection_seconds = elapsed_seconds(projection_start, projection_stop);
            }

            memory_manager.end_epoch();

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                        dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds,
                        constraint_seconds, export_seconds, projection_seconds);

            // Periodic memory pool statistics
            if (memory_log_stride > 0 && ((n + 1) % memory_log_stride) == 0) {
                const auto& stats = memory_manager.pool().stats();
                std::cout << "[memory.epoch] step=" << (n + 1)
                          << " allocs=" << stats.allocation_count.load()
                          << " reuses=" << stats.reuse_count.load()
                          << " in_use_mb=" << (stats.total_in_use.load() / (1024.0 * 1024.0))
                          << " peak_mb=" << (stats.peak_in_use.load() / (1024.0 * 1024.0))
                          << std::endl;
            }
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
            memory_manager.begin_epoch();
            const auto wall_start = std::chrono::steady_clock::now();
            const double dt = tensorium_RG::z4c::compute_dt_cfl(*root_state, control, cfg.padding);
            current_step = n;
            current_dt = dt;
            current_time = t + dt;
            const auto evolve_start = std::chrono::steady_clock::now();
            stepper.step(*root_state, dt, n);
            const auto evolve_stop = std::chrono::steady_clock::now();
            t += dt;

            PuncturePlaneSample puncture_sample = sample_puncture_minima(*root_state);
            PuncturePlaneSample puncture_minima = puncture_sample;
            if (puncture_tracker.initialized) {
                advance_shift_puncture_tracker(*root_state, puncture_tracker, dt);
                puncture_sample = make_tracker_sample(*root_state, puncture_tracker);

                const size_t local_min_radius_cells =
                    static_cast<size_t>(std::ceil(std::max(6.0, cfg.fmr.tracker_recenter_cells)));
                puncture_minima = sample_puncture_minima_near_tracker(
                    *root_state, puncture_tracker, local_min_radius_cells);

                double drift_left = std::numeric_limits<double>::quiet_NaN();
                double drift_right = std::numeric_limits<double>::quiet_NaN();
                const bool have_drift = compute_tracker_drift(puncture_sample, puncture_minima,
                                                              drift_left, drift_right);
                const double drift_warn_radius =
                    cfg.fmr.tracker_drift_warn_cells * grid_max_spacing(*root_state);
                const double recenter_radius =
                    cfg.fmr.tracker_recenter_cells * grid_max_spacing(*root_state);
                if (cfg.fmr.tracker_recenter_on_drift && have_drift &&
                    (drift_left > recenter_radius || drift_right > recenter_radius)) {
                    puncture_tracker.p1 = {puncture_minima.x_left, puncture_minima.y_left, 0.0};
                    puncture_tracker.p2 = {puncture_minima.x_right, puncture_minima.y_right, 0.0};
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p1);
                    clamp_tracker_to_domain(*root_state, puncture_tracker.p2);
                    refresh_shift_puncture_tracker_beta(*root_state, puncture_tracker);
                    puncture_sample = make_tracker_sample(*root_state, puncture_tracker);
                } else if (have_drift &&
                           (drift_left > drift_warn_radius || drift_right > drift_warn_radius) &&
                           (n % std::max<size_t>(output_stride, size_t(1)) == 0)) {
                    std::cout << "[tracker][warn] step=" << n << " drift_left=" << drift_left
                              << " drift_right=" << drift_right
                              << " warn_radius=" << drift_warn_radius << std::endl;
                }
            }
            if (puncture_track.is_open())
                write_puncture_row(puncture_track, n, t, puncture_sample);
            if (puncture_track_minima.is_open())
                write_puncture_row(puncture_track_minima, n, t, puncture_minima);
            write_puncture_row_hdf5(puncture_track_h5, n, t, puncture_sample);
            write_puncture_row_hdf5(puncture_track_minima_h5, n, t, puncture_minima);
            append_ecc_track(t, puncture_sample);

            double constraint_seconds = 0.0;
            const bool need_constraint_export =
                constraint_export_stride > 0 &&
                (constraint_log.is_open() || constraint_log_h5.is_open()) &&
                (current_step % constraint_export_stride) == 0;
            const bool need_constraint_slice_export =
                export_constraint_slices && constraint_slice_stride > 0 &&
                (current_step % constraint_slice_stride) == 0;
            if (need_constraint_export) {
                const auto constraint_start = std::chrono::steady_clock::now();
                auto stats = compute_constraint_stats(
                    *root_state, constraint_scratch,
                    std::max(cfg.padding, constraint_guard_cells));
                const auto constraint_stop = std::chrono::steady_clock::now();
                constraint_seconds = elapsed_seconds(constraint_start, constraint_stop);
                if (constraint_log.is_open()) {
                    constraint_log << current_step << "," << current_time << "," << current_dt
                                   << "," << stats.l2_theta << "," << stats.l2_Z << ","
                                   << stats.l2_H << "," << stats.l2_M << "," << stats.max_H
                                   << "," << stats.max_det_drift << "," << stats.max_trace_A
                                   << "," << stats.samples << "\n";
                }
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
                bool wrote_hdf5_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_slice = export_slice_hdf5<MovingPunctureHierarchy>(
                        *root_state, nullptr, n, t, "Output/viz");
                    if (!wrote_hdf5_slice) {
                        std::cout << "[warn] HDF5 slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_slice)) {
                    export_slice_csv<MovingPunctureHierarchy>(*root_state, nullptr, n, "Output/viz");
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (export_hdf5 && volume_export_stride > 0 && (n % volume_export_stride) == 0) {
                std::cout << ">> Exporting domain volume " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                if (!export_volume_hdf5_xdmf(*root_state, n, t, "Output/viz")) {
                    std::cout << "[warn] HDF5 volume export failed at step " << n << std::endl;
                }
                const auto export_stop = std::chrono::steady_clock::now();
                export_seconds += elapsed_seconds(export_start, export_stop);
            }
            if (need_constraint_slice_export) {
                std::cout << ">> Exporting constraint slice " << n << "..." << std::endl;
                const auto export_start = std::chrono::steady_clock::now();
                compute_constraint_slice_fields(*root_state, constraint_scratch);
                bool wrote_hdf5_constraint_slice = false;
                if (export_hdf5) {
                    wrote_hdf5_constraint_slice =
                        export_constraint_slice_hdf5(*root_state, constraint_scratch.H,
                                                     constraint_scratch.M, constraint_scratch.C,
                                                     n, t, "Output/viz", constraint_guard_cells);
                    if (!wrote_hdf5_constraint_slice) {
                        std::cout << "[warn] HDF5 constraint-slice export failed at step " << n
                                  << "; falling back to CSV" << std::endl;
                    }
                }
                if (export_csv || (export_hdf5 && !wrote_hdf5_constraint_slice)) {
                    export_constraint_slice_csv(*root_state, constraint_scratch.H,
                                                constraint_scratch.M, constraint_scratch.C, n,
                                                "Output/viz", constraint_guard_cells);
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

            memory_manager.end_epoch();

            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_seconds = elapsed_seconds(wall_start, wall_stop);
            const double evolve_seconds = elapsed_seconds(evolve_start, evolve_stop);
            std::printf("dt = %.4e  step=%zu/%zu  t=%.4f  wall=%.3fs  evolve=%.3fs  constraints=%.3fs  export=%.3fs  projection=%.3fs\n",
                        dt, n + 1, cfg.steps, t, wall_seconds, evolve_seconds,
                        constraint_seconds, export_seconds, projection_seconds);

            // Periodic memory pool statistics
            if (memory_log_stride > 0 && ((n + 1) % memory_log_stride) == 0) {
                const auto& stats = memory_manager.pool().stats();
                std::cout << "[memory.epoch] step=" << (n + 1)
                          << " allocs=" << stats.allocation_count.load()
                          << " reuses=" << stats.reuse_count.load()
                          << " in_use_mb=" << (stats.total_in_use.load() / (1024.0 * 1024.0))
                          << " peak_mb=" << (stats.peak_in_use.load() / (1024.0 * 1024.0))
                          << std::endl;
            }
        }
    }

    std::cout << "[done] steps=" << cfg.steps << " t_final=" << t << std::endl;

    // Print memory pool statistics
    const auto& pool_stats = memory_manager.pool().stats();
    std::cout << "[memory.stats] total_allocated_mb="
              << (pool_stats.total_allocated.load() / (1024.0 * 1024.0))
              << " peak_usage_mb=" << (pool_stats.peak_in_use.load() / (1024.0 * 1024.0))
              << " allocations=" << pool_stats.allocation_count.load()
              << " reuse_count=" << pool_stats.reuse_count.load();
    if (pool_stats.allocation_count.load() > 0) {
        const double hit_rate = 100.0 * pool_stats.reuse_count.load() /
                                pool_stats.allocation_count.load();
        std::cout << " hit_rate=" << hit_rate << "%";
    }
    std::cout << std::endl;

    if (ecc_tuning_pass) {
        const auto fit =
            tensorium_RG::z4c::fit_moving_puncture_eccentricity(ecc_track, cfg.ecc_control);
        if (!fit.valid) {
            std::cout << "[ecc][warn] iteration " << (pass + 1)
                      << " failed to fit eccentricity from the puncture track; "
                         "keeping the current momenta and stopping the tuning loop."
                      << std::endl;
            break;
        }

        const auto update = tensorium_RG::z4c::suggest_moving_puncture_momentum_update(
            fit, cfg.mass1, cfg.mass2, cfg.tangential_momentum, cfg.radial_momentum,
            cfg.ecc_control);
        std::cout << "[ecc] iteration " << (pass + 1)
                  << " e=" << fit.eccentricity
                  << " fit_tmin=" << fit.fit_tmin
                  << " fit_tmax=" << fit.fit_tmax
                  << " samples=" << fit.used_samples
                  << " phase_advance=" << fit.phase_advance
                  << " omega=" << fit.orbital_frequency
                  << " r_mean=" << fit.mean_separation
                  << " drift=" << fit.secular_rdot
                  << " a_cos=" << fit.cos_coefficient
                  << " b_sin=" << fit.sin_coefficient
                  << " rmse=" << fit.fit_rmse << std::endl;
        if (!update.valid) {
            std::cout << "[ecc][warn] iteration " << (pass + 1)
                      << " produced an invalid momentum update; "
                         "keeping the current momenta and stopping the tuning loop."
                      << std::endl;
            break;
        }

        std::cout << "[ecc] update"
                  << " delta_p_tang=" << update.delta_tangential_momentum
                  << " delta_p_rad=" << update.delta_radial_momentum
                  << " p_tang_next=" << update.new_tangential_momentum
                  << " p_rad_next=" << update.new_radial_momentum << std::endl;
        cfg.tangential_momentum = update.new_tangential_momentum;
        cfg.user_tangential_momentum = update.new_tangential_momentum;
        cfg.radial_momentum = update.new_radial_momentum;
        if (cfg.use_interpolated_init) {
            tensorium_RG::init::reset_twopunctures_c_backend_cache();
            std::cout << "[ecc] reset TwoPunctures backend cache for the next pass" << std::endl;
        }
        continue;
    }
    }

    return 0;
}
