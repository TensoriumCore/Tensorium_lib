#define TENSORIUM_FORCE_NO_CBLAS 1
#include "../includes/Tensorium/Tensorium.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Fields/BSSNGridSoA.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/Grid/BSSNGridOperations.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/InitialData/BSSNInitialData.hpp"
#include "../includes/Tensorium/Physics/DiffGeometry/BSSN_Grid/TimeIntegration/BSSNRK4.hpp"
#include <algorithm>
#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <vector>

namespace py = pybind11;
using namespace tensorium;

namespace {

using NumpyArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using NumpyArrayF = py::array_t<float, py::array::c_style | py::array::forcecast>;
using NumpyArrayD3 = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename T> std::string vector_repr(const Vector<T> &v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

template <typename T> std::string matrix_repr(const Matrix<T> &m) {
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < m.rows; ++i) {
        oss << "  [";
        for (size_t j = 0; j < m.cols; ++j) {
            oss << m(i, j);
            if (j + 1 < m.cols)
                oss << ", ";
        }
        oss << "]";
        if (i + 1 < m.rows)
            oss << ",";
        oss << "\n";
    }
    oss << "]";
    return oss.str();
}

Vector<double> numpy_to_vectord(const NumpyArrayD &arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected a 1D NumPy array for Vectord");
    const auto *ptr = static_cast<const double *>(buf.ptr);
    return Vector<double>(std::vector<double>(ptr, ptr + buf.shape[0]));
}

Vector<float> numpy_to_vectorf(const NumpyArrayF &arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected a 1D NumPy array for Vector");
    const auto *ptr = static_cast<const float *>(buf.ptr);
    return Vector<float>(std::vector<float>(ptr, ptr + buf.shape[0]));
}

Matrix<double> numpy_to_matrixd(const NumpyArrayD &arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected a 2D NumPy array for Matrixd");
    const auto rows = static_cast<size_t>(buf.shape[0]);
    const auto cols = static_cast<size_t>(buf.shape[1]);
    Matrix<double> out(rows, cols);
    auto           in = arr.unchecked<2>();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out(i, j) = in(i, j);
    return out;
}

Matrix<float> numpy_to_matrixf(const NumpyArrayF &arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected a 2D NumPy array for Matrix");
    const auto rows = static_cast<size_t>(buf.shape[0]);
    const auto cols = static_cast<size_t>(buf.shape[1]);
    Matrix<float> out(rows, cols);
    auto          in = arr.unchecked<2>();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out(i, j) = in(i, j);
    return out;
}

Tensor<double, 2> numpy_to_tensor2d(const NumpyArrayD &arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected a 2D NumPy array for Tensor2d");
    std::array<size_t, 2> dims = {static_cast<size_t>(buf.shape[0]),
                                  static_cast<size_t>(buf.shape[1])};
    Tensor<double, 2> out(dims);
    auto              in = arr.unchecked<2>();
    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            out(i, j) = in(i, j);
    return out;
}

Tensor<double, 4> numpy_to_tensor4d(const NumpyArrayD &arr) {
    auto buf = arr.request();
    if (buf.ndim != 4)
        throw std::runtime_error("Expected a 4D NumPy array for Tensor4d");
    std::array<size_t, 4> dims = {static_cast<size_t>(buf.shape[0]),
                                  static_cast<size_t>(buf.shape[1]),
                                  static_cast<size_t>(buf.shape[2]),
                                  static_cast<size_t>(buf.shape[3])};
    Tensor<double, 4> out(dims);
    auto              in = arr.unchecked<4>();
    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            for (size_t k = 0; k < dims[2]; ++k)
                for (size_t l = 0; l < dims[3]; ++l)
                    out(i, j, k, l) = in(i, j, k, l);
    return out;
}

py::array_t<double> vectord_to_numpy(const Vector<double> &v) {
    py::array_t<double> out(v.size());
    auto                a = out.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i)
        a(i) = v[i];
    return out;
}

py::array_t<float> vectorf_to_numpy(const Vector<float> &v) {
    py::array_t<float> out(v.size());
    auto               a = out.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i)
        a(i) = v[i];
    return out;
}

py::array_t<double> matrixd_to_numpy(const Matrix<double> &m) {
    py::array_t<double> out({m.rows, m.cols});
    auto                a = out.mutable_unchecked<2>();
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t j = 0; j < m.cols; ++j)
            a(i, j) = m(i, j);
    return out;
}

py::array_t<float> matrixf_to_numpy(const Matrix<float> &m) {
    py::array_t<float> out({m.rows, m.cols});
    auto               a = out.mutable_unchecked<2>();
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t j = 0; j < m.cols; ++j)
            a(i, j) = m(i, j);
    return out;
}

py::array_t<double> tensor2d_to_numpy(const Tensor<double, 2> &t) {
    const auto dims = t.shape();
    py::array_t<double> out({dims[0], dims[1]});
    auto                a = out.mutable_unchecked<2>();
    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            a(i, j) = t(i, j);
    return out;
}

py::array_t<double> tensor4d_to_numpy(const Tensor<double, 4> &t) {
    const auto dims = t.shape();
    py::array_t<double> out({dims[0], dims[1], dims[2], dims[3]});
    auto                a = out.mutable_unchecked<4>();
    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            for (size_t k = 0; k < dims[2]; ++k)
                for (size_t l = 0; l < dims[3]; ++l)
                    a(i, j, k, l) = t(i, j, k, l);
    return out;
}

template <typename T>
void fill_matrix_from_nested_vector(Matrix<T> &m, const std::vector<std::vector<T>> &values) {
    if (values.size() != m.rows)
        throw std::runtime_error("Shape mismatch in fill(): invalid row count");
    if (!values.empty() && values[0].size() != m.cols)
        throw std::runtime_error("Shape mismatch in fill(): invalid column count");

    for (size_t i = 0; i < m.rows; ++i) {
        if (values[i].size() != m.cols)
            throw std::runtime_error("Shape mismatch in fill(): ragged nested list");
        for (size_t j = 0; j < m.cols; ++j)
            m(i, j) = values[i][j];
    }
}

using BSSNGridD = tensorium_RG::BSSNGridSoA<double>;
using GaugeParamsD = tensorium_RG::bssn::GaugeParameters<double>;
using BSSNStepperD =
    tensorium_RG::bssn::BSSNRKStepper<double, tensorium_RG::bssn::BoundaryRadiative>;

inline int checked_vec_component(int component) {
    if (component < 0 || component > 2)
        throw std::out_of_range("component must be in [0, 2]");
    return component;
}

inline int checked_sym_component(int component) {
    if (component < 0 || component > 5)
        throw std::out_of_range("component must be in [0, 5]");
    return component;
}

template <typename T>
py::array_t<T> field_to_numpy(const tensorium_RG::Field3D<T> &field, const BSSNGridD &grid,
                              bool include_halo) {
    const size_t i0 = include_halo ? 0 : grid.dims.ng;
    const size_t j0 = include_halo ? 0 : grid.dims.ng;
    const size_t k0 = include_halo ? 0 : grid.dims.ng;
    const size_t ni = include_halo ? (grid.dims.nx + 2 * grid.dims.ng) : grid.dims.nx;
    const size_t nj = include_halo ? (grid.dims.ny + 2 * grid.dims.ng) : grid.dims.ny;
    const size_t nk = include_halo ? (grid.dims.nz + 2 * grid.dims.ng) : grid.dims.nz;

    py::array_t<T> out({ni, nj, nk});
    auto           out_v = out.template mutable_unchecked<3>();
    for (size_t i = 0; i < ni; ++i)
        for (size_t j = 0; j < nj; ++j)
            for (size_t k = 0; k < nk; ++k)
                out_v(i, j, k) = field.ptr()[field.idx(i0 + i, j0 + j, k0 + k)];
    return out;
}

template <typename T>
void numpy_to_field(tensorium_RG::Field3D<T> &field, const BSSNGridD &grid,
                    const py::array_t<T, py::array::c_style | py::array::forcecast> &arr,
                    bool include_halo, const char *field_name) {
    auto buf = arr.request();
    if (buf.ndim != 3)
        throw std::runtime_error(std::string(field_name) + " expects a 3D array");

    const size_t i0 = include_halo ? 0 : grid.dims.ng;
    const size_t j0 = include_halo ? 0 : grid.dims.ng;
    const size_t k0 = include_halo ? 0 : grid.dims.ng;
    const size_t ni = include_halo ? (grid.dims.nx + 2 * grid.dims.ng) : grid.dims.nx;
    const size_t nj = include_halo ? (grid.dims.ny + 2 * grid.dims.ng) : grid.dims.ny;
    const size_t nk = include_halo ? (grid.dims.nz + 2 * grid.dims.ng) : grid.dims.nz;

    if (static_cast<size_t>(buf.shape[0]) != ni || static_cast<size_t>(buf.shape[1]) != nj ||
        static_cast<size_t>(buf.shape[2]) != nk)
        throw std::runtime_error(std::string(field_name) + " shape mismatch");

    auto in = arr.template unchecked<3>();
    for (size_t i = 0; i < ni; ++i)
        for (size_t j = 0; j < nj; ++j)
            for (size_t k = 0; k < nk; ++k)
                field.ptr()[field.idx(i0 + i, j0 + j, k0 + k)] = in(i, j, k);
}

} // namespace

PYBIND11_MODULE(tensorium, m) {
    py::module_ tns = m.def_submodule("tns", "High-performance math operations");
    py::module_ bssn = m.def_submodule("bssn", "BSSN_Grid runtime bindings");

    bssn.attr("XX") = py::int_(static_cast<int>(tensorium_RG::XX));
    bssn.attr("XY") = py::int_(static_cast<int>(tensorium_RG::XY));
    bssn.attr("XZ") = py::int_(static_cast<int>(tensorium_RG::XZ));
    bssn.attr("YY") = py::int_(static_cast<int>(tensorium_RG::YY));
    bssn.attr("YZ") = py::int_(static_cast<int>(tensorium_RG::YZ));
    bssn.attr("ZZ") = py::int_(static_cast<int>(tensorium_RG::ZZ));
    bssn.def("has_twopunctures_c", []() {
#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
        return true;
#else
        return false;
#endif
    });
    bssn.def("set_spatial_derivative_order",
             [](int order) { tensorium_RG::fd::set_max_spatial_derivative_order(order); },
             py::arg("order"));
    bssn.def("spatial_derivative_order",
             []() { return tensorium_RG::fd::max_spatial_derivative_order(); });
    bssn.def("set_fd_dx", [](double dx) { tensorium_RG::fd::set_fd_dx(dx); }, py::arg("dx"));
    bssn.def(
        "set_boundary_faces",
        [](bool rhs_ix1, bool rhs_ox1, bool rhs_ix2, bool rhs_ox2, bool rhs_ix3, bool rhs_ox3,
           bool rf_ix1, bool rf_ox1, bool rf_ix2, bool rf_ox2, bool rf_ix3, bool rf_ox3) {
            tensorium_RG::bssn::BoundaryRadiative::set_rhs_sommerfeld_faces(rhs_ix1, rhs_ox1,
                                                                             rhs_ix2, rhs_ox2,
                                                                             rhs_ix3, rhs_ox3);
            tensorium_RG::bssn::BoundaryRadiative::set_reflective_faces(rf_ix1, rf_ox1, rf_ix2,
                                                                         rf_ox2, rf_ix3, rf_ox3);
        },
        py::arg("rhs_ix1"), py::arg("rhs_ox1"), py::arg("rhs_ix2"), py::arg("rhs_ox2"),
        py::arg("rhs_ix3"), py::arg("rhs_ox3"), py::arg("rf_ix1"), py::arg("rf_ox1"),
        py::arg("rf_ix2"), py::arg("rf_ox2"), py::arg("rf_ix3"), py::arg("rf_ox3"));

    py::class_<GaugeParamsD>(bssn, "GaugeParameters")
        .def(py::init<>())
        .def_readwrite("beta_B_coeff", &GaugeParamsD::beta_B_coeff)
        .def_readwrite("eta", &GaugeParamsD::eta)
        .def_readwrite("mass_scale", &GaugeParamsD::mass_scale)
        .def_readwrite("kappa1", &GaugeParamsD::kappa1)
        .def_readwrite("kappa2", &GaugeParamsD::kappa2)
        .def_readwrite("kappa3", &GaugeParamsD::kappa3)
        .def_readwrite("kappa_z", &GaugeParamsD::kappa_z)
        .def_readwrite("covariant_z4", &GaugeParamsD::covariant_z4)
        .def_readwrite("chi_div_floor", &GaugeParamsD::chi_div_floor)
        .def_readwrite("ko_sigma", &GaugeParamsD::ko_sigma)
        .def_readwrite("min_lapse_for_K", &GaugeParamsD::min_lapse_for_K)
        .def_readwrite("max_K_squared", &GaugeParamsD::max_K_squared)
        .def_readwrite("alpha_floor", &GaugeParamsD::alpha_floor)
        .def_readwrite("chi_floor", &GaugeParamsD::chi_floor)
        .def_readwrite("use_theta_in_lapse", &GaugeParamsD::use_theta_in_lapse)
        .def_readwrite("use_shift_advection", &GaugeParamsD::use_shift_advection)
        .def_readwrite("lapse_harmonicf", &GaugeParamsD::lapse_harmonicf)
        .def_readwrite("lapse_harmonic", &GaugeParamsD::lapse_harmonic)
        .def_readwrite("lapse_oplog", &GaugeParamsD::lapse_oplog)
        .def_readwrite("lapse_advect", &GaugeParamsD::lapse_advect)
        .def_readwrite("shift_Gamma", &GaugeParamsD::shift_Gamma)
        .def_readwrite("shift_advect", &GaugeParamsD::shift_advect)
        .def_readwrite("shift_alpha2Gamma", &GaugeParamsD::shift_alpha2Gamma)
        .def_readwrite("shift_H", &GaugeParamsD::shift_H)
        .def_readwrite("shift_eta", &GaugeParamsD::shift_eta)
        .def_readwrite("slow_start_lapse", &GaugeParamsD::slow_start_lapse)
        .def_readwrite("ssl_damping_amp", &GaugeParamsD::ssl_damping_amp)
        .def_readwrite("ssl_damping_time", &GaugeParamsD::ssl_damping_time)
        .def_readwrite("ssl_damping_index", &GaugeParamsD::ssl_damping_index)
        .def_readwrite("current_time", &GaugeParamsD::current_time)
        .def_readwrite("use_direct_shift_rhs", &GaugeParamsD::use_direct_shift_rhs)
        .def_readwrite("evolve_Z", &GaugeParamsD::evolve_Z)
        .def_readwrite("frozen_Z_is_synced", &GaugeParamsD::frozen_Z_is_synced)
        .def_readwrite("gamma_damping_uses_metric", &GaugeParamsD::gamma_damping_uses_metric)
        .def_readwrite("apply_rhs_sommerfeld", &GaugeParamsD::apply_rhs_sommerfeld);

    py::class_<BSSNGridD>(bssn, "BSSNGrid")
        .def(py::init<size_t, size_t, size_t, size_t, double, double, double>(), py::arg("nx"),
             py::arg("ny"), py::arg("nz"), py::arg("ng"), py::arg("dx"), py::arg("dy"),
             py::arg("dz"))
        .def_property_readonly("nx", [](const BSSNGridD &g) { return g.dims.nx; })
        .def_property_readonly("ny", [](const BSSNGridD &g) { return g.dims.ny; })
        .def_property_readonly("nz", [](const BSSNGridD &g) { return g.dims.nz; })
        .def_property_readonly("ng", [](const BSSNGridD &g) { return g.dims.ng; })
        .def_property_readonly("dx", [](const BSSNGridD &g) { return g.dx; })
        .def_property_readonly("dy", [](const BSSNGridD &g) { return g.dy; })
        .def_property_readonly("dz", [](const BSSNGridD &g) { return g.dz; })
        .def_property("x0", [](const BSSNGridD &g) { return g.x0; },
                      [](BSSNGridD &g, double v) { g.x0 = v; })
        .def_property("y0", [](const BSSNGridD &g) { return g.y0; },
                      [](BSSNGridD &g, double v) { g.y0 = v; })
        .def_property("z0", [](const BSSNGridD &g) { return g.z0; },
                      [](BSSNGridD &g, double v) { g.z0 = v; })
        .def("set_origin", [](BSSNGridD &g, double x0, double y0, double z0) {
            g.x0 = x0;
            g.y0 = y0;
            g.z0 = z0;
        })
        .def("domain_bounds", [](const BSSNGridD &g) {
            size_t i0, i1, j0, j1, k0, k1;
            g.domain_bounds(i0, i1, j0, j1, k0, k1);
            return py::make_tuple(i0, i1, j0, j1, k0, k1);
        })
        .def("coords", [](const BSSNGridD &g, size_t i, size_t j, size_t k) {
            double x, y, z;
            g.coords(i, j, k, x, y, z);
            return py::make_tuple(x, y, z);
        })
        .def("alpha", [](const BSSNGridD &g, bool include_halo) {
            return field_to_numpy(g.alpha, g, include_halo);
        }, py::arg("include_halo") = false)
        .def("chi", [](const BSSNGridD &g, bool include_halo) {
            return field_to_numpy(g.chi, g, include_halo);
        }, py::arg("include_halo") = false)
        .def("K", [](const BSSNGridD &g, bool include_halo) {
            return field_to_numpy(g.K, g, include_halo);
        }, py::arg("include_halo") = false)
        .def("Theta", [](const BSSNGridD &g, bool include_halo) {
            return field_to_numpy(g.Theta, g, include_halo);
        }, py::arg("include_halo") = false)
        .def("beta", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.beta[checked_vec_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("B", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.B[checked_vec_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("tildeGamma", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.tildeGamma[checked_vec_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("Z", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.Z[checked_vec_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("gamma_tilde", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.gamma_tilde[checked_sym_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("gamma_tilde_inv", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.gamma_tilde_inv[checked_sym_component(component)], g,
                                  include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("A_tilde", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.A_tilde[checked_sym_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("Ricci", [](const BSSNGridD &g, int component, bool include_halo) {
            return field_to_numpy(g.Ricci[checked_sym_component(component)], g, include_halo);
        }, py::arg("component"), py::arg("include_halo") = false)
        .def("set_alpha", [](BSSNGridD &g, const NumpyArrayD3 &arr, bool include_halo) {
            numpy_to_field(g.alpha, g, arr, include_halo, "alpha");
        }, py::arg("array"), py::arg("include_halo") = false)
        .def("set_chi", [](BSSNGridD &g, const NumpyArrayD3 &arr, bool include_halo) {
            numpy_to_field(g.chi, g, arr, include_halo, "chi");
        }, py::arg("array"), py::arg("include_halo") = false)
        .def("set_K", [](BSSNGridD &g, const NumpyArrayD3 &arr, bool include_halo) {
            numpy_to_field(g.K, g, arr, include_halo, "K");
        }, py::arg("array"), py::arg("include_halo") = false);

    py::class_<BSSNStepperD>(bssn, "BSSNRKStepper")
        .def(py::init<const BSSNGridD &, size_t>(), py::arg("prototype"), py::arg("padding") = 4)
        .def("set_gauge_parameters", &BSSNStepperD::set_gauge_parameters, py::arg("params"))
        .def("set_state_log_stride", &BSSNStepperD::set_state_log_stride, py::arg("stride"))
        .def("step", &BSSNStepperD::step, py::arg("grid"), py::arg("dt"),
             py::arg("step_index") = 0);

    bssn.def("compute_dt_cfl",
             [](const BSSNGridD &grid, double cfl, double gauge_speed, size_t padding) {
                 tensorium_RG::bssn::CFLControl<double> control;
                 control.cfl = cfl;
                 control.gauge_speed = gauge_speed;
                 return tensorium_RG::bssn::compute_dt_cfl(grid, control, padding);
             },
             py::arg("grid"), py::arg("cfl"), py::arg("gauge_speed") = 1.0,
             py::arg("padding") = 4);

    bssn.def("apply_radiative_halos", [](BSSNGridD &grid) {
        tensorium_RG::bssn::apply_halos_grid<tensorium_RG::bssn::BoundaryRadiative>(grid);
    }, py::arg("grid"));
    bssn.def("project_state", [](BSSNGridD &grid) { tensorium_RG::bssn::project_bssn_state(grid); },
             py::arg("grid"));
    bssn.def("zero_z4c_fields",
             [](BSSNGridD &grid) { tensorium_RG::init::zero_z4c_fields(grid); }, py::arg("grid"));

    bssn.def("minkowski",
             [](BSSNGridD &grid, double M, double xc, double yc, double zc, double r_floor) {
                 tensorium_RG::init::minkowski<double>(grid, M, xc, yc, zc, r_floor);
             },
             py::arg("grid"), py::arg("M") = 1.0, py::arg("xc") = 0.0, py::arg("yc") = 0.0,
             py::arg("zc") = 0.0, py::arg("r_floor") = 1e-6);

    bssn.def("schwarzschild_isotropic",
             [](BSSNGridD &grid, double M, double xc, double yc, double zc, double r_floor) {
                 tensorium_RG::init::schwarzschild_isotropic<double>(grid, M, xc, yc, zc, r_floor);
             },
             py::arg("grid"), py::arg("M"), py::arg("xc") = 0.0, py::arg("yc") = 0.0,
             py::arg("zc") = 0.0, py::arg("r_floor") = 1e-6);

    bssn.def("kerr_schild_single",
             [](BSSNGridD &grid, double M, double a, double xc, double yc, double zc,
                double r_floor) {
                 tensorium_RG::init::kerr_schild_single<double>(grid, M, a, xc, yc, zc, r_floor);
             },
             py::arg("grid"), py::arg("M"), py::arg("a"), py::arg("xc") = 0.0,
             py::arg("yc") = 0.0, py::arg("zc") = 0.0, py::arg("r_floor") = 1e-6);

    bssn.def(
        "binary_bowen_york_puncture_init",
        [](BSSNGridD &grid, double m1, double x1, double y1, double z1,
           const std::array<double, 3> &P1, const std::array<double, 3> &S1, double m2, double x2,
           double y2, double z2, const std::array<double, 3> &P2, const std::array<double, 3> &S2,
           double r_floor) {
            tensorium_RG::init::binary_bowen_york_puncture_init<double>(
                grid, m1, x1, y1, z1, P1.data(), S1.data(), m2, x2, y2, z2, P2.data(), S2.data(),
                r_floor);
        },
        py::arg("grid"), py::arg("m1"), py::arg("x1"), py::arg("y1"), py::arg("z1"),
        py::arg("P1"), py::arg("S1"), py::arg("m2"), py::arg("x2"), py::arg("y2"),
        py::arg("z2"), py::arg("P2"), py::arg("S2"), py::arg("r_floor") = 1e-6);

    bssn.def(
        "binary_bowen_york_puncture_interpolated_init",
        [](BSSNGridD &grid, double m1, double x1, double y1, double z1,
           const std::array<double, 3> &P1, const std::array<double, 3> &S1, double m2, double x2,
           double y2, double z2, const std::array<double, 3> &P2, const std::array<double, 3> &S2,
           size_t interp_seed_n, double r_floor) {
            tensorium_RG::init::binary_bowen_york_puncture_interpolated_init<double>(
                grid, m1, x1, y1, z1, P1.data(), S1.data(), m2, x2, y2, z2, P2.data(), S2.data(),
                interp_seed_n, r_floor);
        },
        py::arg("grid"), py::arg("m1"), py::arg("x1"), py::arg("y1"), py::arg("z1"),
        py::arg("P1"), py::arg("S1"), py::arg("m2"), py::arg("x2"), py::arg("y2"),
        py::arg("z2"), py::arg("P2"), py::arg("S2"), py::arg("interp_seed_n") = 64,
        py::arg("r_floor") = 1e-6);

    bssn.def(
        "binary_bowen_york_puncture_twopunctures_c_init",
        [](BSSNGridD &grid, double m1, double x1, double y1, double z1,
           const std::array<double, 3> &P1, const std::array<double, 3> &S1, double m2, double x2,
           double y2, double z2, const std::array<double, 3> &P2, const std::array<double, 3> &S2,
           size_t interp_seed_n, double r_floor) {
#if defined(TENSORIUM_HAS_TWOPUNCTURES_C)
            tensorium_RG::init::binary_bowen_york_puncture_twopunctures_c_init<double>(
                grid, m1, x1, y1, z1, P1.data(), S1.data(), m2, x2, y2, z2, P2.data(), S2.data(),
                interp_seed_n, r_floor);
#else
            (void)grid;
            (void)m1;
            (void)x1;
            (void)y1;
            (void)z1;
            (void)P1;
            (void)S1;
            (void)m2;
            (void)x2;
            (void)y2;
            (void)z2;
            (void)P2;
            (void)S2;
            (void)interp_seed_n;
            (void)r_floor;
            throw std::runtime_error(
                "TwoPuncturesC support is not enabled in this Python extension build");
#endif
        },
        py::arg("grid"), py::arg("m1"), py::arg("x1"), py::arg("y1"), py::arg("z1"),
        py::arg("P1"), py::arg("S1"), py::arg("m2"), py::arg("x2"), py::arg("y2"),
        py::arg("z2"), py::arg("P2"), py::arg("S2"), py::arg("interp_seed_n") = 64,
        py::arg("r_floor") = 1e-6);

    py::class_<Tensor<double, 4>>(tns, "Tensor4d")
        .def(py::init<const std::array<size_t, 4> &>(), py::arg("shape"))
        .def_static("from_numpy", &numpy_to_tensor4d, py::arg("array"))
        .def("to_numpy", &tensor4d_to_numpy)
        .def_property_readonly("shape", [](const Tensor<double, 4> &t) { return t.shape(); })
        .def("print", &Tensor<double, 4>::print)
        .def("__getitem__", [](const Tensor<double, 4> &t,
                               std::tuple<size_t, size_t, size_t, size_t> idx) {
            return t(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx), std::get<3>(idx));
        })
        .def("__setitem__", [](Tensor<double, 4> &t, std::tuple<size_t, size_t, size_t, size_t> idx,
                               double v) {
            t(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx), std::get<3>(idx)) = v;
        });

    py::class_<Tensor<double, 2>>(tns, "Tensor2d")
        .def(py::init<const std::array<size_t, 2> &>(), py::arg("shape"))
        .def_static("from_numpy", &numpy_to_tensor2d, py::arg("array"))
        .def("to_numpy", &tensor2d_to_numpy)
        .def_property_readonly("shape", [](const Tensor<double, 2> &t) { return t.shape(); })
        .def("print", &Tensor<double, 2>::print)
        .def("trace", [](const Tensor<double, 2> &t) {
            const auto dims = t.shape();
            if (dims[0] != dims[1])
                throw std::invalid_argument("trace() requires a square tensor");
            double sum = 0.0;
            for (size_t i = 0; i < dims[0]; ++i)
                sum += t(i, i);
            return sum;
        })
        .def("get", [](const Tensor<double, 2> &t, size_t i, size_t j) { return t(i, j); },
             py::arg("i"), py::arg("j"))
        .def("__call__", [](const Tensor<double, 2> &t, size_t i, size_t j) { return t(i, j); },
             py::arg("i"), py::arg("j"))
        .def("__getitem__", [](const Tensor<double, 2> &t, std::pair<size_t, size_t> idx) {
            return t(idx.first, idx.second);
        })
        .def("__setitem__", [](Tensor<double, 2> &t, std::pair<size_t, size_t> idx, double val) {
            t(idx.first, idx.second) = val;
        });

    py::class_<tensorium_RG::RiemannTensor<double>>(tns, "Riemann")
        .def_static("print_componentwise", &tensorium_RG::RiemannTensor<double>::print_componentwise,
                    py::arg("R"), py::arg("threshold") = 1e-12);

    py::class_<tensorium_RG::ChristoffelSym<double>>(tns, "Christoffel")
        .def("print", &tensorium_RG::ChristoffelSym<double>::print);

    py::class_<tensorium_RG::Metric<double>>(tns, "Metric")
        .def(py::init<const std::string &, double, double>(), py::arg("metric_type") = "minkowski",
             py::arg("mass") = 1.0, py::arg("spin") = 0.0)
        .def("__call__", [](tensorium_RG::Metric<double> &metric, const Vector<double> &x) {
            Tensor<double, 2> g({4, 4});
            metric(x, g);
            return g;
        });

    tns.def("compute_christoffel",
            [](const Vector<double> &x, const Tensor<double, 2> &g, const Tensor<double, 2> &g_inv,
               const std::string &metric_type, double mass, double spin) {
                auto metric = tensorium_RG::Metric<double>(metric_type, mass, spin);
                return tensorium::compute_christoffel(x, 1e-5, g, g_inv, metric);
            },
            py::arg("x"), py::arg("g"), py::arg("g_inv"), py::arg("metric_type"), py::arg("mass"),
            py::arg("spin"));

    tns.def("compute_riemann_tensor",
            [](const Vector<double> &x, const std::string &metric_type, double mass, double spin) {
                auto metric = tensorium_RG::Metric<double>(metric_type, mass, spin);
                return tensorium::compute_riemann_tensor<double>(x, 1e-5, metric);
            },
            py::arg("x"), py::arg("metric_type"), py::arg("mass"), py::arg("spin"));

    tns.def("contract_riemann_to_ricci", &tensorium::contract_riemann_to_ricci<double>,
            py::arg("riemann"), py::arg("g_inv"));
    tns.def("compute_ricci_scalar", &tensorium::compute_ricci_scalar<double>, py::arg("ricci"),
            py::arg("g_inv"));
    tns.def("print_ricci_tensor", &tensorium::print_ricci_tensor<double>, py::arg("ricci"));
    tns.def("print_ricci_scalar", &tensorium::print_ricci_scalar<double>, py::arg("ricci"),
            py::arg("g_inv"));
    tns.def("inv_mat_tensor", &tensorium::inv_mat_tensor<double>, py::arg("g"));

    py::class_<Vector<float>>(m, "Vector")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<std::vector<float>>(), py::arg("values"))
        .def_static("from_numpy", &numpy_to_vectorf, py::arg("array"))
        .def("to_numpy", &vectorf_to_numpy)
        .def("print", &Vector<float>::print)
        .def("__len__", &Vector<float>::size)
        .def("__getitem__", [](const Vector<float> &v, size_t i) { return v[i]; })
        .def("__setitem__", [](Vector<float> &v, size_t i, float value) { v[i] = value; })
        .def("__repr__", &vector_repr<float>);

    py::class_<Vector<double>>(m, "Vectord")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<std::vector<double>>(), py::arg("values"))
        .def_static("from_numpy", &numpy_to_vectord, py::arg("array"))
        .def("to_numpy", &vectord_to_numpy)
        .def("__len__", &Vector<double>::size)
        .def("__getitem__", [](const Vector<double> &v, size_t i) { return v[i]; })
        .def("__setitem__", [](Vector<double> &v, size_t i, double value) { v[i] = value; })
        .def("__repr__", &vector_repr<double>);

    py::class_<Matrix<float>>(m, "Matrix")
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def_static("from_numpy", &numpy_to_matrixf, py::arg("array"))
        .def("to_numpy", &matrixf_to_numpy)
        .def("print", &Matrix<float>::print)
        .def("__getitem__", [](const Matrix<float> &mat, std::pair<size_t, size_t> idx) {
            return mat(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix<float> &mat, std::pair<size_t, size_t> idx, float val) {
            mat(idx.first, idx.second) = val;
        })
        .def("rows", [](const Matrix<float> &mat) { return mat.rows; })
        .def("cols", [](const Matrix<float> &mat) { return mat.cols; })
        .def("fill", &fill_matrix_from_nested_vector<float>, py::arg("values"))
        .def("__repr__", &matrix_repr<float>);

    py::class_<Matrix<double>>(m, "Matrixd")
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def_static("from_numpy", &numpy_to_matrixd, py::arg("array"))
        .def("to_numpy", &matrixd_to_numpy)
        .def("print", &Matrix<double>::print)
        .def("__getitem__", [](const Matrix<double> &mat, std::pair<size_t, size_t> idx) {
            return mat(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix<double> &mat, std::pair<size_t, size_t> idx, double val) {
            mat(idx.first, idx.second) = val;
        })
        .def("rows", [](const Matrix<double> &mat) { return mat.rows; })
        .def("cols", [](const Matrix<double> &mat) { return mat.cols; })
        .def("fill", &fill_matrix_from_nested_vector<double>, py::arg("values"))
        .def("__repr__", &matrix_repr<double>);

    tns.def("add_vec", &add_vec<float>, py::arg("a"), py::arg("b"));
    tns.def("add_vec", &add_vec<double>, py::arg("a"), py::arg("b"));
    tns.def("sub_vec", &sub_vec<float>, py::arg("a"), py::arg("b"));
    tns.def("sub_vec", &sub_vec<double>, py::arg("a"), py::arg("b"));
    tns.def("scl_vec", &scl_vec<float>, py::arg("a"), py::arg("scalar"));
    tns.def("scl_vec", &scl_vec<double>, py::arg("a"), py::arg("scalar"));
    tns.def("dot_vec", &dot_vec<float>, py::arg("a"), py::arg("b"));
    tns.def("dot_vec", &dot_vec<double>, py::arg("a"), py::arg("b"));
    tns.def("norm_1", &norm1_vec<float>, py::arg("a"));
    tns.def("norm_1", &norm1_vec<double>, py::arg("a"));
    tns.def("norm_2", &norm2_vec<float>, py::arg("a"));
    tns.def("norm_2", &norm2_vec<double>, py::arg("a"));
    tns.def("norm_inf", &normInf_vec<float>, py::arg("a"));
    tns.def("norm_inf", &normInf_vec<double>, py::arg("a"));
    tns.def("cosine", &cosine_vec<float>, py::arg("a"), py::arg("b"));
    tns.def("cosine", &cosine_vec<double>, py::arg("a"), py::arg("b"));
    tns.def("lerp", &lerp_vec<float>, py::arg("a"), py::arg("b"), py::arg("t"));
    tns.def("lerp", &lerp_vec<double>, py::arg("a"), py::arg("b"), py::arg("t"));
    tns.def("linear_comb", &linear_combination_vec<float>, py::arg("vectors"), py::arg("coeffs"));
    tns.def("linear_comb", &linear_combination_vec<double>, py::arg("vectors"), py::arg("coeffs"));
    tns.def("cross", &cross_vec<float>, py::arg("a"), py::arg("b"));
    tns.def("cross", &cross_vec<double>, py::arg("a"), py::arg("b"));

    tns.def("add_mat", &add_mat<float>, py::arg("A"), py::arg("B"));
    tns.def("add_mat", &add_mat<double>, py::arg("A"), py::arg("B"));
    tns.def("sub_mat", &sub_mat<float>, py::arg("A"), py::arg("B"));
    tns.def("sub_mat", &sub_mat<double>, py::arg("A"), py::arg("B"));
    tns.def("scl_mat", &scl_mat<float>, py::arg("A"), py::arg("scalar"));
    tns.def("scl_mat", &scl_mat<double>, py::arg("A"), py::arg("scalar"));
    tns.def("lerp_mat", &lerp_mat<float>, py::arg("A"), py::arg("B"), py::arg("t"));
    tns.def("lerp_mat", &lerp_mat<double>, py::arg("A"), py::arg("B"), py::arg("t"));
    tns.def("mul", &mul_mat<float>, py::arg("A"), py::arg("B"));
    tns.def("mul", &mul_mat<double>, py::arg("A"), py::arg("B"));
    tns.def("transpose_mat", &transpose_mat<float>, py::arg("A"));
    tns.def("transpose_mat", &transpose_mat<double>, py::arg("A"));
    tns.def("trace_mat", &trace_mat<float>, py::arg("A"));
    tns.def("trace_mat", &trace_mat<double>, py::arg("A"));
    tns.def("mul_vec", &mul_vec<float>, py::arg("A"), py::arg("x"));
    tns.def("mul_vec", &mul_vec<double>, py::arg("A"), py::arg("x"));
    tns.def("inverse_mat", &inverse_mat<float>, py::arg("A"));
    tns.def("inverse_mat", &inverse_mat<double>, py::arg("A"));
    tns.def("det_mat", &det_mat<float>, py::arg("A"));
    tns.def("det_mat", &det_mat<double>, py::arg("A"));
    tns.def("rank_mat", &rank_mat<float>, py::arg("A"));
    tns.def("rank_mat", &rank_mat<double>, py::arg("A"));

    tns.def("gauss_solve", &gauss_solve<float>, py::arg("A"), py::arg("b"));
    tns.def("gauss_solve", &gauss_solve<double>, py::arg("A"), py::arg("b"));
    tns.def("jacobi_solve", &jacobi_solve<float>, py::arg("A"), py::arg("b"),
            py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000);
    tns.def("jacobi_solve", &jacobi_solve<double>, py::arg("A"), py::arg("b"),
            py::arg("tol") = 1e-6, py::arg("max_iter") = 1000);
    tns.def("row_echelon",
            [](Matrix<float> &A, float eps) { tensorium::row_echelon<float>(A, nullptr, eps); },
            py::arg("A"), py::arg("eps") = 1e-12f);
    tns.def("row_echelon",
            [](Matrix<float> &A, Vector<float> &b, float eps) {
                tensorium::row_echelon<float>(A, &b, eps);
            },
            py::arg("A"), py::arg("b"), py::arg("eps") = 1e-12f);
    tns.def("row_echelon",
            [](Matrix<double> &A, double eps) { tensorium::row_echelon<double>(A, nullptr, eps); },
            py::arg("A"), py::arg("eps") = 1e-12);
    tns.def("row_echelon",
            [](Matrix<double> &A, Vector<double> &b, double eps) {
                tensorium::row_echelon<double>(A, &b, eps);
            },
            py::arg("A"), py::arg("b"), py::arg("eps") = 1e-12);

    tns.def("tensor_product",
            [](const Tensor<double, 2> &A, const Tensor<double, 2> &B) { return mul_tensor(A, B); },
            py::arg("A"), py::arg("B"));

    tns.def("contract_tensor", [](const Tensor<double, 4> &T, size_t i, size_t j) {
        if (i >= 4 || j >= 4 || i == j)
            throw std::invalid_argument("Invalid contraction indices");
        if (i > j)
            std::swap(i, j);

        if (i == 0 && j == 1)
            return contract_tensor<0, 1>(T);
        if (i == 0 && j == 2)
            return contract_tensor<0, 2>(T);
        if (i == 0 && j == 3)
            return contract_tensor<0, 3>(T);
        if (i == 1 && j == 2)
            return contract_tensor<1, 2>(T);
        if (i == 1 && j == 3)
            return contract_tensor<1, 3>(T);
        return contract_tensor<2, 3>(T);
    });
}
