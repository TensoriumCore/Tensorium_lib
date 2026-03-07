#include "../includes/Tensorium/Tensorium.hpp"
#include <algorithm>
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

} // namespace

PYBIND11_MODULE(tensorium, m) {
    py::module_ tns = m.def_submodule("tns", "High-performance math operations");

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
