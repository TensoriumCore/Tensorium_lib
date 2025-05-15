#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Morpheus/Morpheus.hpp"
#include <sstream>
#include <iostream>
#include <omp.h>

namespace py = pybind11;
using namespace morpheus;

// === Conversions NumPy -> Morpheus ===
morpheus::Vector<double> numpy_to_vector(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1D array for Vector");
    return morpheus::Vector<double>(std::vector<double>((double*)buf.ptr, (double*)buf.ptr + buf.shape[0]));
}

morpheus::Tensor<double, 2> numpy_to_tensor2(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected 2D array for Tensor<2>");
    std::array<size_t, 2> dims = { static_cast<size_t>(buf.shape[0]), static_cast<size_t>(buf.shape[1]) };
    morpheus::Tensor<double, 2> tensor(dims);
    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            tensor(i, j) = *((double*)buf.ptr + i * dims[1] + j);
    return tensor;
}

PYBIND11_MODULE(morpheus, m) {
    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads\n";

    py::module_ morph = m.def_submodule("morph", "High-performance math operations");

	py::class_<morpheus::Tensor<double, 2>>(morph, "Tensor2d")
		.def("print", &morpheus::Tensor<double, 2>::print)
		.def("contract", [](const morpheus::Tensor<double, 2>& T, size_t i, size_t j) {
			if (i >= 2 || j >= 2 || i == j)
				throw std::invalid_argument("Invalid contraction indices");
				return contract_tensor<0,1>(T); // Remplace 0,1 par i,j si tu fais un dispatch dynamique
			}, py::arg("i"), py::arg("j"), "Contract tensor along axes (i, j)");

    py::class_<morpheus::Tensor<double, 3>>(morph, "Tensor3d")
        .def("print", &morpheus::Tensor<double, 3>::print);

    py::class_<morpheus::Tensor<double, 4>>(morph, "Tensor4d")
        .def("print", &morpheus::Tensor<double, 4>::print)
        .def("__getitem__", [](const morpheus::Tensor<double, 4>& T, std::tuple<size_t, size_t, size_t, size_t> idx) {
            return T(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx), std::get<3>(idx));
        })
        .def("__setitem__", [](morpheus::Tensor<double, 4>& T, std::tuple<size_t, size_t, size_t, size_t> idx, double val) {
            T(std::get<0>(idx), std::get<1>(idx), std::get<2>(idx), std::get<3>(idx)) = val;
        });

    py::class_<morpheus_RG::RiemannTensor<double>>(morph, "Riemann")
        .def_static("print_componentwise", &morpheus_RG::RiemannTensor<double>::print_componentwise,
                    py::arg("R"), py::arg("threshold") = 1e-12,
                    "Print Riemann tensor components with optional threshold");

    py::class_<morpheus_RG::ChristoffelSym<double>>(morph, "Christoffel")
        .def("print", &morpheus_RG::ChristoffelSym<double>::print);

    py::class_<morpheus_RG::Metric<double>>(morph, "Metric")
        .def(py::init<const std::string&, double, double>())
        .def("__call__", [](morpheus_RG::Metric<double>& metric, const Vector<double>& X) {
            morpheus::Tensor<double, 2> g({4, 4});
            metric(X, g);
            return g;
        });
    morph.def("compute_christoffel", [](const Vector<double>& X,
                                         const morpheus::Tensor<double, 2>& g,
                                         const morpheus::Tensor<double, 2>& ginv,
                                         const std::string& metric_type,
                                         double M, double a) {
        auto metric = morpheus_RG::Metric<double>(metric_type, M, a);
        return morpheus::compute_christoffel(X, 1e-5, g, ginv, metric);
    });
    morph.def("compute_riemann_tensor", [](const Vector<double>& X,
                                            const std::string& metric_type,
                                            double M, double a) {
        auto metric = morpheus_RG::Metric<double>(metric_type, M, a);
        return morpheus::compute_riemann_tensor<double>(X, 1e-5, metric);
    });
	morph.def("contract_riemann_to_ricci", 
			[](const morpheus::Tensor<double, 4>& R, const morpheus::Tensor<double, 2>& ginv) {
			return morpheus::contract_riemann_to_ricci(R, ginv);
			},
			py::arg("riemann"), py::arg("g_inv"),
			"Contract a Riemann tensor R_{ρσμν} to the Ricci tensor R_{μν} using g^{ρσ}"
			);
	morph.def("compute_ricci_scalar",
			[](const morpheus::Tensor<double, 2>& Ricci, const morpheus::Tensor<double, 2>& ginv) {
			return morpheus::compute_ricci_scalar(Ricci, ginv);
			},
			py::arg("ricci"), py::arg("g_inv"),
			"Compute the Ricci scalar R from the Ricci tensor R_{μν} and the inverse metric g^{μν}"
			);

	// === Ricci Tensor display ===
	morph.def("print_ricci_tensor", &morpheus::print_ricci_tensor<double>, 
			py::arg("R"), "Print components of the Ricci tensor");


	morph.def("print_ricci_scalar",
			&morpheus::print_ricci_scalar<double>,
			py::arg("Ricci_tensor"), py::arg("g_inv"),
			"Print the Ricci scalar from Ricci tensor and inverse metric");


    morph.def("inv_mat_tensor", &morpheus::inv_mat_tensor<double>, "Inverse of a 2D tensor");

    py::class_<Vector<float>>(m, "Vector")
        .def(py::init<std::vector<float>>())
        .def("print", &Vector<float>::print)
        .def("__len__", &Vector<float>::size)
        .def("__getitem__", [](const Vector<float>& v, size_t i) { return v[i]; })
        .def("__repr__", [](const Vector<float>& v) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                oss << v[i];
                if (i + 1 < v.size()) oss << ", ";
            }
            oss << "]";
            return oss.str();
        });

    py::class_<Vector<double>>(m, "Vectord")
        .def(py::init<std::vector<double>>())
        .def("__len__", &Vector<double>::size)
        .def("__getitem__", [](const Vector<double>& v, size_t i) { return v[i]; })
        .def("__repr__", [](const Vector<double>& v) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                oss << v[i];
                if (i + 1 < v.size()) oss << ", ";
            }
            oss << "]";
            return oss.str();
        });

    py::class_<Matrix<float>>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("print", &Matrix<float>::print)
        .def("__getitem__", [](const Matrix<float>& m, std::pair<size_t, size_t> idx) {
            return m(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix<float>& m, std::pair<size_t, size_t> idx, float val) {
            m(idx.first, idx.second) = val;
        })
        .def("rows", [](const Matrix<float>& m) { return m.rows; })
        .def("cols", [](const Matrix<float>& m) { return m.cols; })
        .def("fill", [](Matrix<float>& m, const std::vector<std::vector<float>>& values) {
            if (values.size() != m.rows || values[0].size() != m.cols)
                throw std::runtime_error("Shape mismatch in fill()");
            for (size_t i = 0; i < m.rows; ++i)
                for (size_t j = 0; j < m.cols; ++j)
                    m(i, j) = values[i][j];
        })
        .def("__repr__", [](const Matrix<float>& m) {
            std::ostringstream oss;
            oss << "[\n";
            for (size_t i = 0; i < m.rows; ++i) {
                oss << "  [";
                for (size_t j = 0; j < m.cols; ++j) {
                    oss << m(i, j);
                    if (j + 1 < m.cols) oss << ", ";
                }
                oss << "]";
                if (i + 1 < m.rows) oss << ",";
                oss << "\n";
            }
            oss << "]";
            return oss.str();
        });

    // Math backend
    morph.def("add_vec", &add_vec<float>, "Add two vectors");
    morph.def("sub_vec", &sub_vec<float>, "Subtract two vectors");
    morph.def("scl_vec", &scl_vec<float>, "Scale a vector by scalar");
    morph.def("dot_vec", &dot_vec<float>, "Dot product between two vectors");
    morph.def("norm_1", &norm1_vec<float>, "L1 norm of vector");
    morph.def("norm_2", &norm2_vec<float>, "L2 norm of vector");
    morph.def("norm_inf", &normInf_vec<float>, "Infinity norm of vector");
    morph.def("cosine", &cosine_vec<float>, "Cosine similarity between vectors");
    morph.def("lerp", &lerp_vec<float>, "Linear interpolation between two vectors");
    morph.def("linear_comb", &morpheus::linear_combination_vec<float>, "Linear combination of vectors");
    morph.def("cross", &cross_vec<float>, "Cross product (only defined for 3D vectors)");

	morph.def("tensor_product", [](const morpheus::Tensor<double, 2>& A,
				const morpheus::Tensor<double, 2>& B) {
			return mul_tensor(A, B);
			}, py::arg("A"), py::arg("B"), "Tensor (outer) product of two tensors");
	morph.def("contract_tensor", [](const morpheus::Tensor<double, 4>& T, size_t i, size_t j) {
			if (i >= 4 || j >= 4 || i == j)
			throw std::invalid_argument("Invalid contraction indices");

			if (i == 0 && j == 1) return contract_tensor<0,1>(T);
			if (i == 0 && j == 2) return contract_tensor<0,2>(T);
			if (i == 0 && j == 3) return contract_tensor<0,3>(T);
			if (i == 1 && j == 2) return contract_tensor<1,2>(T);
			if (i == 1 && j == 3) return contract_tensor<1,3>(T);
			if (i == 2 && j == 3) return contract_tensor<2,3>(T);

			throw std::invalid_argument("Unsupported contraction indices");
			});
	morph.def("tensor_product", [](const morpheus::Tensor<double, 2>& A,
				const morpheus::Tensor<double, 2>& B) {
			return mul_tensor(A, B);
			}, py::arg("A"), py::arg("B"), "Tensor (outer) product of two tensors");
	morph.def("add_mat", &add_mat<float>, "Add two matrices");
    morph.def("sub_mat", &sub_mat<float>, "Subtract two matrices");
    morph.def("scl_mat", &scl_mat<float>, "Scale matrix by scalar");
    morph.def("mul", &mul_mat<float>, "Multiply two matrices");
    morph.def("transpose_mat", &transpose_mat<float>, "Transpose a matrix");
    morph.def("trace_mat", &trace_mat<float>, "Trace of a matrix");
    morph.def("mul_vec", &mul_vec<float>, "Multiply matrix by vector");
    morph.def("inverse_mat", &inverse_mat<float>, "Inverse of a matrix");
    morph.def("det_mat", &det_mat<float>, "Determinant of a matrix");
    morph.def("rank_mat", &rank_mat<float>, "Rank of a matrix");

    morph.def("gauss_solve", &gauss_solve<float>, "Solve linear system with Gauss elimination");
    morph.def("jacobi_solve", &jacobi_solve<float>, py::arg("A"), py::arg("b"), py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000, "Jacobi iterative method");
	morph.def("row_echelon", &morpheus::row_echelon<float>,
          py::arg("A"), py::arg("b") = nullptr, py::arg("eps") = 1e-12f,
          "Convert matrix A (and optionally b) to row echelon form");

}
