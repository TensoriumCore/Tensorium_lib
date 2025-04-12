#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Morpheus/Vector.hpp"
#include "Morpheus/Matrix.hpp"
#include "Morpheus/Functional.hpp"
#include <sstream>
#include <iostream>
#include <omp.h>

namespace py = pybind11;
using namespace morpheus;

PYBIND11_MODULE(morpheus, m) {
	std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads\n";

	// === Vector<float> ===
	py::class_<Vector<float>>(m, "Vector")
		.def(py::init<std::vector<float>>())
		.def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> array) {
			auto buf = array.request();
			if (buf.ndim != 1)
				throw std::runtime_error("Vector: NumPy array must be 1-dimensional");
			std::vector<float> data(reinterpret_cast<float*>(buf.ptr),
			                        reinterpret_cast<float*>(buf.ptr) + buf.shape[0]);
			return Vector<float>(data);
		}))
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

	// === Matrix<float> ===
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



		py::module_ morph = m.def_submodule("morph", "High-performance math operations");

		// === Vector ops ===
		morph.def("add", &add<float>, "Add two vectors");
		morph.def("sub", &sub<float>, "Subtract two vectors");
		morph.def("scl", &scl<float>, "Scale a vector by scalar");
		morph.def("dot", &dot<float>, "Dot product between two vectors");
		morph.def("norm_1", &norm1<float>, "L1 norm of vector");
		morph.def("norm_2", &norm2<float>, "L2 norm of vector");
		morph.def("norm_inf", &normInf<float>, "Infinity norm of vector");
		morph.def("cosine", &cosine<float>, "Cosine similarity between vectors");
		morph.def("lerp", &lerp<float>, "Linear interpolation between two vectors");
		morph.def("linear_comb", &morpheus::linear_combination<float>, "Linear combination of vectors");
		morph.def("cross", &cross<float>, "Cross product (only defined for 3D vectors)");

		// === Matrix ops ===
		morph.def("add_mat", &add_mat<float>, "Add two matrices");
		morph.def("sub_mat", &sub_mat<float>, "Subtract two matrices");
		morph.def("scl_mat", &scl_mat<float>, "Scale matrix by scalar");
		morph.def("mul", &mul_mat<float>, "Multiply two matrices");
}
