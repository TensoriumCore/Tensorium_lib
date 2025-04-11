
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Morpheus/Vector.hpp"
#include "Morpheus/Functional.hpp"
#include <sstream>
namespace py = pybind11;
using namespace morpheus;

PYBIND11_MODULE(morpheus, m) {
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


	m.def("add", static_cast<Vector<float> (*)(const Vector<float>&, const Vector<float>&)>(&add<float>), "Add two vectors");
	m.def("sub", static_cast<Vector<float> (*)(const Vector<float>&, const Vector<float>&)>(&sub<float>), "Subtract two vectors");
	m.def("scl", static_cast<Vector<float> (*)(const Vector<float>&, float)>(&scl<float>), "Scale a vector");

}

