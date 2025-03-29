#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

void distance_2d(std::vector<double> &input, size_t width, size_t height,
                 double resolution, double boundary_value);

auto distance_2d_py(std::vector<double> &input, size_t width, size_t height,
                    double resolution, double boundary_value)
    -> py::array_t<double> {
  distance_2d(input, width, height, resolution, boundary_value);
  return py::array_t<double>(input.size(), input.data());
}

PYBIND11_MODULE(_liblocalization_cpp, m) {
  m.def("distance_2d", &distance_2d_py, "distance_2d");
}
