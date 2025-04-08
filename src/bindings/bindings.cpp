#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Global variables for storing interpolation data and corresponding points.
static std::vector<double> g_interpolationData;
static std::vector<std::vector<double>> g_interpolationPoints;

// Function to set the interpolation data and points from Python.
void set_interpolation_data(const std::vector<double>& data, const std::vector<std::vector<double>>& points) {
    g_interpolationData = data;
    g_interpolationPoints = points;
}

// Function to retrieve the stored interpolation data.
std::vector<double> get_interpolation_data() {
    return g_interpolationData;
}

PYBIND11_MODULE(feelpp_interface, m) {
    m.doc() = "Bindings for interfacing PINN data with Feel++ FEM solver";
    m.def("set_interpolation_data", &set_interpolation_data, "Set interpolation data and points",
          py::arg("data"), py::arg("points"));
    m.def("get_interpolation_data", &get_interpolation_data, "Get interpolation data");
}
