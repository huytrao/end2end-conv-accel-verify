#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Để chuyển đổi std::vector tự động
#include "cnn_dut.h"

namespace py = pybind11;

PYBIND11_MODULE(cnn_dut_module, m) {
    m.doc() = "Pybind11 wrapper for our CNN DUT";

    py::class_<CNN_DUT>(m, "CNN_DUT")
        .def(py::init<
             const Tensor4D_8&, const Vector32&,
             const Tensor4D_8&, const Vector32&,
             const std::vector<std::vector<int8_t>>&, const Vector32&,
             const std::vector<std::vector<int8_t>>&, const Vector32&>())
        .def("predict", &CNN_DUT::predict, "Run inference on a single image");
}