#include <pybind11/pybind11.h>

#include "VolteraCalibrationCppTools/Intrisics.h"
#include "VolteraCalibrationCppTools/LaserExtrinsics.h"

PYBIND11_MODULE(voltera_calibration_cpp_tools, m) {
  m.doc() =
      "c++ plugin for performing certain parts of the Voltera camera/laser "
      "calibration and operation in c++ but called from python.";

  // m.def("add", &add, "A function which adds two numbers");
}