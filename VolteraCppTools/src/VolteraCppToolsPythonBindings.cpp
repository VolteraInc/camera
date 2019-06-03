#include <pybind11/pybind11.h>

#include "VolteraCppTools/Intrinsics.h"
#include "VolteraCppTools/LaserExtrinsics.h"

PYBIND11_MODULE(VolteraCppTools_Python, m) {
  m.doc() =
      "c++ plugin for performing certain parts of the Voltera camera/laser "
      "calibration and operation in c++ but called from python.";

  m.def("calculate_intrinsics", &voltera::runIntrinsics,
        "A function that takes a list of lists of data points and solves for "
        "the camera matrix, distortion, and the position of the stage.");

  m.def("calculate_laser_plane", &voltera::runLaserExtrinsics,
        "A function that takes a list of laser points and heights, as well as "
        "the camera matrix, distortion and extrinsics from the previous "
        "intrinsics fittings, and solves for the laser plane.");
}