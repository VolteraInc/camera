#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "VolteraCppTools/AnalysisImage.h"
#include "VolteraCppTools/Intrinsics.h"
#include "VolteraCppTools/LaserExtrinsics.h"

namespace py = pybind11;

PYBIND11_MODULE(VolteraCppTools_Python, m) {
  m.doc() =
      "c++ plugin for performing certain parts of the Voltera camera/laser "
      "calibration and operation in c++ but called from python.";

  m.def("calculate_intrinsics", &voltera::runIntrinsics,
        "A function that takes a list of lists of data points and solves for "
        "the camera matrix, distortion, and the position of the stage.",
        py::arg("input_points"), py::arg("cam_matrix"), py::arg("distortion"),
        py::arg("extrinsics"), py::arg("fix_cam") = false,
        py::arg("fix_extrinsics") = false);

  m.def("calculate_laser_plane", &voltera::runLaserExtrinsics,
        "A function that takes a list of laser points and heights, as well as "
        "the camera matrix, distortion and extrinsics from the previous "
        "intrinsics fittings, and solves for the laser plane.",
        py::arg("data_points"), py::arg("cam_matrix"), py::arg("distortion"),
        py::arg("extrinsics"), py::arg("laser_plane"));

  m.def("analyze_laser_image", &voltera::findImagePeaks,
        "This function takes a single channel image and returns the peak "
        "positions of each row.",
        py::arg("image_array"));
}