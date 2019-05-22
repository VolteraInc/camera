#include "VolteraCalibrationFitting/LaserExtrinsics.h"
#include "VolteraCalibrationFitting/LaserReprojectionError.h"

namespace voltera {

void runLaserExtrinsics(
    const std::vector<std::vector<std::pair<unsigned int, double>>> &data,
    double *cam_matrix, double *distortion, double *extrinsics,
    double *laser_plane) {}
} // namespace voltera