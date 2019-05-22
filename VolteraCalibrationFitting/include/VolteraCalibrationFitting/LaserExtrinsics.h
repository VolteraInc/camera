#ifndef LASER_EXTRINSICS_H
#define LASER_EXTRINSICS_H

#include <vector>

namespace voltera {

void runLaserExtrinsics(
    const std::vector<std::vector<std::pair<unsigned int, double>>> &data,
    double *cam_matrix, double *distortion, double *extrinsics,
    double *laser_plane);
}

#endif