#ifndef LASER_EXTRINSICS_H
#define LASER_EXTRINSICS_H

#include <vector>

namespace voltera {

void runLaserExtrinsics(const std::vector<std::vector<double>> &data,
                        const double *cam_matrix, const double *distortion,
                        const double *extrinsics, double *laser_plane);
}

#endif