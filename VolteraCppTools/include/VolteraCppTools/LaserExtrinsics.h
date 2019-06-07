#ifndef LASER_EXTRINSICS_H
#define LASER_EXTRINSICS_H

#include <vector>

namespace voltera {

void runLaserExtrinsics(const std::vector<std::vector<double>> &data,
                        const std::vector<double> &cam_matrix, 
                        const std::vector<double> & distortion,
                        const std::vector<double> &extrinsics, 
                        std::vector <double> &laser_plane);
}

#endif