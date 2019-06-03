#ifndef INTRINSICS_H
#define INTRINSICS_H

#include <vector>

namespace voltera {

void runIntrinsics(const std::vector<std::vector<double>> &data,
                   double *cam_matrix, double *distortion, double *extrinsics,
                   bool fix_cam = false, bool fix_extrinsics = false);
}

#endif