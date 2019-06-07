#ifndef INTRINSICS_H
#define INTRINSICS_H

#include <vector>

namespace voltera {

void runIntrinsics(const std::vector<std::vector<double>> &data,
                   std::vector <double> & cam_matrix, 
                   std::vector <double> & distortion, 
                   std::vector <double> & extrinsics,
                   bool fix_cam = false, 
                   bool fix_extrinsics = false);
}

#endif