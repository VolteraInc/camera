#ifndef PREVIEW_POSITION_H
#define PREVIEW_POSITION_H

#include <vector>

namespace voltera {

void previewIntrinsics(const std::vector<std::vector<double>> &data,
                       const std::vector<double> &cam_matrix,
                       const std::vector<double> &distortion,
                       const std::vector<double> &extrinsics,
                       double image_width = 3280, double image_height = 2464);
}

#endif
