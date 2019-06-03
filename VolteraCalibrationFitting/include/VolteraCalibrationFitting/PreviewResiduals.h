#ifndef PREVIEW_POSITION_H
#define PREVIEW_POSITION_H

#include <vector>

namespace voltera {

void previewIntrinsics(const std::vector<std::vector<double>> &data,
                       double *cam_matrix, double *distortion,
                       double *extrinsics, double image_width = 3280,
                       double image_height = 2464);
}

#endif
