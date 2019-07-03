#ifndef ANALYSIS_IMAGE_H
#define ANALYSIS_IMAGE_H

#include <Eigen/Dense>

namespace voltera {

Eigen::VectorXd
findImagePeaks(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> &image);
}

#endif