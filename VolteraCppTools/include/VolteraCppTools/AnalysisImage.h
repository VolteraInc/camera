#ifndef ANALYSIS_IMAGE_H
#define ANALYSIS_IMAGE_H

#include <Eigen/Dense>

namespace voltera {

Eigen::VectorXd &findImagePeaks(Eigen::Ref<Eigen::MatrixXd> image);
}

#endif