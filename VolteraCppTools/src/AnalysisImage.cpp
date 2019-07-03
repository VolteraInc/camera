#include "VolteraCppTools/AnalysisImage.h"

namespace voltera {

Eigen::VectorXd
findImagePeaks(Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> &image) {

  Eigen::VectorXd return_vector(image.cols());

  for (int index = 0; index < image.cols(); ++index) {
    double max_value(0.0);
    int max_index(0);

    max_value = image.col(index).maxCoeff(&max_index);

    // Put some subpixel logic here

    return_vector(index) = static_cast<double>(max_index);
  }

  return return_vector;
}

} // namespace voltera