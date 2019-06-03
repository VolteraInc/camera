#include "VolteraCppTools/PreviewResiduals.h"

#include "VolteraCppTools/ReprojectionError.h"
#include <opencv2/opencv.hpp>

namespace voltera {

void previewIntrinsics(const std::vector<std::vector<double>> &data,
                       double *cam_matrix, double *distortion,
                       double *extrinsics, double image_width,
                       double image_height) {

  std::vector<std::vector<double>> residuals;
  residuals.reserve(data.size());
  for (const auto &point : data) {
    if (point.size() != 5) {
      std::cerr << "Invalid point in input, skipping..." << std::endl;
      continue;
    }
    double res[2];
    voltera::ReprojectionError temp_cost_function(point[0], point[1], point[2],
                                                  point[3], point[4]);
    temp_cost_function(cam_matrix, distortion, extrinsics, res);

    std::vector<double> single_residual(
        {point[0], point[1], point[0] - res[0], point[1] - res[1]});

    residuals.push_back(single_residual);
  }

  cv::Mat image(image_height, image_width, CV_8UC3);

  for (const auto &point : residuals) {
    cv::Point image_point(point[0], point[1]);
    cv::Point projected_point(point[2], point[3]);
    // Difference line
    cv::line(image, image_point, projected_point, cv::Scalar(0, 0, 255));
    // original point green
    cv::circle(image, image_point, 3, cv::Scalar(0, 255, 0));
    // original point blue
    cv::circle(image, projected_point, 3, cv::Scalar(255, 0, 0));
  }

  cv::Mat display_image(image_height / 4, image_width / 4, CV_8UC3);
  cv::resize(image, display_image, display_image.size());

  cv::namedWindow("Display window");
  cv::imshow("Display window", display_image);

  cv::waitKey(0);
  cv::destroyAllWindows();
}

} // namespace voltera