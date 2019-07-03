#include "VolteraCppTools/AnalysisImage.h"
#include "test_path.h"
#include "gtest/gtest.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>
loadImage(const std::string &filename) {
  cv::Mat image;

  image = cv::imread(filename, cv::IMREAD_GRAYSCALE); // Read the file

  if (!image.data) // Check for invalid input
  {
    std::cout << "Could not open or find the image" << std::endl;
    throw std::runtime_error("Could not load image " + filename);
  }

  // Map the OpenCV matrix with Eigen:
  // Eigen::Map<
  //     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  // out_m(image.ptr<double>(), image.rows, image.cols);

  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output_matrix(
      image.rows, image.cols);
  cv::cv2eigen(image, output_matrix);

  return output_matrix;
}

void showImage(const std::string &filename,
               Eigen::Ref<Eigen::VectorXd> points) {

  cv::Mat image, out_image;
  image = cv::imread(filename, cv::IMREAD_COLOR); // Read the file

  for (auto i = 0; i < points.size(); ++i) {
    cv::circle(image, cv::Point(i, points[i]), 2, cv::Scalar(255, 0, 0),
               cv::FILLED);
  }

  cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
  cv::resize(image, out_image, cv::Size(600, 600), 0, 0, cv::INTER_CUBIC);
  cv::imshow("image", out_image);
  cv::waitKey(0);
}

// Eigen::VectorXd findImagePeaks(Eigen::Ref<Eigen::MatrixXd> image);

TEST(TestLineFinder, general_test) {
  std::string filename = test_data_path + "laser_image.jpg";
  Eigen::VectorXd out_data;

  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image =
      loadImage(filename);

  out_data = voltera::findImagePeaks(image);

  // showImage(filename, out_data);
  ASSERT_EQ(33, out_data[200]);
  ASSERT_EQ(1283, out_data[600]);
  ASSERT_EQ(1277, out_data[800]);

  ASSERT_TRUE(true);
}