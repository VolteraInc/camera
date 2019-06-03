#include "VolteraCalibrationFitting/LaserExtrinsics.h"
#include "VolteraCalibrationFitting/LaserReprojectionError.h"

namespace voltera {

void runLaserExtrinsics(const std::vector<std::vector<double>> &data,
                        const double *cam_matrix, const double *distortion,
                        const double *extrinsics, double *laser_plane) {

  ceres::Problem problem;
  for (const auto &point : data) {

    // for (const auto &point : data) {
    if (point.size() != 3) {
      std::cerr << "Invalid point in input, skipping..." << std::endl;
      continue;
    }
    ceres::CostFunction *cost_function =
        voltera::LaserReprojectionError::Create(
            point[1], point[2], point[0], cam_matrix, distortion, extrinsics);

    if (cost_function != nullptr) {
      problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                               laser_plane);
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 2000;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n\n";

  /**
   * Generate the laser plane.
   */
  double temp_laser_plane_normal[voltera::LaserReprojectionError::SIZE_POINT3D]{
      0.0, 0.0, -1};
  double laser_plane_normal[voltera::LaserReprojectionError::SIZE_POINT3D]{
      0.0, 0.0, 0.0};
  double laser_plane_point[voltera::LaserReprojectionError::SIZE_POINT3D]{
      0.0, 0.0, laser_plane[voltera::LaserReprojectionError::HEIGHT]};

  ceres::AngleAxisRotatePoint(laser_plane, temp_laser_plane_normal,
                              laser_plane_normal);

  double d(-(laser_plane_normal[voltera::LaserReprojectionError::Point3D::Z] *
             laser_plane_point[voltera::LaserReprojectionError::Point3D::Z]));

  std::cout << "Laser Plane" << std::endl;
  for (auto i = 0; i < voltera::LaserReprojectionError::SIZE_POINT3D; ++i) {
    std::cout << laser_plane_normal[i] << ", ";
  }
  std::cout << d << std::endl << std::endl;

  std::cout << "Laser Point" << std::endl;
  for (auto i = 0; i < voltera::LaserReprojectionError::SIZE_POINT3D; ++i) {
    std::cout << laser_plane_point[i] << ", ";
  }

  std::cout << std::endl << std::endl;
}
} // namespace voltera