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

  std::cout << "Laser Plane" << std::endl;
  for (auto i = 0; i < voltera::LaserReprojectionError::SIZE_LASER_PLANE; ++i) {
    std::cout << laser_plane[i] << ", ";
  }
  std::cout << std::endl << std::endl;

  std::cout << std::endl << std::endl;
}
} // namespace voltera