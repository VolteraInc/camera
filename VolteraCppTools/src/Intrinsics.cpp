#include "VolteraCalibrationFitting/Intrinsics.h"
#include "VolteraCalibrationFitting/ReprojectionError.h"
#include <ceres/ceres.h>
#include <iostream>

void voltera::runIntrinsics(const std::vector<std::vector<double>> &data,
                            double *cam_matrix, double *distortion,
                            double *extrinsics, bool fix_cam,
                            bool fix_extrinsics) {

  ceres::Problem problem;
  for (const auto &point : data) {

    // for (const auto &point : data) {
    if (point.size() != 5) {
      std::cerr << "Invalid point in input, skipping..." << std::endl;
      continue;
    }
    ceres::CostFunction *cost_function = voltera::ReprojectionError::Create(
        point[0], point[1], point[2], point[3], point[4]);
    // double res[2];
    // voltera::ReprojectionError temp_cost_function(point[0], point[1],
    // point[2],
    //                                              point[3], point[4]);
    // temp_cost_function(cam_matrix, distortion, extrinsics, res);
    // std::cout << point[0] << ", " << point[1] << ", " << res[0] << "," <<
    // res[1] << std::endl;
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, cam_matrix,
                             distortion, extrinsics);
  }

  /*
    problem.SetParameterLowerBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_X, -0.01);
    problem.SetParameterUpperBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_X, 0.01);
    problem.SetParameterLowerBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_Y, -0.01);
    problem.SetParameterUpperBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_Y, 0.01);
    problem.SetParameterLowerBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_Z, -0.01);
    problem.SetParameterUpperBound(
        extrinsics, voltera::ReprojectionError::ANGLE_AXIS_Z, 0.01);

  problem.SetParameterLowerBound(extrinsics, voltera::ReprojectionError::X_OFF,
                                 stage_offset[0] - 0.005);
  problem.SetParameterUpperBound(extrinsics, voltera::ReprojectionError::X_OFF,
                                 stage_offset[0] + 0.005);
  problem.SetParameterLowerBound(extrinsics, voltera::ReprojectionError::Y_OFF,
                                 stage_offset[1] - 0.005);
  problem.SetParameterUpperBound(extrinsics, voltera::ReprojectionError::Y_OFF,
                                 stage_offset[1] + 0.005);
  problem.SetParameterLowerBound(extrinsics, voltera::ReprojectionError::Z_OFF,
                                 stage_offset[2] - 0.01);
  problem.SetParameterUpperBound(extrinsics, voltera::ReprojectionError::Z_OFF,
                                 stage_offset[2] + 0.01);
  */
  problem.SetParameterLowerBound(cam_matrix, voltera::ReprojectionError::FX,
                                 cam_matrix[voltera::ReprojectionError::FX] -
                                     1000);
  problem.SetParameterUpperBound(cam_matrix, voltera::ReprojectionError::FX,
                                 cam_matrix[voltera::ReprojectionError::FX] +
                                     1000);
  problem.SetParameterLowerBound(cam_matrix, voltera::ReprojectionError::FY,
                                 cam_matrix[voltera::ReprojectionError::FY] -
                                     1000);
  problem.SetParameterUpperBound(cam_matrix, voltera::ReprojectionError::FY,
                                 cam_matrix[voltera::ReprojectionError::FY] +
                                     1000);
  problem.SetParameterLowerBound(cam_matrix, voltera::ReprojectionError::CX,
                                 cam_matrix[voltera::ReprojectionError::CX] -
                                     100);
  problem.SetParameterUpperBound(cam_matrix, voltera::ReprojectionError::CX,
                                 cam_matrix[voltera::ReprojectionError::CX] +
                                     100);
  problem.SetParameterLowerBound(cam_matrix, voltera::ReprojectionError::CY,
                                 cam_matrix[voltera::ReprojectionError::CY] -
                                     100);
  problem.SetParameterUpperBound(cam_matrix, voltera::ReprojectionError::CY,
                                 cam_matrix[voltera::ReprojectionError::CY] +
                                     100);

  if (fix_cam) {
    problem.SetParameterBlockConstant(cam_matrix);
    problem.SetParameterBlockConstant(distortion);
  }
  if (fix_extrinsics) {
    problem.SetParameterBlockConstant(extrinsics);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 2000;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n\n";

  std::cout << "Camera Matrix" << std::endl;
  for (auto i = 0; i < voltera::ReprojectionError::SIZE_CAMMATRIX; ++i) {
    std::cout << cam_matrix[i] << ", ";
  }
  std::cout << std::endl << std::endl;

  std::cout << "Distortion" << std::endl;
  for (auto i = 0; i < voltera::ReprojectionError::SIZE_DISTORTION; ++i) {
    std::cout << distortion[i] << ", ";
  }
  std::cout << std::endl << std::endl;

  std::cout << "Extrinsics" << std::endl;
  for (auto i = 0; i < voltera::ReprojectionError::SIZE_EXTRINSICS; ++i) {
    std::cout << extrinsics[i] << ", ";
  }
  std::cout << std::endl << std::endl;
}