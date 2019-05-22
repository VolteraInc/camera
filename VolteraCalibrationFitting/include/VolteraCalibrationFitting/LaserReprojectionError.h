#ifndef LASER_REPROJECTION_ERROR_H
#define LASER_REPROJECTION_ERROR_H

/**
 * Laser_ReprojectionError.h
 *
 * Ryan Wicks
 * 21 May 2019
 * Copyright Voltera Inc., 2019
 *
 * This class represents the residual of a point on the laser plane.
 *
 */

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include <vector>

namespace voltera {

struct LaserReprojectionError {
  LaserReprojectionError(double observed_i, double observed_j, double height,
                         const double *camera_matrix, const double *distortion,
                         const double *initial_position)
      : m_int_x(0.0), m_int_y(0.0), m_int_z(0.0), is_valid(true) {

    // initial_position[0,1,2] are the angle-axis rotation.
    double temp_plane_point[Point3D::SIZE_POINT3D]{0.0, 0.0, height};
    double temp_plane_normal[Point3D::SIZE_POINT3D]{0.0, 0.0, -1.0};

    double plane_point[Point3D::SIZE_POINT3D]{0.0, 0.0, 0.0};
    double plane_normal[Point3D::SIZE_POINT3D]{0.0, 0.0, 0.0};

    /*****
     * Transform the plane from world co-ords to cam co-ords with the initial
     * position.
     */
    ceres::AngleAxisRotatePoint(initial_position, temp_plane_point,
                                plane_point);
    ceres::AngleAxisRotatePoint(initial_position, temp_plane_normal,
                                plane_normal);

    double d(-(plane_normal[Point3D::X] * plane_point[Point3D::X] +
               plane_normal[Point3D::Y] * plane_point[Point3D::Y] +
               plane_normal[Point3D::Z] * plane_point[Point3D::Z]));

    // initial_position[3,4,5] are the translation.
    plane_point[Point3D::X] += initial_position[Extrinsics::X_OFF];
    plane_point[Point3D::Y] += initial_position[Extrinsics::Y_OFF];
    plane_point[Point3D::Z] += initial_position[Extrinsics::Z_OFF];

    // Undistort the 2d sensor point so it can be projected onto the plane.
    double fx = camera_matrix[CamMatrix::FX];
    double fy = camera_matrix[CamMatrix::FY];
    double cx = camera_matrix[CamMatrix::CX];
    double cy = camera_matrix[CamMatrix::CY];
    double k1 = distortion[Distortion::K1];
    double k2 = distortion[Distortion::K2];
    double p1 = distortion[Distortion::P1];
    double p2 = distortion[Distortion::P2];
    double k3 = distortion[Distortion::K3];

    cv::Point2d sensor_point(observed_i, observed_j);
    cv::Point2d out_point(0.0, 0.0);

    // Project point onto expected plane.
    cv::Mat cam_matrix =
        (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    cv::undistortPoints(sensor_point, out_point, cam_matrix, dist,
                        cv::noArray(), cam_matrix);

    double denom = plane_normal[Point3D::X] * out_point.x +
                   plane_normal[Point3D::Y] * out_point.y +
                   plane_normal[Point3D::Z] * 1.0;

    // CHECK denominator != 0
    if (std::abs(denom) < 0.0000001) {
      is_valid = false;
      return;
    }

    double t = -d / denom;

    m_int_x = t * out_point.x;
    m_int_y = t * out_point.y;
    m_int_z = t * 1.0;
  };

  // array indexes for a 3D point.
  enum Point3D : int { X, Y, Z, SIZE_POINT3D };

  // array indexes for different extrinsics parameters
  enum Extrinsics : int {
    ANGLE_AXIS_X,
    ANGLE_AXIS_Y,
    ANGLE_AXIS_Z,
    X_OFF,
    Y_OFF,
    Z_OFF,
    SIZE_EXTRINSICS
  };

  // array indexes for different camera matrix parameters.
  enum CamMatrix : int { FX, FY, CX, CY, SIZE_CAMMATRIX };

  // array indexes of the distortion
  enum Distortion : int { K1, K2, P1, P2, K3, SIZE_DISTORTION };

  // array indexes of the laser plane
  enum LaserPlane : int { RX, RY, RZ, HEIGHT, SIZE_LASER_PLANE };

  // array indexes of residuals
  enum Residual : int { SIZE_RESIDUAL };

  template <typename T>
  bool operator()(const T *const laser_plane, T *residual) const {
    /**
     * Generate the laser plane.
     */
    T temp_laser_plane_normal[SIZE_POINT3D]{
        static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1)};
    T laser_plane_normal[SIZE_POINT3D]{static_cast<T>(0), static_cast<T>(0),
                                       static_cast<T>(0)};
    T laser_plane_point[SIZE_POINT3D]{static_cast<T>(0), static_cast<T>(0),
                                      static_cast<T>(laser_plane[HEIGHT])};

    ceres::AngleAxisRotatePoint(laser_plane, temp_laser_plane_normal,
                                laser_plane_normal);

    T d(-(laser_plane_normal[Point3D::Z] * laser_plane_point[Point3D::Z]));

    *residual = laser_plane_normal[Point3D::X] * m_int_x +
                laser_plane_normal[Point3D::Y] * m_int_y +
                laser_plane_normal[Point3D::Z] * m_int_z +
                d; // Point plane distance.

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code. Returns null pointer is the inputs are invalid
  static ceres::CostFunction *Create(const double observed_i, double observed_j,
                                     double height, const double *camera_matrix,
                                     const double *distortion,
                                     const double *initial_position) {
    LaserReprojectionError *reprojector(new LaserReprojectionError(
        observed_i, observed_j, height, camera_matrix, distortion,
        initial_position));
    if (reprojector->is_valid) {
      return new ceres::AutoDiffCostFunction<LaserReprojectionError,
                                             SIZE_RESIDUAL, SIZE_LASER_PLANE>(
          reprojector);
    } else {
      delete reprojector;
      return nullptr;
    }
  }

  double m_int_x;
  double m_int_y;
  double m_int_z;

  bool is_valid;
}; // namespace voltera

} // namespace voltera

#endif // REPROJECTION_ERROR_H