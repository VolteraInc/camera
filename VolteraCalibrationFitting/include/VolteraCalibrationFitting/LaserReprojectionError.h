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

#include <vector>

namespace voltera {

struct LaserReprojectionError {
  ReprojectionError(double observed_i, double observed_j, double height,
                    const double *camera_matrix, const double *distortion,
                    const double *extrinsics)
      : m_i(observed_i), m_j(observed_j), m_height(height){};

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

  // array indexes of residuals
  enum Residual : int { SIZE_RESIDUAL };

  template <typename T>
  bool operator()(const T *const laser_plane, T *residuals) const {
    // initial_position[0,1,2] are the angle-axis rotation.
    T plane_point[Point3D::SIZE_POINT3D];
    T plane_normal[Point3D::SIZE_POINT3D];
    T target_plane_point[Point3D::SIZE_POINT3D]{
        static_cast<T>(0), static_cast<T>(0), static_cast<T>(m_height)};
    T target_plane_normal[Point3D::SIZE_POINT3D]{
        static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1.0)};

    ceres::AngleAxisRotatePoint(initial_position, target_plane_normal,
                                plane_point);
    ceres::AngleAxisRotatePoint(initial_position, target_plane_normal,
                                plane_normal);
    // initial_position[3,4,5] are the translation.
    plane_point[Point3D::X] += initial_position[Extrinsics::X_OFF];
    plane_point[Point3D::Y] += initial_position[Extrinsics::Y_OFF];
    plane_point[Point3D::Z] += initial_position[Extrinsics::Z_OFF];

    // Project the 3d point now relative to the camera onto the sensor
    const T fx = camera_matrix[CamMatrix::FX];
    const T fy = camera_matrix[CamMatrix::FY];
    const T cx = camera_matrix[CamMatrix::CX];
    const T cy = camera_matrix[CamMatrix::CY];

    const T k1 = distortion[Distortion::K1];
    const T k2 = distortion[Distortion::K2];
    const T p1 = distortion[Distortion::P1];
    const T p2 = distortion[Distortion::P2];
    const T k3 = distortion[Distortion::K3];

    // taken from
    // https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    // Convert to homogeneous co-orodinates
    T xp(p[Point3D::X] / p[Point3D::Z]);
    T yp(p[Point3D::Y] / p[Point3D::Z]);

    // Undistort the points (radtan)
    T r2(xp * xp + yp * yp);

    T radial_coeff(1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    T xpp(xp * radial_coeff + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp));
    T ypp(yp * radial_coeff + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp);

    // project the points.
    T out_point[Residual::SIZE_RESIDUAL]{fx * xpp + cx, fy * ypp + cy};

    residuals[Residual::XR] = m_observed_x - out_point[Residual::XR];
    residuals[Residual::YR] = m_observed_y - out_point[Residual::YR];
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_i, double observed_j,
                                     double height, const double *camera_matrix,
                                     const double *distortion,
                                     const double *initial_position) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, SIZE_RESIDUAL,
                                            SIZE_LASER_PLANE>(
        new ReprojectionError(observed_i, observed_j, height, camera_matrix,
                              distortion, initial_postion)));
  }

  double m_observed_x;
  double m_observed_y;
  double m_stage[Point3D::SIZE_POINT3D];
};

} // namespace voltera

#endif // REPROJECTION_ERROR_H