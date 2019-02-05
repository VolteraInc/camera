"""
This file contains routines for calibrating a system using a linear stage.
"""
import numpy as np
import scipy.optimize as so
from ..analysis.undistort import Undistort
from ..analysis.transform import Transform

def get_projected(point, undistorter, transformer):
    """
    given a 3d point (measured in stage co-ords), an undistortion class and a transformation class, return the projected 2d point
    """
    return undistorter.project_point_with_distortion(transformer.transform_point(point))

def unpack_params (params: list):
    """
    Function to unpack and return the required parameters from the minimization parameter list.
    params:
    fx, fy, cx, cy, d1, d2, d3, d4, d5, rvec1, rvec2, rvec3, tvec1, tvec2, tvec3

    return cam_matrix, distortion_matrix, rvec, tvec, object_scale
    """
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    distortion = np.asarray(params[4:9])
    rvec = np.asarray (params[9:12])
    tvec = np.asarray (params[12:])
    cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return (cam_matrix, distortion, rvec, tvec)

def point_residuals (params, points_3d, points_2d):
    """
    Get residuals of projected 3d points, compared to the imaged points (two points for each points)
    """
    cam_matrix, distortion, rvec, tvec = unpack_params(params)
    undistort = Undistort(cam_matrix, distortion)
    transform = Transform(rotation=rvec, translation=tvec)
    residuals = np.array([ get_projected(point_3d, undistort, transform) - point_2d for point_3d, point_2d in zip(points_3d, points_2d)])
    return residuals.ravel()

def single_residual (params, points_3d, points_2d):
    """
    Sum of abs value of residuals, used for basin hopping.
    """
    return np.abs(point_residuals(params, points_3d, points_2d)).sum()


def calibrate_from_3d_points ( points_3d, observed_points_2d, camera_matrix = None, distortion = None ):
    """
    Given a set of points in 3d, the observed points in 2d, solve for the camera parameters and inital R and t.
    """
    if not isinstance(camera_matrix, np.ndarray):
        camera_matrix = np.array([[4000, 0, 650], [0, 4000, 350], [0, 0, 1]])
    if not isinstance(distortion, np.ndarray):
        distortion = np.array([0, 0, 0, 0, 0])

    parameters = [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], 
                  distortion[0], distortion[1], distortion[2], distortion[3], distortion[4],
                  0, 0, 0, 0, 0, 0.0001]

    #guess = so.basinhopping(single_residual, parameters, minimizer_kwargs={"args":(points_3d, observed_points_2d)})

    #print("Initial annealing:")
    #if guess.success:
    #    print ("Fit sucessful.")
    #else:
    #    print ("Fit failed.")
    #camera_matrix, distortion, rvecs, tvecs = unpack_params(guess.x)
    #print ("Camera Matrix:")
    #print (camera_matrix)
    #print ("Distortion Matrix")
    #print (distortion)
    #print ("rvec")
    #print(rvecs)
    #print ("tvec")
    #print(tvecs)

    res = so.least_squares(point_residuals, parameters, jac="3-point", x_scale='jac', method='trf', verbose=2, args=(points_3d, observed_points_2d))

    print("Actual Fit:")
    if res.success:
        print ("Fit sucessful.")
    else:
        print ("Fit failed.")
    camera_matrix, distortion, rvecs, tvecs = unpack_params(res.x)

    print ("Camera Matrix:")
    print (camera_matrix)
    print ("Distortion Matrix")
    print (distortion)
    print ("rvec")
    print(rvecs)
    print ("tvec")
    print(tvecs)
    
    return Undistort(camera_matrix, distortion)
