"""
This set of tests will be used to test the approach of using at translation stage to image single points to figure out camera intrinsics and disstrotion
"""
import pytest
import random
import numpy as np
import scipy.optimize as so
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.transform import Transform
import volteracamera.intrinsics.stage_calibration as sc

initial_camera_matrix = np.array([[5257.0, 0.0, 640.0], [0.0, 5257.0, 360.0], [0.0, 0.0, 1.0]])
initial_distortion = np.array([0.1, -0.05, 0.05, -0.05, 0.002])
initial_R = np.array([0.01, -0.001, 0.05])
initial_t = np.array([0.1, -0.1, 0.015])

def generate_dataset():
    """
    Generate the 3d points, translations of the stage to get there and 2d point correspondences. 
    """
    initial_point = np.array([0, 0, 0])
    initial_U = Undistort(initial_camera_matrix, initial_distortion)
    initial_T = Transform (rotation=initial_R, translation=initial_t)

    #distance: 1.5 cm, FOV Width: (0.17920477879410118, 0.10080268807168191) cm
    #distance: 2 cm, FOV Width: (0.23893970505880158, 0.13440358409557587) cm
    #distance: 2.5 cm, FOV Width: (0.29867463132350197, 0.16800448011946983) cm
    distances = np.array([1.5, 2.0, 2.5]) - initial_t[2]
    width_x = [0.175, 0.23, 0.29]
    width_y = [0.10, 0.13, 0.16]

    translations = []
    for d, w, h in zip(distances, width_x, width_y):
        for x in np.arange(-w, w, w/10):
            for y in np.arange(-h, h, h/10):
                translations.append(np.array([x, y, d]))
    
    points_3d = []
    points_2d = []
    trans_out = []
    for t in translations:
        T = Transform(translation=t)
        point_3d = initial_T.transform_point(T.transform_point(initial_point))
        point_2d = initial_U.project_point_with_distortion(point_3d)

        if np.abs(point_2d[0]) > initial_camera_matrix[0, 2] or np.abs(point_2d[1]) > initial_camera_matrix[1, 2]:
            continue
        
        points_3d.append(point_3d)
        points_2d.append(point_2d)
        trans_out.append(t)

    return (points_3d, trans_out, points_2d)

def test_unpack_parameters():
    """
    Test the function that unpacks the residuals
    """
    res = [100, 200, 300, 400, 1, 2, 3, 4, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    cam_matrix, distortion, rvec, tvec = sc.unpack_params(res)
    np.testing.assert_array_equal(cam_matrix, np.array([[100, 0, 300], [0, 200, 400], [0, 0, 1]]))
    np.testing.assert_array_equal(distortion, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(rvec, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(tvec, np.array([0.4, 0.5, 0.6]))

def test_get_projected():
    """
    Test the projection function again opencv
    """
    rvec=np.array(initial_R, dtype="float32")
    tvec=np.array(initial_t, dtype="float32")
    undistorter=Undistort(initial_camera_matrix, initial_distortion)
    transformer=Transform(rvec, tvec)
    _, translations, points_2d = generate_dataset()

    for point_2d, point_3d in zip (points_2d, translations):
        point = sc.get_projected(point_3d, undistorter, transformer)
        np.testing.assert_array_almost_equal( point_2d, point, decimal=4 )

def test_residuals():
    """
    Test that the residual function behaves as expected
    """
    _, translations, points_2d_actual = generate_dataset()

    parameters = [initial_camera_matrix[0, 0], initial_camera_matrix[1, 1], initial_camera_matrix[0, 2], initial_camera_matrix[1, 2], 
                  initial_distortion[0], initial_distortion[1], initial_distortion[2], initial_distortion[3], initial_distortion[4],
                  *initial_R, *initial_t]

    np.testing.assert_almost_equal (sc.single_residual(parameters, translations, points_2d_actual), 0.0)

def test_xyz_intrinsics():
    """
    This test generates several hundred 3-d point in the field of view of the camera, and 2-d correspondences with a given set of intrinsics and distortion. 
    These values are then used to test the camera calibration code by minimizing the 2-d correspondences.
    """
    _, translations, points_2d_actual = generate_dataset()

    spread = 0.5
    points_2d_actual = [ [point[0] + random.uniform(-spread, spread), point[1] + random.uniform(-spread, spread)] for point in points_2d_actual ]
    undistortion = sc.calibrate_from_3d_points(translations, points_2d_actual)

    np.testing.assert_array_almost_equal(undistortion.camera_matrix, initial_camera_matrix )
    np.testing.assert_array_almost_equal(undistortion.distortion, initial_distortion)

   

    