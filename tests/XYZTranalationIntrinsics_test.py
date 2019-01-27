"""
This set of tests will be used to test the approach of using at translation stage to image single points to figure out camera intrinsics and disstrotion
"""
import numpy as np
import scipy.optimize as so
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.transform import Transform

def test_xyz_intrinsics():
    """
    This test generates several hundred 3-d point in the field of view of the camera, and 2-d correspondences with a given set of intrinsics and distortion. 
    These values are then used to test the camera calibration code by minimizing the 2-d correspondences.
    """
    initial_point = np.array([0, 0, 0])
    initial_camera_matrix = np.array([[5257, 0, 640], [0, 5257, 360], [0, 0, 1]])
    initial_distortion = np.array([0.1, -0.05, 0.05, -0.05, 0.002])
    initial_R = np.array([0, 0, 0])
    initial_t = np.array([0, 0, 0.015])
    initial_T = 

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
                translations.append([x, y, d])
    
    points_3d = []
    
