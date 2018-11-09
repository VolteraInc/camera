"""
Test the intrinsics tools, make sure the results are sensible.
"""
import pytest

from volteracamera.intrinsics.circles_calibration import (generate_symmetric_circle_grid, run_calibration)
from volteracamera.intrinsics.calibration_generation import generate_random_cal_images
from volteracamera.analysis.undistort import Undistort

REL=0.01

def test_intrinsics_calibration():
    """
    Meta test of the entire process
    """
    image, points, _ = generate_symmetric_circle_grid(5, 4, 150)
    images = generate_random_cal_images(image, 30)
    undistort = run_calibration(images, object_points_in=points)

    cx = image.shape[0] / 2
    cy = image.shape[1] / 2
    fx = 250
    fy = 250 
    print (undistort)
    assert cx == pytest.approx(undistort.camera_matrix[0, 2], rel=REL)
    assert cy == pytest.approx(undistort.camera_matrix[1, 2], rel=REL)
    assert fx == pytest.approx(undistort.camera_matrix[0, 0], rel=REL)
    assert fy == pytest.approx(undistort.camera_matrix[1, 1], rel=REL)

