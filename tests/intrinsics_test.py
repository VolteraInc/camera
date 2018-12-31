"""
Test the intrinsics tools, make sure the results are sensible.
"""
import pytest
import numpy as np
from volteracamera.intrinsics.circles_calibration import (generate_symmetric_circle_grid, run_calibration)
from volteracamera.intrinsics.calibration_generation import generate_random_cal_images, create_projected_image
from volteracamera.analysis.undistort import Undistort
#import volteracamera.intrinsics.charuco_calibration.run_calibration as crc
#import volteracamera.intrinsics.charuco_calibration.IMG as crcIMG

import cv2

REL=0.1

def test_null_image_transform():
    """
    Test the image rotation when there is no change (should be equal)
    """
    image, _, _ = generate_symmetric_circle_grid(5, 4, 150)
    with pytest.raises (ZeroDivisionError):
        _ = create_projected_image(image, [0, 0, 0], [0, 0, -1.0])

def test_simple_image_transform():
    """
    Test the image rotation when there is no change (should be equal)
    """
    image, _, _ = generate_symmetric_circle_grid(5, 4, 150)
    new_image = create_projected_image(image, [0, 0, 0], [0, 0, 0])
    diff = image - new_image
    assert (diff.flatten() > 1).sum() == 0


def test_intrinsics_calibration():
    """
    Meta test of the entire process
    """
    image, points, pattern_size = generate_symmetric_circle_grid(5, 4, 150)
    images = generate_random_cal_images(image, 25)
    assert images[0].shape == image.shape

    undistort = run_calibration(images, points, pattern_size, display=False)

    cx = image.shape[1] / 2 #python shape goes height, width order
    cy = image.shape[0] / 2 
    fx = 250
    fy = 250 
    print (undistort)
    assert cx == pytest.approx(undistort.camera_matrix[0, 2], rel=REL)
    assert cy == pytest.approx(undistort.camera_matrix[1, 2], rel=REL)
    assert fx == pytest.approx(undistort.camera_matrix[0, 0], rel=REL)
    assert fy == pytest.approx(undistort.camera_matrix[1, 1], rel=REL)

# def test_intrinsics_charuco_calibration():
#     """
#     Meta test of the entire process
#     """
#     images = generate_random_cal_images(crcIMG, 30)
#     assert images[0].shape == crcIMG.shape
    
#     images[0] = crcIMG

#     undistort = crc(images, display=True)

#     cx = crcIMG.shape[1] / 2 #python shape goes height, width order
#     cy = crcIMG.shape[0] / 2 
#     fx = 250
#     fy = 250 
#     print (undistort)
#     assert cx == pytest.approx(undistort.camera_matrix[0, 2], rel=REL)
#     assert cy == pytest.approx(undistort.camera_matrix[1, 2], rel=REL)
#     assert fx == pytest.approx(undistort.camera_matrix[0, 0], rel=REL)
#     assert fy == pytest.approx(undistort.camera_matrix[1, 1], rel=REL)

