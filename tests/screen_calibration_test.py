"""
Testing the screen calibration routines.
"""
import pytest
import volteracamera.intrinsics.circles_calibration as cc
import volteracamera.intrinsics.screen_based_calibration as sc
import volteracamera.intrinsics.calibration_generation as cg
from PIL import Image
import numpy as np
import random

def test_cam_matrix():
    """
    Test the default image camera matrix generation (the one used to project the iamge on the screen.)
    """
    image, _, _ = cc.generate_symmetric_circle_grid()
    width, height = sc.get_image_dimensions(image)
    cam_matrix = sc.get_camera_matrix(width, height)

    assert (width == 750)
    assert (height == 900)
    np.testing.assert_array_equal(cam_matrix, np.array([[250, 0, 750/2],[0, 250, 900/2],[0, 0, 1]]))

# def test_trans_matrix():
#     """
#     Test the transformation matrix.
#     """
#     rvec = np.array([np.pi/2, 0, 0])
#     tvec = np.array([3, 5, 7])
#     ref_matrix = np.array ([[1, 0, 0, 3], [0, 0, -1, 5], [0, 1, 0, 7], [0, 0, 0, 1]])
#     trans_matrix = sc.get_transformation_matrix(rvec, tvec)
#     np.testing.assert_array_almost_equal(ref_matrix, trans_matrix)

#     reduced_matrix = np.array ([[1, 0, 0, 3], [0, 0, -1, 5], [0, 1, 0, 7]])
#     np.testing.assert_array_almost_equal(reduced_matrix, sc.reduce_4_to_3(trans_matrix))

def resize_image_without_scaling (image: np.ndarray, width: int = 1280, height: int = 720 )->np.ndarray:
    """
    resize an np.array into a new np.array.
    """
    input_image = Image.fromarray(image)
    original_width, original_height = input_image.size #width, height order for PIL image.
    ratio = width/height
    original_ratio = original_width/original_height
    if original_ratio >= ratio:
        new_width = width
        new_height = int(new_width * ratio)
    else:
        new_height = height
        new_width = int(new_height / ratio)
    resized_image = np.array(input_image.resize((new_width, new_height), Image.BICUBIC))
    final_image = np.zeros((height, width))
    top_corner = (int((final_image.shape[0] - resized_image.shape[0])/2), 
                  int(final_image.shape[1] - resized_image.shape[1])/2)
    final_image[top_corner[0]:(top_corner[0]+new_height), top_corner[1]:(top_corner[1]+new_width)] = resized_image
    return final_image

def test_resize_image():
    """
    test the above method
    """
    original_image = np.ones((640, 480))
    final_image = np.zeros((1280, 640))
    new_width, new_height = (int(640*640/480), 640)
    final_image[int((1280-new_width)/2):(int((1280-new_width)/2)+new_width),:] = np.ones((int(640*(640/480)), 640))
    resize_image_without_scaling(original_image)


def test_calibration_through_screen():
    """
    Test the calibration process through a screen.
    """
    #Generate base image.
    image, points, pattern = cc.generate_symmetric_circle_grid()
    width, height = sc.get_image_dimensions(image)
    screen_projection_matrix = sc.get_camera_matrix(width, height)
    #Generate the rvecs and tvecs.
    random.seed(1)
    transforms = [cg.generate_random_rvec_tvec() for _ in range(30)]
    #generate the images on the screen.
    images_on_screen = [sc.create_projected_image(image, rvec, tvec) for rvec, tvec in transforms]
    #generate the images on the sensor using a fake camera matrix.
    rvec = [0, 0, 0]
    tvec = [0, 0, 0.015]
    images_on_sensor = [sc.create_projected_image(resize_image_without_scaling(an_image), rvec, tvec, 5357) for an_image in images_on_screen]
    #Minimize the combined system.
    result = sc.calibrate_through_screen(images_on_sensor, transforms, points, pattern, screen_projection_matrix)

    np.assert_array_almost_equal (result.camera_matrix, sc.get_camera_matrix())
    np.assert_array_almost_equal (result.distortion, np.array([0, 0, 0, 0, 0]))
    assert (False)
