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
    original_height, original_width = input_image.size # height, width order for PIL image.
    width_scaling = width/original_width
    height_scaling = height/original_height

    scaling = height_scaling if height_scaling < width_scaling else width_scaling
    new_width = int(scaling*original_width)
    new_height = int(scaling*original_height)

    input_image = input_image.resize((new_height, new_width), Image.BICUBIC)
    resized_image = np.array(input_image)

    final_image = np.zeros((width, height)) + 255
    top_width, top_height = (int((width - new_width)/2), 
                  int((height - new_height)/2))

    final_image[top_width:(top_width+new_width), top_height:(top_height+new_height)] = resized_image
    return final_image

def test_resize_image():
    """
    test the above method
    """
    original_image = np.zeros((640, 480))
    final_image = np.zeros((1280, 720)) + 255
    new_width, new_height = (int(640*720/480), int(480 * 720/480))
    final_image[int((1280-new_width)/2):(int((1280-new_width)/2)+new_width),:] = np.zeros((new_width, new_height))

    np.testing.assert_array_equal (resize_image_without_scaling(original_image), final_image)
    
import cv2
def _display_analyzed(image: np.ndarray)->None:
    """
    Display an analyzed checkerboard image. Do not use directly, use through analyze_calibration_image
    """
    #a_copy = cv2.resize(image,  (500, 500))

    cv2.imshow('frame', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

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
    transforms = [cg.generate_random_rvec_tvec() for _ in range(5)]
    #generate the images on the screen.
    images_on_screen = [sc.create_projected_image(image, rvec, tvec) for rvec, tvec in transforms]
    #generate the images on the sensor using a fake camera matrix.
    rvec = [0, 0, 0]
    tvec = [0, 0, 0.015]
    images_on_sensor = [sc.create_projected_image(resize_image_without_scaling(an_image), rvec, tvec, 5357).astype("uint8") for an_image in images_on_screen]

    #Minimize the combined system.
    result = sc.calibrate_through_screen(images_on_sensor, transforms, points, pattern, screen_projection_matrix, display=False)

    np.testing.assert_array_almost_equal (result.camera_matrix, sc.get_camera_matrix())
    np.testing.assert_array_almost_equal (result.distortion, np.array([0, 0, 0, 0, 0]))
    assert (False)
