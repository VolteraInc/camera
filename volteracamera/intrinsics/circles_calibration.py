"""
Tools for Generating and calibrating with sphere images.
"""
import cv2
import numpy as np
from ..analysis.undistort import Undistort

CAMERA_MATRIX_GUESS = np.array([[5357, 0, 640], [0, 5357, 360], [0, 0, 1]], dtype=float)
DISTORTION_GUESS = np.array([0, 0, 0, 0, 0], dtype=float)

DEFAULT_X = 4
DEFAULT_Y = 5
DEFAULT_SPACING = 150

def generate_asymmetric_circle_grid(size_x = DEFAULT_X, size_y = DEFAULT_Y, circle_spacing = DEFAULT_SPACING)->tuple:
    """
    Generate an asymmetric circle pattern.
    size x and y are the number of circles, and the circle spacing is the spacing in pixels.
    Returns the image, point list and the pattern size.
    """
    points = []
    radius = int(circle_spacing/2)
    pattern = (size_y, size_x)
    for i in range(size_x):
        for j in range(size_y):
            points.append([int((2*j + (i % 2))*circle_spacing),
                           int(i*circle_spacing), 0])
    image = np.zeros((size_x*circle_spacing + 2*circle_spacing,
                      size_y*circle_spacing*2 + circle_spacing), np.uint8) + 255

    for center in points:
        [x, y, _] = center
        point_2d = (x+int(circle_spacing), y+int(circle_spacing))
        cv2.circle(image, point_2d, radius, (0, 0, 0), -1)

    return (image, points, pattern)


def generate_symmetric_circle_grid(size_x = DEFAULT_X, size_y = DEFAULT_Y, circle_spacing = DEFAULT_SPACING)->tuple:
    """
    Generate an asymmetric circle pattern.
    size x and y are the number of circles, and the circle spacing is the spacing in pixels.
    Returns the image, point list and the pattern size.
    """
    points = []
    radius = int(circle_spacing/5)
    pattern = (size_x, size_y)
    for j in range(size_y):
        for i in range(size_x):
            points.append(
                np.array([int(i*circle_spacing), int(j*circle_spacing), 0]))
    image = np.zeros((size_y*circle_spacing + circle_spacing,
                      size_x*circle_spacing + circle_spacing), np.uint8) + 255

    for center in points:
        [x, y, _] = center
        point_2d = (x+int(circle_spacing), y+int(circle_spacing))
        cv2.circle(image, point_2d, radius, (0, 0, 0), -1)

    return (image, np.asarray(points), pattern)


def analyze_calibration_image(image: np.ndarray, pattern_size: tuple, display=False)->tuple:
    """
    Analyze the image and return a list of object corners and a list of associated with each corner.

    display=True displays the image.

    If the function can't find the pattern, it throws a runtime error.
    """
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
    ret, centers = cv2.findCirclesGrid(
        gray_image, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    if display:
        _display_analyzed(image, pattern_size, centers)
    if centers is None or not ret:
        raise RuntimeError("Could not find circle pattern in image.")
    return centers


def _display_analyzed(image: np.ndarray, pattern_size: tuple, markers: np.ndarray)->None:
    """
    Display an analyzed checkerboard image. Do not use directly, use through analyze_calibration_image
    """
    a_copy = np.copy(image)
    cv2.drawChessboardCorners(a_copy, pattern_size, markers, True)

    #a_copy = cv2.resize(a_copy,  (500, 500))

    cv2.imshow('frame', a_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()


def run_calibration(image_list: list, object_points_in: list, pattern_size: tuple, display: bool = False) -> tuple:
    """
    Run the circles calibration given a set of images (loaded into numpy arrays) 
    and return a tuple of the camera matrix, distortion parameters, rvecs and tvecs.

    Raises a runtime error if there are no images.
    """
    if not image_list:
        raise RuntimeError("No images passed to calibration routine.")

    image_size = (image_list[0].shape[1], image_list[0].shape[0])
    all_corners = []
    object_points = []
    for image in image_list:
        try:
            current_corners = analyze_calibration_image(image, pattern_size, display)
            all_corners.append(current_corners)
            object_points.append(object_points_in)
        except RuntimeError:
            print("Missed Image")
            pass
    object_points = np.asarray(object_points).astype('float32')
    all_corners = np.asarray(all_corners).astype('float32')
    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, all_corners, image_size, CAMERA_MATRIX_GUESS, DISTORTION_GUESS)#, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # rvecs, tvecs, calibration[3], calibration[4])
    return Undistort(camera_matrix, distortion)
