"""
Tools for Generating and calibrating with sphere images.
"""
import cv2
import numpy as np


def generate_asymmetric_circle_grid (size_x: int, size_y: int, circle_spacing: int )->tuple:
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
            points.append([int((2*j + (i % 2))*circle_spacing), int(i*circle_spacing), 0])
    image = np.zeros((size_x*circle_spacing + 2*circle_spacing, size_y*circle_spacing*2 + circle_spacing), np.uint8) + 255

    for center in points:
        [x, y, _] = center
        point_2d = (x+int(circle_spacing), y+int(circle_spacing))
        cv2.circle (image, point_2d, radius, (0, 0, 0), -1)  

    return (image, points, pattern)

def generate_symmetric_circle_grid (size_x: int, size_y: int, circle_spacing: int )->tuple:
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
            points.append(np.array([int(i*circle_spacing), int(j*circle_spacing), 0]))
    image = np.zeros((size_y*circle_spacing + circle_spacing, size_x*circle_spacing + circle_spacing), np.uint8) + 255

    for center in points:
        [x, y, _] = center
        point_2d = (x+int(circle_spacing), y+int(circle_spacing))
        cv2.circle (image, point_2d, radius, (0, 0, 0), -1)  

    return (image, np.asarray(points), pattern)



IMG, OBJECT_POINTS, PATTERN_SIZE = generate_symmetric_circle_grid(5, 4, 150)

#Dump the calibration board to a file
#cv2.imwrite('symmetric.png',IMG)

def analyze_calibration_image(image: np.ndarray, display=False)->tuple:
    """
    Analyze the image and return a list of object corners and a list of associated with each corner.
    
    display=True displays the image.

    If the function can't find the pattern, it throws a runtime error.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, centers = cv2.findCirclesGrid(gray_image, PATTERN_SIZE, flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    if display:
        _display_analyzed (image, centers)
    if centers is None:
        raise RuntimeError ("Could not find circle pattern in image.")  
    return centers


def _display_analyzed(image: np.ndarray, markers: np.ndarray )->None:
    """
    Display an analyzed checkerboard image. Do not use directly, use through analyze_calibration_image
    """
    a_copy = np.copy(image)
    cv2.drawChessboardCorners(a_copy, PATTERN_SIZE, markers, True)

    a_copy = cv2.resize(a_copy,  (500, 500))

    cv2.imshow('frame', a_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
def run_calibration(image_list: list)->tuple:
    """
    Run the ChArUco calibration given a set of images (loaded into numpy arrays) 
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
            current_corners = analyze_calibration_image(image)
            all_corners.append(current_corners)
            object_points.append(OBJECT_POINTS)
        except RuntimeError:
            print ("Missed Image")
            pass
    object_points = np.asarray(object_points).astype('float32')
    all_corners = np.asarray(all_corners).astype('float32')
    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, all_corners, image_size,None,None)

    return (camera_matrix, distortion) #rvecs, tvecs, calibration[3], calibration[4])



