"""
Tools for Generating and calibrating with ChArUco images.

Code modified from here: 
http://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
"""
import cv2
import cv2.aruco as aruco
import numpy as np
from ..analysis.undistort import Undistort

DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
BOARD = aruco.CharucoBoard_create(5,5,.025,.0125,DICTIONARY)
IMG = BOARD.draw((150,150))

#Dump the calibration board to a file
#cv2.imwrite('charuco.png',IMG)

def analyze_calibration_image(image: np.ndarray, display=False)->tuple:
    """
    Analyze the image and return a list of object corners and a list of associated with each corner.
    
    display=True displays the image.

    If the function can't find the pattern, it throws a runtime error.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_markers = cv2.aruco.detectMarkers(gray_image, DICTIONARY)

    if len(result_markers[0])>0:
        result_corners = cv2.aruco.interpolateCornersCharuco(result_markers[0], result_markers[1], gray_image, BOARD)
        if result_corners[1] is not None and result_corners[2] is not None and len(result_corners[1])>3:
            if display:
                _display_charuco(image, result_markers[0], result_markers[1], result_corners[1], result_corners[2])
            return (result_corners[1], result_corners[2])
    raise RuntimeError ("Could not find ChArUco pattern in image.")

def _display_charuco(image: np.ndarray, markers: np.ndarray, ids_markers: np.ndarray, corners: np.ndarray, ids_corners: np.ndarray )->None:
    """
    Display an analyzed ChArUco image. Do not use directly, use through analyze_calibration_image
    """
    a_copy = np.copy(image)
    cv2.aruco.drawDetectedMarkers(a_copy, markers, ids_markers)
    cv2.aruco.drawDetectedCornersCharuco(a_copy, corners, ids_corners)

    a_copy = cv2.resize(a_copy, (1000, 1000))

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

    image_size = image_list[0].shape
    all_corners = []
    all_ids = []
    for image in image_list:
        try:
            current_corners, current_ids = analyze_calibration_image(image)
            all_corners.append(current_corners)
            all_ids.append(current_ids)
        except RuntimeError:
            print ("Missed Image")
            pass
    width, height, _ = image_size
    calibration = aruco.calibrateCameraCharuco(all_corners, all_ids, BOARD, (height, width), None, None)

    return Undistort(calibration[1], calibration[2]) #rvecs, tvecs, calibration[3], calibration[4])



