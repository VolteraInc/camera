"""
This is a set of tools and routines for running calibrations based on a capture on a screen (images of a checkerboard)
"""
from .circles_calibration import generate_symmetric_circle_grid
from ..analysis.transform import Transform
from ..analysis.undistort import Undistort
import numpy as np
import transforms3d as tfd
import cv2

FOCAL_LENGTH=250
SMALL_NUM = 0.0000001

def get_image_dimensions(image: np.ndarray)->tuple:
    """
    Get the image width and height.
    """
    if len(image.shape) == 3:    
        height, width, _ = image.shape #python matrix shape is height, width order.
    else:
        height, width = image.shape 
    return width, height

def get_camera_matrix(width: int, height:int, f = FOCAL_LENGTH)->np.ndarray:
    """
    Generate the camera matrix from a given image.
    """
    center = np.array([width/2, height/2, 0])
    cam_matrix = np.array([[f, 0, center[0]],
                           [0, f, center[1]],
                           [0, 0, 1]])
    return cam_matrix

def reduce_4_to_3(input_matrix: np.ndarray)->np.ndarray:
    """
    Take a 4x4 matrix and turn it into 3x4.
    """
    reducing_matrix = np.array ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    return np.dot(reducing_matrix, input_matrix)
    
def project_point (point: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, projection_matrix: np.ndarray, distortion:np.ndarray)->np.ndarray:
    """
    General purpose projection function for transforming a point from world to camera space through rvec(twist formalization) and tvec and then
    projecting it onto the screen/sensor described by the projection and distortion matrix.
    It returns a 2d point on the final sensor/screen in the z=0 plane.
    """
    transform = Transform(rotation = rvec, translation = tvec)
    undistortion = Undistort (camera_matrix = projection_matrix, distortion=[0, 0, 0, 0, 0])
    moved_point = transform.transform_point(point)
    projected_point = undistortion.undistort_points(moved_point)
    if np.abs(projected_point[2]) < SMALL_NUM:
        return np.array ([0, 0])
    else:
        return np.array([projected_point[0], projected_point[1]])/projected_point[2]

def project_point_to_screen(point: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, projection_matrix: np.ndarray)->np.ndarray:
    """
    This function takes a point in three space as a numpy array, transforms it to a new position through the
    transformation given by tvec and rvec (translation and axis angle rotation) and projects it onto a screen at the origin (normal along z)
    using a projection matrix. The returned point is then normalized and a screen co-ordinate is returned (3d point).
    """
    return project_point (point, rvec, tvec, projection_matrix, np.ndarray ([0, 0, 0, 0, 0]))

    
def project_point_to_screen_to_camera (point: np.ndarray, rvec_cam: np.ndarray, tvec_cam: np.ndarray, 
                                       projection_matrix_cam: np.ndarray, distortion_cam: np.ndarray,
                                       rvec_screen: np.ndarray, tvec_screen: np.ndarray, 
                                       projection_matrix_screen: np.ndarray)->np.ndarray:
    """
    This function transforms a point onto the screen, and then onto the the camera (two stages of projection)
    _cam correspond to camera to screen parameters
    _screen correspond to point to world to screen parameters
    returns a 2d point on the sensor.
    """
    screen_point = project_point_to_screen (point, rvec_screen, tvec_screen, projection_matrix_screen)
    screen_point = np.append (screen_point, 0.0)
    sensor_point = project_point(screen_point, rvec_cam, tvec_cam, projection_matrix_cam, distortion_cam)
    return sensor_point

def create_projected_image(image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, focal_length: int = FOCAL_LENGTH)->np.ndarray:
    """
    Create a new image from a given input image that shows the input projected in space by the 
    homography given by rvec and tvec. rvec is in angle axis form. focal_lenght is used to generate 
    the calibration matrix.
    """
    norm = np.sqrt(np.dot (rvec, rvec))
    if np.abs(norm) > SMALL_NUM:
        axis = rvec/norm
    else:
        axis = [1, 0, 0]
    rot_matrix = tfd.axangles.axangle2mat(axis, norm)
    width, height = get_image_dimensions(image)
    center = np.array([width/2, height/2, 0])
    cam_matrix = get_camera_matrix(width, height, focal_length)
    points = np.array([[0, 0, 1],
              [width, height, 1],
              [0, height, 1],
              [width, 0, 1]]) - center
    
    transformed_points = [np.dot(rot_matrix, point) + tvec  for point in points]
    
    projected_points = np.array([np.dot(cam_matrix, point) for point in transformed_points], dtype="float32")
    
    for point in projected_points:
        if np.abs(point[2]) < SMALL_NUM:
            raise ZeroDivisionError("Image points at origin, skipping image projection.")
    projected_points = np.array([[point[0]/point[2], point[1]/point[2]] for point in projected_points], dtype="float32")
    object_points = np.array([np.dot(cam_matrix, point) for point in points], dtype="float32")    

    # for point in object_points:
    #     if np.abs(point[2]) < SMALL_NUM:
    #         raise ZeroDivisionError("Image points at origin, skipping image projection")    
    # object_points = np.array([[point[0]/point[2], point[1]/point[2]] for point in object_points ], dtype="float32")
    object_points = np.array([[point[0], point[1]] for point in object_points ], dtype="float32")

    H = cv2.getPerspectiveTransform(object_points, projected_points)
    
    warped_image = cv2.warpPerspective(image, H, (width, height), borderValue=(255, 255, 255))
    return warped_image

def calibrate_through_screen(image_list: list, transforms: list, 
        object_points_in: list, pattern_size: tuple, 
        screen_projection_matrix: np.ndarray, display: bool = False) -> tuple:
    """
    Take a set of images, a set of transforms to the screen for those 
    images, the object points and object size for that point set and 
    the screen projeciton matrix and calculate the camera calibration.

    First, find the image point in each image, and discard sets where 
    there are no images.

    Then we iteratively determine the difference between the image points 
    and the true points projected on the sensor through the screen.
    """
