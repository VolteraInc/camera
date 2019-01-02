"""
This is a set of tools and routines for running calibrations based on a capture on a screen (images of a checkerboard)
"""
from .circles_calibration import generate_symmetric_circle_grid, analyze_calibration_image
from ..analysis.transform import Transform
from ..analysis.undistort import Undistort
import numpy as np
import transforms3d as tfd
import cv2
from scipy.optimize import minimize, least_squares

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

    transform = Transform(rotation = list(rvec), translation = list(tvec))
    undistortion = Undistort (camera_matrix = projection_matrix, distortion=[0, 0, 0, 0, 0])
    moved_point = transform.transform_point(point)
    out_point =  undistortion.project_point_with_distortion(moved_point)

    #import pdb; pdb.set_trace()
    return out_point

def project_point_to_screen(point: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, projection_matrix: np.ndarray, pixel_size: float)->np.ndarray:
    """
    This function takes a point in three space as a numpy array, transforms it to a new position through the
    transformation given by tvec and rvec (translation and axis angle rotation) and projects it onto a screen at the origin (normal along z)
    using a projection matrix. The returned point is then normalized and a screen co-ordinate is returned (3d point).
    """
    return project_point (point, rvec, tvec, projection_matrix, np.ndarray ([0, 0, 0, 0, 0])) * pixel_size

    
def project_point_to_screen_to_camera (point: np.ndarray, rvec_cam: np.ndarray, tvec_cam: np.ndarray, 
                                       projection_matrix_cam: np.ndarray, distortion_cam: np.ndarray,
                                       rvec_screen: np.ndarray, tvec_screen: np.ndarray, 
                                       projection_matrix_screen: np.ndarray, screen_pixel_size: float)->np.ndarray:
    """
    This function transforms a point onto the screen, and then onto the the camera (two stages of projection)
    _cam correspond to camera to screen parameters
    _screen correspond to point to world to screen parameters
    returns a 2d point on the sensor.
    """
    screen_point = project_point_to_screen (point, rvec_screen, tvec_screen, projection_matrix_screen, screen_pixel_size)
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
              [width, 0, 1]]) - center # the 1 in the z parameter is needed for the getPerspectiveTransform function.
    
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

def unpack_params (params: list):
    """
    Function to unpack and return the required parameters from the minimization parameter list.
    params:
    fx, fy, cx, cy, object_scale, d1, d2, d3, d4, d5, rvec1, rvec2, rvec3, tvec1, tvec2, tvec3

    return cam_matrix, distortion_matrix, rvec, tvec, object_scale
    """
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    object_scale = params[4]
    distortion = np.asarray(params[5:10])
    #rvec = np.asarray (params[10:13])
    tvec = np.asarray (params[10:])
    rvec =  np.asarray([0, 0, 0])
    cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return (cam_matrix, distortion, rvec, tvec, object_scale)

# def full_screen_residual (params:list, image_points:list, transforms:list, object_points:list, screen_projection_matrix:np.ndarray)->list:
#     """
#     Return the sum of the radial differences between the different image and object points
#     """
#     camera_matrix, distortion, rvec_cam, tvec_cam, screen_pixel_size = unpack_params(params)
#     residuals = 0.0
#     for image_set, transform in zip(image_points, transforms):
#         for image_point, object_point in zip (image_set, object_points):
#             projected_point = project_point_to_screen_to_camera (object_point, rvec_cam, tvec_cam, camera_matrix, distortion,
#                                        transform[0], transform[1], screen_projection_matrix, screen_pixel_size)
#             diff = image_point[0] - projected_point
#             residuals += diff[0]*diff[0] + diff[1]*diff[1]

#     return residuals

def full_screen_residual (params:list, image_points:list, object_points:list)->list:
    """
    Return the sum of the radial differences between the different image and object points
    """
    camera_matrix, distortion, rvec_cam, tvec_cam, object_scale = unpack_params(params)
    
    diff_points = [ image_p - project_point(object_scale*object_p, rvec_cam, tvec_cam, camera_matrix, distortion) for image_p, object_p in zip(image_points, object_points) ]
    
    print (np.asarray(diff_points).mean(axis=0))
    #residuals = np.asarray([ diff[0]*diff[0] + diff[1]*diff[1] for diff in diff_points])
   
    return np.asarray(diff_points).flatten()

def calibrate_through_screen(captured_image_list: list, original_image_list: list, 
        pattern_size: tuple, display: bool = False) -> tuple:
    """
    Take a set of images (original and captured), the object size for the point set

    First, find the image point in each image, and discard sets where 
    there are no points. Find the matching object points in the original images.

    Then we iteratively determine the difference between the image points 
    and the true points projected on the sensor through the screen.
    """
    if not captured_image_list:
        raise RuntimeError("No images passed to calibration routine.")
    if len(captured_image_list) != len(original_image_list):
        raise RuntimeError("There must be a transform provided for each image.")

    image_size = (captured_image_list[0].shape[1], captured_image_list[0].shape[0])
    object_image_size = (original_image_list[0].shape[1], original_image_list[0].shape[0])
    captured_corners = np.asarray([0, 0])
    original_corners = ([0, 0, 0])
    for cap_image, orig_image in zip (captured_image_list, original_image_list):
        try:
            current_corners = np.asarray(analyze_calibration_image(cap_image, pattern_size, display)).astype('float32')
            orig_current_corners = np.asarray(analyze_calibration_image(orig_image, pattern_size, display)).astype('float32')
            captured_corners = np.vstack  ((captured_corners, np.asarray( [ point[0] for point in current_corners])))
            original_corners = np.vstack ((original_corners, [ np.asarray([point[0][0] - object_image_size[0]/2, point[0][1] - object_image_size[1]/2, 0.0]) for point in orig_current_corners]))
        except RuntimeError:
            print("Missed Image")
            pass
    
    captured_corners = captured_corners[1:]
    original_corners = original_corners[1:]
    #captured_corners = np.asarray(captured_corners).astype('float32')
    #original_corners = np.asarray(original_corners).astype('float32')
    
    #ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    #    original_corners, captured_corners, image_size, None, None)#, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
      
    #parameters = [5000, 5000, image_size[0]/2, image_size[1]/2, 1/1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.015]
    parameters = [5357, 5357, image_size[0]/2, image_size[1]/2, 57.59E-6 * object_image_size[0]/image_size[0], 0, 0, 0, 0, 0, 0, 0, 0.015]
    # bounds = np.array([[0,np.inf], [0,np.inf], [(image_size[0]-image_size[0]*0.2),(image_size[0]+image_size[0]*0.2)], 
    #          [(image_size[1]-image_size[1]*0.2),(image_size[1]+image_size[1]*0.2)], [0, np.inf], 
    #          [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.1, 0.1]])
    _, _, _, _, scaling = unpack_params(parameters)
    bounds = [[0, 0, (image_size[0]/2-image_size[0]/2*0.2), (image_size[1]/2-image_size[1]/2*0.2), 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1, -1, -2], 
              [np.inf, np.inf, (image_size[0]/2+image_size[0]/2*0.2), (image_size[1]/2+image_size[1]/2*0.2), np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, 2 ]]
    #               (-np.pi,np.pi),(-np.pi,np.pi),(-np.pi,np.pi), (None, None), (None, None),(None, None))

    #res = minimize(full_screen_residual, parameters, args=(captured_corners, original_corners))
    res = least_squares(full_screen_residual, parameters, method='trf', verbose=2, args=(captured_corners, original_corners), bounds=bounds)
    
    if res.success:
        print ("Fit sucessful.")
    else:
        print ("Fit failed.")
    camera_matrix, distortion, rvecs, tvecs, object_scale = unpack_params(res.x)

    print ("Camera Matrix:")
    print (camera_matrix)
    print ("Distortion Matrix")
    print (distortion)
    print ("Scaling")
    print (object_scale)
    print ("rvec")
    print(rvecs)
    print ("tvec")
    print(tvecs)
    
    return Undistort(camera_matrix, distortion)


# def calibrate_through_screen(image_list: list, transforms: list, 
#         object_points_in: list, pattern_size: tuple, 
#         screen_projection_matrix: np.ndarray, screen_pixel_size: float, display: bool = False) -> tuple:
#     """
#     Take a set of images, a set of transforms to the screen for those 
#     images, the object points and object size for that point set and 
#     the screen projeciton matrix and calculate the camera calibration.

#     First, find the image point in each image, and discard sets where 
#     there are no images.

#     Then we iteratively determine the difference between the image points 
#     and the true points projected on the sensor through the screen.
#     """
#     if not image_list:
#         raise RuntimeError("No images passed to calibration routine.")
#     if len(image_list) != len(transforms):
#         raise RuntimeError("There must be a transform provided for each image.")

#     image_size = (image_list[0].shape[1], image_list[0].shape[0])
#     all_corners = []
#     used_transforms = []
#     for image, transform in zip (image_list, transforms):
#         try:
#             current_corners = np.asarray(analyze_calibration_image(image, pattern_size, display)).astype('float32')
#             all_corners.append (current_corners)
#             used_transforms.append(transform)
#         except RuntimeError:
#             print("Missed Image")
#             pass
#     object_points = np.asarray(object_points_in).astype('float32')
    
#     #parameters = [5000, 5000, image_size[0]/2, image_size[1]/2, screen_pixel_size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15]
#     parameters = [5000, 5000, image_size[0]/2, image_size[1]/2, screen_pixel_size, 0.15]
#     bounds = ((0,None), (0,None), ((image_size[0]-image_size[0]*0.2),(image_size[0]+image_size[0]*0.2)), \
#               ((image_size[1]-image_size[1]*0.2),(image_size[1]+image_size[1]*0.2)), (0, None), \
#               (None, None), (None, None), (None, None), (None, None), (None, None), \
#               (-np.pi,np.pi),(-np.pi,np.pi),(-np.pi,np.pi), (None, None), (None, None),(None, None))

#     res = minimize(full_screen_residual, parameters, args=(all_corners, used_transforms, object_points, screen_projection_matrix))
#     #res = least_squares(full_screen_residual, parameters, method='lm', args=(all_corners, used_transforms, object_points, screen_projection_matrix), verbose=2)

#     camera_matrix, distortion, rvec, tvec, pixel_size = unpack_params(res.x)

#     if res.success:
#         print ("Fit Successful.")
#     else:
#         print ("Fit Unsuccessful.")

#     print ("Camera Matrix:")
#     print (camera_matrix)
#     print ("Distortion Matrix")
#     print (distortion)
#     print ("rvec")
#     print (rvec)
#     print ("tvec")
#     print (tvec)
#     print ("pixel_size")
#     print (pixel_size)

    
#     return Undistort(camera_matrix, distortion)
