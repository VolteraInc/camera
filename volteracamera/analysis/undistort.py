"""
Given a camera model (camera matrix and a 5 point radtan undistortion model),
this class can be used to undistort points and lists of laser points, as well as
undistort images.
"""
import cv2
import numpy as np

class Undistort (object):
    """
    This class is used to undistort camera points and images given a set of intrinsics 
    (camera matrix, 3x3 and distortion parameters (5x1).
    """

    def __init__(self, camera_matrix, distortion):
        """
        3x3 camera matrix and 5x1 distortion model (radtan). Throws RuntimeError
        if these dimensions aren't correct.
        """
        if camera_matrix.shape = (3, 3):
            raise RuntimeError("Camera matrix has the wrong shape.")
        if len(distortion) != 5:
            raise RuntimeError("Distortion parameters are the wrong size.")

        self.camera_matrix = camera_matrix
        self.distortion = distortion

    def undistort_point ( point_2d ):
        """
        Undistort a single point.
        """
        return cv2.undistortPoints(point_2d, self.camera_matrix, self.distortion)

    def undistort_points ( points_2d ):
        """
        Undistort a list of points
        """
        return cv2.undistortPoints(points_2d, self.camera_matrix, self.distortion)

    def undistort_image ( image ):
        """
        Return an undistorted image.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion)
