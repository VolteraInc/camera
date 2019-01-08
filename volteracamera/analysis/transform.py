"""
Transformation class
"""

import numpy as np
import transforms3d as tfd
from .plane import Plane

class Transform (object):
    """
    Class containing transforms.
    """
    def __init__(self, rotation=None, translation = None):
        """
        Set up a transformation object given a rotation in the twist notation (axis angle with the rotation being the normalization factor) and a translation.

        The default is the null transform.
        """

        if type(rotation) is type(None):
            rotation = [0.0, 0.0, 0.0]
        if type(translation) is type(None):
            translation = [0.0, 0.0, 0.0]

        if len(rotation) != 3:
            raise TypeError("Rotation must be in 3 vector twist form.")
        if len(translation) != 3:
            raise TypeError("Translation must be in 3 vector form.")

        self._rotation = np.array(rotation)
        self._translation = np.array(translation)
        self._update_matrix()

    def _update_matrix (self):
        """
        Update the rotation matrix after a value change.
        """
        angle = np.sqrt(np.dot(self._rotation, self._rotation))
        axis = [0, 0, 1]
        if angle != 0.0:
           axis = self._rotation/angle

        self._rot_matrix = tfd.axangles.axangle2mat(axis, angle)
        self._matrix = np.identity(4)
        self._matrix[0:3, 0:3] = self._rot_matrix
        self._matrix[0:3, 3] = self._translation

    def matrix(self):
        """
        Return the transformation matrix.
        """
        return self._matrix

    def transform_point(self, point):
        """
        Transform a point
        """
        if len(point) != 3:
            raise TypeError("Point must be three vector.")
        temp_point = np.append (np.asarray(point), 1.0)
        return np.dot(self._matrix, temp_point)[0:3]

    def transform_plane(self, plane):
        """
        This method transforms a plane class by performing the full transform on 
        the point and rotating the normal.
    
        return a new plane class.
        """
        if not isinstance(plane, Plane):
            raise TypeError("transform_plane only accepts plane class types.")

        new_point = self.transform_point(plane.point)
        new_normal = np.dot (self._rot_matrix, plane.normal)
        return Plane(point_on_plane = new_point, normal = new_normal) 

    def combine(self, other):
        """
        Multiply two transforms together.
        """
        new_matrix = np.dot(self.matrix(), other.matrix())
        tvec = new_matrix[0:3, 3].transpose()
        R = new_matrix[0:3, 0:3]
        axis, angle = tfd.axangles.mat2axangle(R)
        rvec = axis*angle
        return Transform (rotation=rvec, translation=tvec)