"""
This class is used to represent a plane given a point on the plane and a normal.
"""
import numpy as np


class Plane (object):
    """
    This class is used to represent a plane given a point on the plane and a surface normal.
    """

    def __init__(self, point_on_plane = None, normal = None):
        """
        constructor, default point at 0 and plane normal in the positive z direction
        """
        if point_on_plane is None:
            point_on_plane = [0., 0., 0.]
        if normal is None:
            normal = [0., 0., 1.]

        if (len(point_on_plane) != 3):
            raise TypeError("Point on plane must be a 3-vector")
        if (len(normal) != 3):
            raise TypeError("Normal must be a 3-vector")
        
        self.point = np.asarray(point_on_plane)
        self.normal = np.asarray(normal)
        self._normalize()
    
    def _normalize(self):
        """
        Normalize the plane.
        """
        denominator = np.sqrt(np.dot(self.normal, self.normal))
        if (denominator < 0.0000001):
            raise ZeroDivisionError("Normal is too close to 0.")
        self.normal /= denominator


