"""
Class representing a line in 3D space.
"""
import numpy as np

class Line (object):
    """
    This class stores information about a line in 3d space given by a point on 
    the line and a direction vector

        Y = P + t * n

        Y any point on the line (vector)
        P is a point on the line (vector)
        n is a vector in the line direction (vector)
        t is a scalar parameter 
    """

    def __init__ (self, point = None, direction = None):
        """
        This class takes a point a direction 3 vector. Defaults are p = (0, 0 ,0) 
        and directoin = (0, 0, 1)

        This raises a type error if the inputs are 3-vectors.
        """ 
        if not point:
            point = [0., 0. ,0.]
        if not direction:
            direction = [0., 0., 1.]
        if len(point) != 3:
            raise TypeError("Line point must be a 3 vector")
        if len(direction) != 3:
            raise TypeError("Line direction must be a 3 vector")

        self.point = np.asarray(point)
        self.direction = np.asarray(direction)

    def get_point_on_line (self, t):
        """
        Get a point along the line by passing in an adjustable parameter t.
        """
        return self.point + t*self.direction
       
    def distance_to (self, other):
        """
        Find the minimum distance between two lines. Other must be of line type or a type error is raised.

        From https://en.wikipedia.org/wiki/Skew_lines#Distance 
        Retrieved 15 October 2018

        L = A + B*t (line 1)
        M = C + D*s (line 2)
        
        n = (B x D) / |B x D| 
        d = |n*(C-A)| 
        """ 
        if not isinstance(other, Line):
            raise TypeError ("distance_to measure distance between two lines.")

        nx = np.cross(self.direction, other.direction)
        mag = np.sqrt (np.dot(nx, nx))

        if mag > 0.0000001: 
            nx = nx / mag
        else: #case of parallel lines
            point_diff = self.point - other.point
            return np.sqrt(np.dot(point_diff, point_diff))
        
        projection = np.dot(nx, other.point - self.point)
        return np.sqrt (np.dot(projection, projection))
        
