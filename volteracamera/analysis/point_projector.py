"""
This class takes performs the laser point projection. It takes an undistortion object
and a laser plane object, and then processes image points to points in space.
"""
from .line import Line
from .plane import Plane
from .undistort import Undisort

class PointProjector (object):
    """
    Projects image point to real space.
    """
    
    def __init__(self, undistorter, laser_plane):
        """
        This class requires an Undistort class and a laser plane to be defined before it can be used.
        A type error is raised if these classes aren't passed in correctly.
        """
        if not isinstance (undistorter, Undistort):
            raise TypeError ("PointProjector class requires undistortion object.")
        if not isinstance (laser_plane, Plane):
            raise TypeError ("PointProjector class requires a laser plane object.")

        self.undistort = undistorter
        self.laser_plane = laser_plane

    def project (self, point):
        """
        Project a point in sensor co-ords (i, j) onto a 3d point in space.
        """
        if len(point) != 2:
            raise TypeError ("Only 2d points can be projected.")
        
        ray = self.undistort.get_ray_from_point(point)
        line = Line (point = [0, 0, 0], direction=ray)
        return = self.laser_plane.intersection_point (line)   