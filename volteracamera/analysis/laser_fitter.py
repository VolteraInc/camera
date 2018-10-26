"""
This module handles performing the laser plane fitting given a set of camera intrinsics 
and laser intersection points.

Since the exact location of the laser planes is unknown, the fitting must minimize for two parameters. The first is the offset of the calibrtion planes (for now assumed normal to the sensor) and the second is the transformation of that generates the laser plane. 

The minimization takes place in two steps. In the first, for each set of data, the line of intersection of the laser plane and the calibration plane is calculated. Then, for each ij point, the minimum distance between the intersection line and the laser plane is calculated. For a perfect fit, the distance between the ray and line should be 0 (they cross). The laser plane transform and the calibration plane offset are then iterated to give the minimal total offset of every point and every plane.

"""
import numpy as np
import transforms3d as tfd

from scipy.optimize import least_squares

from .undistort import Undistort
from .transform import Transform
from .plane import Plane
from .line import Line

DISTANCE_TO_SURFACE = 0.015
LAYER_THICKNESS = 0.000127
TAPE_LAYERS = np.arange (6, 0, -1) * LAYER_THICKNESS

STARTING_LASER_PLANE_NORMAL = [0, 0, 1.0]
STARTING_LASER_PLANE_POINT = [0, 0, 0]
MEASURED_PLANE_NORMAL = [0, 0, -1.0] #Assume the target planes are normal to the sensor.

INITIAL_ROTATION = [-np.pi*3/4, 0, 0]
INITIAL_POSITON = [0.0, -0.0075, 0.0075]

#parameter locations for each fitted parameter
LP_RX = 0
LP_RY = 1
LP_RZ = 2
LP_OFFSET_Z = 3
CAL_PLANE_RX = 4
CAL_PLANE_RY = 5
CAL_PLANE_RZ = 6
CAL_PLANE_OFFSET_Z = 7
INITIAL_PARAMS = [ np.sin(45 * np.pi / 180), 0, 0, -0.03, 0, 0, 0, -0.04 ]


class LaserFitter(object):
    """
    This class is used to fit a set of camera intrinsics and laser line positions at known heights.
    """

    def __init__(self, undistort, laser_intersections, relative_distances):
        """
        On construction, this class takes an Undistort class instance, a list of 
        lists of laser points (just vertical laser positions, the horizontal pixel 
        is inferred from the position in the list points) and a list of plane 
        distances for the measured points. The plane distances get smaller the 
        closer they get to the camera.
        """
        if not isinstance (undistort, Undistort):
            raise RuntimeError ("Must provide an instance of the undistortion class.")
        if len(laser_intersections) != len(relative_distances): 
            raise RuntimeError("The number point datasets must equal the number of plane intersection distances.")
        if len(relative_distances) < 2:
            raise RuntimeError("Need at least 2 datasets to fit a plane.")
        self.undistort = undistort
        self.rays = []
        for a_set in laser_intersections:
            points = [[[i, j]] for i, j in enumerate (a_set)]
            self.rays.append(undistort.get_rays_from_points(points))

        self.plane_distance = relative_distances

    @staticmethod
    def _calculate_residual(params, undistort, point_set, plane_offsets):
        """
        For a single i, j point, calculate the reprojection error and return the
        difference between the meausured point and the associated reprojected point.
        """

        laser_rot = [params[LP_RX], params[LP_RY], params[LP_RZ]]
        laser_point = [0, 0, params[LP_OFFSET_Z]]
        laser_transform = Transform (rotation = laser_rot, translation = laser_point)
        laser_plane = laser_transform.transform_plane(Plane()) 
        
        #planes_rot = [params[CAL_PLANE_RX], params[CAL_PLANE_RY], params[CAL_PLANE_RZ]]
        planes_rot = [0, 0, 0]
        planes_point = [0, 0, params[CAL_PLANE_OFFSET_Z]]
        planes_transform = Transform(rotation=planes_rot, translation = planes_point)
        target_planes = [planes_transform.transform_plane(Plane(point_on_plane=[0, 0, offset])) for offset in plane_offsets]
        intersection_lines = [laser_plane.intersection_line(plane) for plane in target_planes]
        #residual = 0.0
        residual = []
        for points, line in zip(point_set, intersection_lines):
            #residual += sum([line.distance_to( Line(direction=point) )**2 for point in points])
            residual.extend([line.distance_to( Line(direction=point) )**2 for point in points])
        return residual

         
    def process(self):
        """
        Run the minimization. 
        """
        res = least_squares (LaserFitter._calculate_residual, INITIAL_PARAMS, args=(self.undistort, self.rays, self.plane_distance), verbose=2)

        laser_rot = [res.x[LP_RX], res.x[LP_RY], res.x[LP_RZ]]
        laser_point = [0, 0, res.x[LP_OFFSET_Z]]
        laser_transform = Transform (rotation = laser_rot, translation = laser_point)
        laser_plane = laser_transform.transform_plane(Plane()) 
 
        return res, laser_plane
