"""
This module handles performing the laser plane fitting given a set of camera intrinsics 
and laser intersection points.
"""
import numpy as np
import transforms3d as tfd

from .undistort import Undistort

DISTANCE_TO_SURFACE = 0.015
LAYER_THICKNESS = 0.000127
TAPE_LAYERS = np.arange (6, 0, -1) * LAYER_THICKNESS

STARTING_LASER_PLANE_NORMAL = [0, 0, 1.0]
STARTING_LASER_PLANE_POINT = [0, 0, 0]
MEASURED_PLANE_NORMAL = [0, 0, -1.0] #Assume the target planes are normal to the sensor.

INITIAL_ROTATION = [-np.pi*3/4, 0, 0]
INITIAL_POSITON = [0.0, -0.0075, 0.0075]

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
        if len(distortion) != 5:
            raise RuntimeError ("Not enough distortion parameters")
        if len(laser_intersections) != len(relative_distances): 
            raise RuntimeError("The number point datasets must equal the number of
                                plane intersection distances.")
        if len(relative_distances) < 2:
            raise RuntimeError("Need at least 2 datasets to fit a plane.")

        self.points = []
        for a_set in laser_intersections:
            points = [(i, j) for i, j in enumerate (a_set)]
            self.points.append(undistort.undistort_points(points))

        self.plane_distance = relative_distances

    def _calculate_residual(self):
        """
        For a single i, j point, calculate the reprojection error and return the
        difference between the meausred point and the associated reprojected point.
        """

    def process(self):
        """
        Run the minimization. 
        """

