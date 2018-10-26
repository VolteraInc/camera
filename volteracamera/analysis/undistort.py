"""
Given a camera model (camera matrix and a 5 point radtan undistortion model),
this class can be used to undistort points and lists of laser points, as well as
undistort images.
"""
import cv2
import json
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
        self.__undistort__ = True
        if not isinstance (camera_matrix, np.ndarray):
            raise TypeError ("camera_matix must be a numpy array")
        if not isinstance (distortion, np.ndarray) and not isinstance(distortion, list):
            raise TypeError ("distortion must be a numpy array or a list")
        if camera_matrix.shape != (3, 3):
            raise RuntimeError("Camera matrix has the wrong shape.")
        if len(distortion) != 5:
            raise RuntimeError("Distortion parameters are the wrong size.")

        self.camera_matrix = camera_matrix
        self.distortion = np.asarray(distortion)

    def _undistort_point (self, point_2d ):
        """
        Undistort a single point.
        """
        return cv2.undistortPoints(point_2d, self.camera_matrix, self.distortion)

    def undistort_image (self, image ):
        """
        Return an  image.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion)

    def _project_point (self, point_2d_undistorted ):
        """
        Return the ray in 3d given the 2 point.
        """
        if len(point_2d_undistorted) != 2:
            return TypeError("only a 2d point can be projected")
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return np.asarray([(u - cx ) / fx, ( v - cy ) / fy, 1.0])

    def get_ray_from_point (self, point_2d):
        """
        Given an image point, undistort and project it.
        """
        point_undistorted = self._undistort_point(point_2d)
        return self._project_point(point_undistorted)

    def write_file(self, filename):
        """
        save json file
        """
        with open(str(filename), 'w') as write_file:
            json.dump(self, write_file,
                      default=encode_undistort_settings,
                      indent=4,
                      sort_keys=True)

    @staticmethod
    def read_json_file(filename):
        """
        read the contents of a json file.
        """
        string_file = str(filename)
        with open(string_file, 'r') as read_file:
            return json.load(read_file, object_hook=decode_undistort_settings)

def encode_undistort_settings(settings):
    """
    encode the undistort object as a jason file.
    """
    #pylint: disable=consider-merging-isinstance
    if isinstance(settings, Undistort):
        return settings.__dict__
    elif isinstance(settings, np.ndarray):
        return settings.tolist()
    else:
        type_name = settings.__class__.__name__
        raise TypeError("Object of type '{type_name}' is not JSON" \
                         "serializable".format(type_name=type_name))

def decode_undistort_settings(settings):
    """
    decode the undistort object from a json file.
    """
    if '__undistort__' in settings:
        undistort = Undistort(
            camera_matrix = np.asarray(settings["camera_matrix"]),
            distortion = np.asarray(settings["distortion"]))
        return undistort

    return None 
