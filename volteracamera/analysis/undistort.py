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

        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        self.distortion = np.asarray(distortion, dtype=np.float32)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__dict__)

    def undistort_points (self, points ):
        """
        Undistort a single point.
        """
        return cv2.undistortPoints(np.asarray(points, dtype=np.float32), self.camera_matrix, self.distortion)

    def project_point_with_distortion_cv (self, points):
        """
        project points with distortion using opencv function.
        """
        out_point, _ = cv2.projectPoints(np.array([points]), rvec=np.array([0, 0, 0], dtype="float32"), tvec=np.array([0, 0, 0], dtype="float32"), cameraMatrix=self.camera_matrix, distCoeffs=self.distortion)
        return out_point[0][0]

    def project_point_with_distortion (self, points):
        """
        project points with distortion
        taken from https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        """
        k1, k2, p1, p2, k3 = self.distortion
        point = points / points[2]

        r2 = point[0]*point[0] + point[1]*point[1]

        radial_coeff = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2 
        xpp = point[0]*radial_coeff + 2.0*p1*point[0]*point[1] + p2*(r2 + 2.0*point[0]*point[0])  
        ypp = point[1]*radial_coeff + p1*(r2 + 2.0*point[1]*point[1]) + 2.0*p2*point[0]*point[1] 

        point[0] = xpp
        point[1] = ypp

        out_point = np.dot(self.camera_matrix, point)

        return np.asarray([out_point[0], out_point[1]])

    def undistort_image (self, image ):
        """
        Return an  image.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion)

    def _project_points (self, points_2d_undistorted ):
        """
        Return the ray in 3d given the 2 point.
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return [np.asarray([(point[0][0] - cx ) / fx, ( point[0][1] - cy ) / fy, 1.0]) for point in points_2d_undistorted]
            

    def get_rays_from_points (self, points_2d):
        """
        Given an image point, undistort and project it.
        """
        points_undistorted = self.undistort_points(points_2d)
        #return self._project_points(points_undistorted)
        points_undistorted = [[point[0][0], point[0][1], 1.0] for point in points_undistorted]
        return points_undistorted

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
