"""
This class is used to represent a plane given a point on the plane and a normal.
"""
import numpy as np
import math
import json
from .line import Line

class Plane (object):
    """
    This class is used to represent a plane given a point on the plane and a surface normal.
    """

    def __init__(self, point_on_plane = None, normal = None):
        """
        constructor, default point at 0 and plane normal in the positive z direction
        """
        self.__plane__ = True
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
    
    def _normalize(self)->None:
        """
        Normalize the plane.
        """
        denominator = np.sqrt(np.dot(self.normal, self.normal))
        if math.isclose(denominator, 0.0):
            raise ZeroDivisionError("Normal is too close to 0.")
        self.normal = self.normal / denominator

    def intersection_line(self, other):
        """
        Find the intersection of two planes and return the line of intersection.

        Raises Runtime Error if the planes are parallel.
        Raises TypeError if input other is not a plane.

        To get the line direction, first take the cross product of the two normals:

        direction = n1 x n2

        To find a point on a line, we have to solve 

        a1*x + b1*y + c1*z + d1 = 0
        a2*x + b2*y + c2*z + d2 = 0

        In this program, planes are stored in point normal form, to get to the 
        algebraic for use:

        n = normal = (a, b, c)
        r = point on plane = (x, y, z)
        p = point in plane = (px, py, pz)
        n . (r - p) = 0
        a * (x - px) + b * (y - py) + c * (z - pz) = 0

        a*x + b*y + c*z - n . p = 0
        d = -n . p
        
        Where p = (x, y, z) is a point on the line. A third equation if needed 
        to solve for x, y, z. For our purpose here, we will choose the point where z=0. 
        In practice, the actual code will have two cases the solution where z=0 
        and the solution where x = 0 to handle the case where the line lies in or 
        parallel to the plane.

        We now have a linear equation, solve using Cremer's rule 

        dA = det(A) = a1*b2-b1*a2

        x = (-d1*b2+b1*d2)/dA
        y = (-a1*d2+d1*a2)/dA
        z = 0 

        or 

        dA = det(A) = b1*c2 - b2*c1
        x = 0
        y = (-d1*c2+d2*c1)/dA
        z = (-b1*d2+d1*b2)/dA

        For the purpose of laser scanning, these two cases are sufficient, any 
        line we produce will cross either the z or x axis (lasers won't run 
        vertical on the sensor)
        """
        if not isinstance (other, Plane):
            raise TypeError("Only plane object can be intersected with another plane.")

        direction = np.cross(self.normal, other.normal)
        dnorm = np.sqrt(np.dot(direction, direction))
        if math.isclose (dnorm, 0.0):
            raise RuntimeError("Two planes parallel, no line of intersection")

        a1, b1, c1 = self.normal
        d1 = - np.dot (self.normal, self.point)
        a2, b2, c2 = other.normal
        d2 = - np.dot (other.normal, other.point)

        zdirn = np.asarray([0., 0., 1.])
    
        dA_z_soln = a1*b2 - b1*a2    
        if not math.isclose(0, dA_z_soln):
            # z=0 solution is valid
            x = (-d1*b2+b1*d2)/dA_z_soln
            y = (-a1*d2+d1*a2)/dA_z_soln
            z = 0
            return Line(point=np.asarray([x, y, z]), direction=direction)
        
        dA_x_soln = b1*c2 - b2*c1   
        if not math.isclose(0, dA_x_soln):
            # x=0 solution is valid
            x = 0
            y = (-d1*c2 + d2*c1)/dA_x_soln
            z = (-b1*d2 + d1*b2)/dA_x_soln
            return Line(point=np.asarray([x, y, z]), direction=direction)

        dA_y_soln = a1*c2 - a2*c1
        x = (-d1*c2 + d2*c1)/dA_y_soln 
        y = 0
        z = (-a1*d2 + a2*d1)/dA_y_soln
        return Line(point=np.asarray([x, y, z]), direction=direction)


    def intersection_point(self, other):
        """
        Find the intersection point of the given line with the plane.
    
        Raises Runtime Error if the line and plane are perpendicular (no solution)
        Raises TypeError if other is not a line type.

        For a given line given by p0 + t*dirn = p, we can sub in to the plane equation and then solve for t.

        t = n . (pp - pl)/(n . dirn_line)
        """
        if not isinstance (other, Line):
            raise TypeError("Only a Line can intersect a plane at a point.")
        
        norm = np.dot (self.normal, other.direction)

        if math.isclose(norm, 0.0):
            raise RuntimeError ("Line and plane are perpendicular, no intersection.")

        t = np.dot( self.normal, (self.point - other.point) ) / norm

        return other.point + t*other.direction 

    def write_file(self, filename):
        """
        save json file
        """
        with open(str(filename), 'w') as write_file:
            json.dump(self, write_file,
                      default=encode_plane_settings,
                      indent=4,
                      sort_keys=True)   

    @staticmethod
    def read_json_file(filename):
        """
        read the contents of a json file.
        """
        string_file = str(filename)
        with open(string_file, 'r') as read_file:
            return json.load(read_file, object_hook=decode_plane_settings)

def encode_plane_settings(settings):
    """
    encode the plane object as a jason file.
    """
    #pylint: disable=consider-merging-isinstance
    if isinstance(settings, Plane):
        return settings.__dict__
    elif isinstance(settings, np.ndarray):
        return settings.tolist()
    else:
        type_name = settings.__class__.__name__
        raise TypeError("Object of type '{type_name}' is not JSON" \
                         "serializable".format(type_name=type_name))

def decode_plane_settings(settings):
    """
    decode the plane object from a json file.
    """
    if '__plane__' in settings:
        plane = Plane(
            point_on_plane = settings["point"],
            normal = settings["normal"])
        return plane

    return None 
