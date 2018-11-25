from flask import Blueprint, render_template, request, jsonify

from io import BytesIO
import numpy as np

bp = Blueprint("viewer", __name__, url_prefix="/viewer")

class Point (object):
    """
    Class for storing points
    """
    def __init__(x, y, z, intensity):
        """
        Initialize the point
        """
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity


t = 0
def generate_points ():
    """
    Method that generates a set of points.
    """
    points = np.asarray([ 
                np.asarray([ np.sin ( 2 * np.pi * 100 * (t+i+j) ) for i in np.range ( -5, 5, 0.1 ) ]) 
                for j in range (-5, 5, 0.1) ]).flatten()  
    t += 0.1
    return list(points)

#routes
@bp.route('/viewer')
def viewer():
    """
    Render the preview page.
    """
    return render_template("viewer/viewer.html")

@bp.route('points')
def points():
    """
    Return a set of points in json format
    """
    points = generate_points()
    return jsonify (points)