from flask import Blueprint, render_template, request, jsonify

from io import BytesIO
import numpy as np

from ..analysis.laser_line_finder import LaserProcessingClient 
from . import get_processor

bp = Blueprint("viewer", __name__, url_prefix="/viewer")

t = 0
point_list = []
def generate_test_points ():
    """
    Method that generates a set of test points.
    """
    global t
    points = np.asarray([ 
                np.asarray( [ { "x": i, "y": j, "z": np.sin ( 2 * np.pi / 10 * (t+i+j) ), "i": 128+127*np.sin(2 * np.pi * (i+j))} for i in np.arange ( -5, 5, 0.1 ) ]) 
                for j in np.arange (-5, 5, 0.1) ]).flatten()  
    t += 0.1
    return list(points)

def generate_points():
    """
    Method to pull in points from the laser procesor.

    TODO: Needs to be updated when the motion control is set up.
    """
    global t
    global points
    laser_client = LaserProcessingClient()
    point_list.append (laser_client.get_data())
    return list(point_list)

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

@bp.route('start')
def start_laser():
    """
    Start collecting image data
    """
    processor = get_processor()
    processor.start()
    return "OK"

@bp.route('stop')
def stop_laser():
    """
    Stop collecting laser data.
    """
    processor = get_processor()
    processor.stop()
    return "OK"