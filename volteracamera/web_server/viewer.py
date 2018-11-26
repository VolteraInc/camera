from flask import Blueprint, render_template, request, jsonify
import logging
from io import BytesIO
import numpy as np

from ..analysis.laser_line_finder import LaserProcessingClient, LaserProcessingServer
from . import get_processor, get_cam
from ..control.camera import Camera

bp = Blueprint("viewer", __name__, url_prefix="/viewer")

t = 0
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
    laser_client = LaserProcessingClient()
    captured_points = laser_client.get_data()
    if not captured_points: 
        logging.debug("No new points are being sent to the client.")
        return None
    logging.debug("{} points are being sent to client.".format (len(captured_points)))
    return captured_points

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
    global t
    t = 0
    processor = get_processor()
    logging.debug("Processor start requested.")
    if not processor.is_alive():
        logging.debug ("Processor start sent.")
        processor.start()
    processor.restart()
    return "OK"

@bp.route('stop')
def stop_laser():
    """
    Stop collecting laser data.
    """
    processor = get_processor()
    processor.stop()
    logging.debug("Processor stop requested.")
    return "OK"