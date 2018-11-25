from flask import Blueprint, render_template, request, jsonify

from io import BytesIO
import numpy as np

bp = Blueprint("viewer", __name__, url_prefix="/viewer")

t = 0
def generate_points ():
    """
    Method that generates a set of points.
    """
    global t
    points = np.asarray([ 
                np.asarray( [ { "x": i, "y": j, "z": np.sin ( 2 * np.pi / 10 * (t+i+j) ), "i": 128+127*np.sin(2 * np.pi * (i+j))} for i in np.arange ( -5, 5, 0.1 ) ]) 
                for j in np.arange (-5, 5, 0.1) ]).flatten()  
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