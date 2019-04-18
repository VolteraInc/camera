from flask import jsonify, session, request
import numpy as np

from . import bp
from volteracamera.analysis.undistort import Undistort

@bp.route('/camera_calibration', methods=["GET", "POST"])
def camera_calibration():
    """
    return a camera calibration file in json.
    """
    if request.method == "GET":
        if 'camera_calibration' not in session:
            session['camera_calibration'] = Undistort(np.array([[5000, 0, 1641], [0, 5000, 1232], [0, 0, 1]]), np.zeros(5))
        return jsonify (session['camera_calibration'])
    else:
        json_object = request.get_json()
        if "__undistort__" in json_object and json_object['__undistort__']:
            session['camera_calibration'] = json_object
        return jsonify ({"success": True})
