from flask import jsonify, session, request
from pathlib import Path
import re
import base64
import cv2
import numpy as np

from . import bp
from volteracamera.analysis.plane import Plane
from volteracamera.analysis.laser_line_finder import LaserLineFinder, reject_outlier


@bp.route('/laser_calibration', methods=["GET", "POST"])
def laser_calibration():
    """
    return a plane calibration file in json.
    """
    if request.method == "GET":
        if 'laser_plane' not in session:
            session['laser_plane'] = Plane()
        return jsonify (session['laser_plane'])
    else:
        json_object = request.get_json()
        if "__plane__" in json_object and json_object['__plane__']:
            session['laser_plane'] = json_object
        return jsonify ({"success": True})

@bp.route('/find_laser_points', methods=["POST"])
def find_laser_points():
    json_object = request.get_json()
    if "image_data" not in json_object:
        return jsonify ({"success": False, "message":"No image data in request.", "points":[]})

    image_string = json_object["image_data"]
    image_string = image_string[image_string.find(',')+1:]
    image_base64 = base64.b64decode(image_string)
    nparr = np.fromstring(image_base64, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with LaserLineFinder() as finder:
        points = finder.process(image_gray)
    points = reject_outlier(np.asarray(points))
    points = [ [x, y] for x, y in enumerate(points) ]
    
    return jsonify({"success": True, "message":"", "points": points})

@bp.route('/parse_laser_height', methods=["POST"])
def find_laser_position ():
    """
    Parse the image filename and return the position.
    """
    json_object = request.get_json()
    if "filename" not in json_object:
        return jsonify ({"success": False, "message": "No filename with height sent.", "position":[]})

    file_path = Path(json_object["filename"])
    file_stem = file_path.stem

    try:
        pos = float(re.findall(r'(\d+(?:\.\d+)?)', file_stem)[1])

        return jsonify ({"success": True, "message": "", "height":pos})
    except:
        return jsonify ({"success": False, "message": "Could not parse the height from the filename", "height":0})


@bp.route('/preview_laser_solution', methods=["POST"])
def preview_laser_solution ():
    """
    Given the solution in the selection boxes, calculate the projected points positions, and return with the actual point positions.
    The javascript viewer will be used to preview the solution and allow for manual refinement.
    """
    return jsonify ({'success': False, 'message': "Method not implemented."})


@bp.route('/fit_laser', methods=['POST'])
def fit_laser():
    """
    This method takes all the points and heights passed to it through json and begins the fitting process.
    Eventualy this will be done in a second thread that allows us to query the status.
    """
    json_object = request.get_json()
    if "heights" not in json_object or "points" not in json_object or "camera_calibration" not in json_object or 'laser_calibration' not in json_object or "rvec" not in json_object or "tvec" not in json_object:
        return jsonify ({'success': False, 'message':"Invalid request from client."})

    position = [ point for point in json_object['heights'] ]
    points_2d = [ point for point in json_object['points'] ]
    camera_cal = json_object['camera_calibration']    
    guess_laser_cal = json_object['laser_calibration']
    rvec = json_object['rvec']
    tvec = json_object['tvec']

    return ({"success": False, "message":"Not Implemented"})
