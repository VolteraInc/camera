from flask import jsonify, session, request
import numpy as np
from pathlib import Path
import base64
import re
import cv2

from . import bp
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.point_extractor import extract_feature_position

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

@bp.route('/camera_find_point', methods=["POST"])
def find_camera_point ():
    """
    take an input image, find the dot feature and return it's i, j position.
    """
    json_object = request.get_json()
    if "image_data" not in json_object:
        return jsonify ({"success": False, "message":"No image data in request.", "point":[]})

    image_string = json_object["image_data"]
    image_string = image_string[image_string.find(',')+1:]
    image_base64 = base64.b64decode(image_string)
    nparr = np.fromstring(image_base64, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    
    mean_point = extract_feature_position (img)
    return jsonify({"success": True, "message":"", "point":[mean_point[0], mean_point[1]]})

@bp.route('/parse_camera_position', methods=["POST"])
def find_camera_position ():
    """
    Parse the image filename and return the position.
    """
    json_object = request.get_json()
    if "filename" not in json_object:
        return jsonify ({"success": False, "message": "No filename with position sent.", "position":[]})

    file_path = Path(json_object["filename"])
    file_stem = file_path.stem

    pos = re.findall(r'(\d+(?:\.\d+)?)', file_stem)

    return jsonify ({"success": True, "message": "", "position":[float(pos[1])/1000.0, float(pos[2])/1000.0, float(pos[3])/1000.0]})

