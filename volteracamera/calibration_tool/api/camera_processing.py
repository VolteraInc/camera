from flask import jsonify, session, request
import numpy as np
from pathlib import Path
import base64
import re
import cv2

from . import bp
from volteracamera.analysis.undistort import Undistort
from volteracamera.analysis.point_extractor import extract_feature_position
from volteracamera.intrinsics.stage_calibration import calibrate_from_3d_points

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

    try:
        return jsonify ({"success": True, "message": "", "position":[float(pos[1])/1000.0, float(pos[2])/1000.0, float(pos[3])/1000.0]})
    except:
        return jsonify ({"success": False, "message": "Could not parse the position from the filename.", "position":[]})

@bp.route('/preview_camera_solution', methods=["POST"])
def preview_camera_solution ():
    """
    Given the solution in the selection boxes, calculate the projected points positions, and return with the actual point positions.
    The javascript viewer will be used to preview the solution and allow for manual refinement.
    """
    return jsonify ({'success': False, 'message': "Method not implemented."})

@bp.route('/fit_camera', methods=['POST'])
def fit_camera():
    """
    This method takes all the points and heights passed to it through json and begins the fitting process.
    Eventualy this will be done in a second thread that allows us to query the status.
    """
    json_object = request.get_json()
    print (json_object)
    if "positions" not in json_object or "points" not in json_object or "calibration" not in json_object or "rvec" not in json_object or "tvec" not in json_object:
        return jsonify ({'success': False, 'message':"Invalid request from client."})

    positions = [ [point[0], -point[1], point[2]] for point in json_object['positions'] ]
    points_2d = [ point for point in json_object['points'] ]
    guess_cal = json_object['calibration']    
    rvec = json_object['rvec']
    tvec = json_object['tvec']

    #print (positions)
    #print (points_2d)
    #print (guess_cal)
    #print (rvec)
    #print(tvec)
    with open("IntrinsicsPoints.csv", "w") as fid:
        for pos_2d, pos_3d in zip (points_2d, positions):
            fid.write(f"{pos_2d[0]}, {pos_2d[1]}, {pos_3d[0]}, {pos_3d[1]}, {pos_3d[2]}\n")

    print ("Starting Fit...")

    #undistort, _, _, rvec, tvec = calibrate_from_3d_points (position, points_2d, guess_cal["camera_matrix"], guess_cal["distortion"])#, rvec, tvec)

    print ("Finished Fit...")

    #return jsonify({"success": True, "message":"", "calibration": undistort, "rvec":rvec, "tvec":tvec})
    return jsonify({"success": True, "message":"", "calibration": Undistort(np.asarray(guess_cal["camera_matrix"]), np.asarray(guess_cal["distortion"])), "rvec":rvec, "tvec":tvec})
